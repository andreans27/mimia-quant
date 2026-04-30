"""
Mimia — Walk-Forward Backtest Engine
======================================
Correct walk-forward backtest with multi-TF XGBoost ensemble.
DateTime-based TF completion (no look-ahead bias) + batch inference.

Usage:
  from src.trading.backtest import run_backtest
  result = run_backtest('ENAUSDT', test_hours=24)
"""
import time as ttime
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import (
    compute_5m_features_5tf, compute_technical_features,
    resample_to_timeframes, ensure_ohlcv_1h,
)
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE,
    SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS,
    get_symbol_threshold, get_symbol_hold_bars, get_dynamic_position_pct,
)

WARMUP_BARS = 200


# ─── Core Backtest ──────────────────────────────────────────

WalkForwardResult = Dict[str, object]



def run_backtest(symbol: str, test_hours: int = 24,
                 verbose: bool = False) -> Optional[WalkForwardResult]:
    """Run walk-forward backtest for one symbol."""
    log_func = (lambda msg: print(f"  {msg}")) if verbose else (lambda _: None)
    t0 = ttime.time()
    log_func(f"{symbol} | {test_hours}h")

    gen = SignalGenerator(symbol)
    cached = gen._load_models(symbol)
    if cached is None:
        log_func("❌ No models")
        return None
    model_groups = cached['groups']

    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        log_func("❌ No data")
        return None

    df_1h = ensure_ohlcv_1h(symbol, min_days=20)
    if df_1h is not None:
        log_func(f"  1h: {len(df_1h)} bars (direct from Binance)")

    feat_df = compute_5m_features_5tf(df_5m, for_inference=True, df_1h=df_1h, symbol=symbol)
    if feat_df is None or len(feat_df) < 500:
        log_func("❌ Feature computation failed")
        return None
    log_func(f"✅ {len(feat_df)} feature rows | {len(feat_df.columns)} features")
    
    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values

    all_model_feats = sorted(set(
        f for _, models in model_groups.items()
        for _, m, mf in models for f in mf
    ))
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    n_mf = len(all_model_feats)

    feat_cols = list(feat_df.columns)
    feat_to_col = {f: i for i, f in enumerate(feat_cols)}

    missing_feats = [f for f in all_model_feats if f not in feat_to_col]
    if missing_feats:
        log_func(f"⚠️ {len(missing_feats)} model features not in feat_df (will be 0.0)")

    long_refs = []
    short_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                arr = np.array(avail, dtype=np.int32)
                ref = (m, arr)
                if tg == 'long':
                    long_refs.append(ref)
                elif tg == 'short':
                    short_refs.append(ref)

    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    feat_index = feat_df.index
    test_start = max(WARMUP_BARS, int(np.searchsorted(feat_index, start_dt)))
    test_range = list(range(test_start, len(feat_index)))
    n_test = len(test_range)

    if n_test == 0:
        log_func("⚠️ No bars in test window")
        return None

    feat_np = feat_df.values
    feat_slice = feat_np[test_start:test_start + n_test]
    feat_matrix = np.zeros((n_test, n_mf), dtype=np.float64)
    
    for mf_name, mf_pos in mf_to_idx.items():
        if mf_name in feat_to_col:
            col_pos = feat_to_col[mf_name]
            feat_matrix[:, mf_pos] = np.clip(
                np.nan_to_num(feat_slice[:, col_pos], nan=0.0), -10, 10
            )

    def _batch_infer(refs, feat_matrix):
        probas = np.zeros(n_test, dtype=np.float64)
        nv = np.zeros(n_test, dtype=np.int32)
        for model, feat_idx in refs:
            try:
                preds = model.predict_proba(feat_matrix[:, feat_idx])[:, 1]
                probas += preds
                nv += 1
            except Exception:
                continue
        return np.where(nv > 0, probas / nv, 0.0)

    def _apply_cal(probas, symbol, side):
        cal_path = Path("data/ml_models") / f"{symbol}_{side}_calibrator.json"
        if cal_path.exists():
            try:
                import json
                with open(cal_path) as f:
                    cal = json.load(f)
                z = cal['coef'] * probas + cal['intercept']
                return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)
            except Exception:
                pass
        return probas

    long_probas = _batch_infer(long_refs, feat_matrix)
    long_probas = _apply_cal(long_probas, symbol, 'long')
    
    short_probas = _batch_infer(short_refs, feat_matrix)
    short_probas = _apply_cal(short_probas, symbol, 'short')

    all_signals = np.zeros(n_test, dtype=np.int32)
    sym_threshold = get_symbol_threshold(symbol)
    long_mask = long_probas >= sym_threshold
    short_mask = short_probas >= sym_threshold
    both = long_mask & short_mask
    pick_long = long_probas >= short_probas
    all_signals[both & pick_long] = 1
    all_signals[both & ~pick_long] = -1
    all_signals[long_mask & ~short_mask] = 1
    all_signals[short_mask & ~long_mask] = -1

    all_probas = np.where(all_signals == 1, long_probas,
                          np.where(all_signals == -1, short_probas,
                                   np.maximum(long_probas, short_probas)))

    capital = INITIAL_CAPITAL
    peak_cap = INITIAL_CAPITAL
    pos = 0; ep = 0.0; eq = 0.0; hr = 0; cd = 0; eproba = 0.0
    entry_ts = None; entry_price_bt = 0.0
    ps = 0; pproba = 0.0
    trade_details = []

    for test_ii, bi in enumerate(test_range):
        price = float(close_arr[bi])
        bar_ts = feat_df.index[bi]
        hr = max(0, hr - 1)
        cd = max(0, cd - 1)

        if pos != 0 and hr <= 0:
            xp = price * (1 - SLIPPAGE) if pos == 1 else price * (1 + SLIPPAGE)
            raw = eq * (xp - ep) if pos == 1 else eq * (ep - xp)
            xc = xp * eq * TAKER_FEE
            net = raw - xc
            capital += raw - xc
            peak_cap = max(peak_cap, capital)
            trade_details.append({
                'direction': 'LONG' if pos == 1 else 'SHORT',
                'entry_proba': eproba, 'pnl': net,
                'entry_time': entry_ts.isoformat() if entry_ts else None,
                'exit_time': bar_ts.isoformat(),
                'entry_price': entry_price_bt,
                'exit_price': float(xp),
            })
            pos = 0; cd = COOLDOWN_BARS

        if ps != 0 and pos == 0 and cd <= 0:
            slip = (1 + SLIPPAGE) if ps == 1 else (1 - SLIPPAGE)
            ep = price * slip
            entry_price_bt = ep
            entry_ts = bar_ts
            pos_size_pct = get_dynamic_position_pct(pproba, symbol)
            eq = (capital * pos_size_pct) / ep
            capital -= ep * eq * TAKER_FEE
            peak_cap = max(peak_cap, capital)
            pos = ps; eproba = pproba; hr = get_symbol_hold_bars(symbol)

        ps = all_signals[test_ii]
        pproba = all_probas[test_ii]

    elapsed = ttime.time() - t0
    nt = len(trade_details)
    w_ = sum(1 for t in trade_details if t['pnl'] > 0)
    wr = (w_ / nt * 100) if nt else 0
    tp = sum(t['pnl'] for t in trade_details)
    gp_ = sum(t['pnl'] for t in trade_details if t['pnl'] > 0)
    gl_ = abs(sum(t['pnl'] for t in trade_details if t['pnl'] <= 0))
    pf = gp_ / gl_ if gl_ else float('inf')
    lp = sum(t['pnl'] for t in trade_details if t['direction'] == 'LONG')
    sp_ = sum(t['pnl'] for t in trade_details if t['direction'] == 'SHORT')

    running = INITIAL_CAPITAL; rpeak = INITIAL_CAPITAL; mdd = 0.0
    for t in trade_details:
        running += t['pnl']
        rpeak = max(rpeak, running)
        mdd = max(mdd, (rpeak - running) / rpeak * 100)

    return {
        'symbol': symbol, 'n_trades': nt, 'win_rate': wr,
        'profit_factor': pf, 'total_pnl': tp,
        'long_pnl': lp, 'short_pnl': sp_,
        'max_dd': mdd, 'elapsed': elapsed,
        'probas': all_probas.tolist(),
        'long_probas': long_probas.tolist(),
        'short_probas': short_probas.tolist(),
        'signals': all_signals.tolist(),
        'timestamps': [feat_df.index[bi] for bi in test_range],
        'trade_details': trade_details,
    }


def run_multi(symbols: Optional[List[str]] = None, test_hours: int = 24,
              verbose: bool = True) -> List[Tuple[str, WalkForwardResult]]:
    if symbols is None:
        symbols = LIVE_SYMBOLS
    results = []
    for sym in symbols:
        r = run_backtest(sym, test_hours=test_hours, verbose=verbose)
        if r:
            results.append((sym, r))
    return results


# ─── Live-Aligned Backtest (per-bar, no look-ahead, optimized) ──────────

_5M_FEATS_CACHE: Dict[str, pd.DataFrame] = {}


def _get_5m_features_cached(df_5m: pd.DataFrame, symbol: str = '') -> pd.DataFrame:
    """Compute 5m features once per df_5m (cached by symbol to avoid id reuse issues)."""
    # Use symbol name as cache key when available
    key = symbol if symbol else str(id(df_5m))
    if key not in _5M_FEATS_CACHE:
        _5M_FEATS_CACHE[key] = compute_technical_features(df_5m, prefix='m5_')
    return _5M_FEATS_CACHE[key]


def run_backtest_live_aligned(
    symbol: str,
    test_hours: int = 24,
    verbose: bool = False,
) -> Optional[Dict]:
    """Run per-bar backtest where each bar's features are computed ONLY from
    data available at that point in time. Simulates live trading exactly.

    Optimized: 5m features pre-computed once (no per-bar recomputation).
    """
    log_func = (lambda msg: print(f"  {msg}")) if verbose else (lambda _: None)
    t0 = ttime.time()
    log_func(f"{symbol} | {test_hours}h (live-aligned, optimized)")

    gen = SignalGenerator(symbol)
    cached = gen._load_models(symbol)
    if cached is None:
        log_func("❌ No models")
        return None
    model_groups = cached['groups']

    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        log_func("❌ No data")
        return None

    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    test_start = max(WARMUP_BARS, int(np.searchsorted(df_5m.index, start_dt)))
    test_end = len(df_5m)
    test_range = list(range(test_start, test_end))
    n_test = len(test_range)
    log_func(f"✅ {n_test} test bars ({df_5m.index[test_start]} → {df_5m.index[-1]})")

    if n_test == 0:
        log_func("⚠️ No bars in test window")
        return None

    # Pre-compute 5m features ONCE (they're trailing-only, no look-ahead issue)
    feats_5m = _get_5m_features_cached(df_5m, symbol)
    log_func(f"  5m features: {len(feats_5m.columns)} cols, {len(feats_5m)} rows")

    # Pre-compute 1h OHLCV DIRECTLY from Binance (NO resample look-ahead)
    full_1h_ohlcv = ensure_ohlcv_1h(symbol, min_days=20)
    if full_1h_ohlcv is None or len(full_1h_ohlcv) < 10:
        log_func("❌ No 1h OHLCV data")
        return None
    n_1h_total = len(full_1h_ohlcv)
    
    # Pre-compute 1h features ONCE (all bars) — per-bar slice will select available
    # This ensures rolling indicators (RSI-14, ATR-14, etc.) are computed from
    # the FULL 1h history, not just the first few incomplete bars.
    full_1h_feats = compute_technical_features(full_1h_ohlcv, prefix='h1_')
    log_func(f"  1h OHLCV: {n_1h_total} bars + {len(full_1h_feats.columns)} feats (direct, no look-ahead)")
    
    # Pre-compute FULL feature matrix (ALL 220 features including market, micro, cross-TF)
    # Then override h1 features per bar with correctly-sliced versions
    full_feat_matrix = compute_5m_features_5tf(
        df_5m, for_inference=True, df_1h=full_1h_ohlcv, symbol=symbol
    )
    if full_feat_matrix is None or len(full_feat_matrix) < 500:
        log_func("❌ Full feature matrix failed")
        return None
    log_func(f"  Full feat matrix: {len(full_feat_matrix)} rows, {len(full_feat_matrix.columns)} cols")
    
    # Identify h1 column prefixes for override
    h1_prefixes = tuple(c for c in full_feat_matrix.columns if c.startswith('h1_'))
    h1_prefixes += tuple(c for c in full_feat_matrix.columns if c in (
        '1h_bullish', 'vol_ratio_5m_vs_1h', 'rsi_div_5m_vs_1h',
        'vol_5m_vs_1h_avg', 'consolidation_ratio', 'is_consolidating',
    ))

    all_model_feats = sorted(set(
        f for _, models in model_groups.items()
        for _, m, mf in models for f in mf
    ))
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    n_mf = len(all_model_feats)

    long_refs = []
    short_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                arr = np.array(avail, dtype=np.int32)
                ref = (m, arr)
                if tg == 'long':
                    long_refs.append(ref)
                elif tg == 'short':
                    short_refs.append(ref)

    def _infer(refs, feat_row_np):
        probas = []
        for model, feat_idx in refs:
            try:
                preds = model.predict_proba(feat_row_np[feat_idx].reshape(1, -1))[:, 1]
                probas.append(preds[0])
            except Exception:
                continue
        return float(np.mean(probas)) if probas else 0.5

    def _apply_cal(proba, side):
        cal_path = Path("data/ml_models") / f"{symbol}_{side}_calibrator.json"
        if cal_path.exists():
            try:
                import json
                with open(cal_path) as f:
                    cal = json.load(f)
                z = cal['coef'] * proba + cal['intercept']
                return float(np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0))
            except Exception:
                pass
        return proba

    # ── Per-bar inference ──
    timestamps = []
    signals_list = []
    long_probas_list = []
    short_probas_list = []

    for ii, bi in enumerate(test_range):
        bar_ts = df_5m.index[bi]
        if verbose and ii % 50 == 0:
            print(f"  Bar {ii}/{n_test} ({bar_ts})")

        # --- Efficient per-bar feature computation ---
        # 1. How many complete 1h bars at this calendar time?
        #    Use FULL 1h OHLCV from Binance — each 1h bar is complete at close_time
        #    CRITICAL: bi // 12 is WRONG when 5m data doesn't start at hour boundary
        #    Instead: count 1h bars with close_time <= bar_ts
        n_complete_1h = int(np.searchsorted(
            full_1h_ohlcv.index + timedelta(hours=1),
            bar_ts,
            side='right'
        ))
        # 2. Worst case: also need warmup. ffilled features need at least some.
        #    If < 1 complete 1h bar: all 1h features are NaN → skip
        if n_complete_1h < 1:
            timestamps.append(bar_ts)
            signals_list.append(0)
            long_probas_list.append(0.5)
            short_probas_list.append(0.5)
            continue

        # 3. Select 1h features from the LAST complete 1h bar
        #    (computed from FULL 1h history — correct RSI, ATR, etc.)
        if n_complete_1h - 1 >= len(full_1h_feats):
            timestamps.append(bar_ts)
            signals_list.append(0)
            long_probas_list.append(0.5)
            short_probas_list.append(0.5)
            continue
        
        # Get ALL features from pre-computed matrix (5m, market, micro, cross-TF)
        if bar_ts not in full_feat_matrix.index:
            timestamps.append(bar_ts)
            signals_list.append(0)
            long_probas_list.append(0.5)
            short_probas_list.append(0.5)
            continue
        combined = full_feat_matrix.loc[[bar_ts]].copy()
        
        # Override h1 features with correctly-sliced per-bar version
        feats_1h_aligned = full_1h_feats.iloc[[n_complete_1h - 1]].copy()
        feats_1h_aligned.index = [bar_ts]
        for col in h1_prefixes:
            if col in feats_1h_aligned.columns:
                combined[col] = feats_1h_aligned[col].values[0]
            elif col in combined.columns:
                # Column exists in full_feat_matrix but not in h1_feats — set to NaN
                # (it's a cross-TF feature that depends on 1h data)
                # Recompute it from available_1h data
                available_1h_close = full_1h_ohlcv.iloc[:n_complete_1h]['close'].astype(float)
                available_1h_candle = full_1h_ohlcv.iloc[:n_complete_1h]
                
                if col == 'vol_ratio_5m_vs_1h':
                    atr_5m = feats_5m.loc[bar_ts, 'm5_atr_14'] if 'm5_atr_14' in feats_5m.columns else np.nan
                    atr_1h = feats_1h_aligned['h1_atr_14'].values[0] if 'h1_atr_14' in feats_1h_aligned.columns else np.nan
                    if not pd.isna(atr_1h) and atr_1h != 0:
                        combined[col] = atr_5m / atr_1h
                elif col == 'rsi_div_5m_vs_1h':
                    rsi_5m = feats_5m.loc[bar_ts, 'm5_rsi_14'] if 'm5_rsi_14' in feats_5m.columns else np.nan
                    rsi_1h = feats_1h_aligned['h1_rsi_14'].values[0] if 'h1_rsi_14' in feats_1h_aligned.columns else np.nan
                    combined[col] = rsi_5m - rsi_1h
                elif col == '1h_bullish':
                    if len(available_1h_close) >= 20:
                        sma20 = available_1h_close.rolling(20).mean()
                        if len(sma20.dropna()) > 0:
                            combined[col] = 1.0 if available_1h_close.iloc[-1] > sma20.iloc[-1] else 0.0
                elif col == 'consolidation_ratio':
                    range_5m = (df_5m['high'].astype(float).loc[:bar_ts].rolling(20).max() 
                               - df_5m['low'].astype(float).loc[:bar_ts].rolling(20).min())
                    if len(available_1h_candle) > 0:
                        range_1h = (available_1h_candle['high'].astype(float).rolling(20).mean() 
                                   - available_1h_candle['low'].astype(float).rolling(20).mean())
                        if not pd.isna(range_5m.iloc[-1]) and not pd.isna(range_1h.iloc[-1]) and range_1h.iloc[-1] != 0:
                            combined[col] = range_5m.iloc[-1] / range_1h.iloc[-1]
                elif col == 'is_consolidating':
                    range_5m = (df_5m['high'].astype(float).loc[:bar_ts].rolling(20).max() 
                               - df_5m['low'].astype(float).loc[:bar_ts].rolling(20).min())
                    if not pd.isna(range_5m.iloc[-1]):
                        combined[col] = float(range_5m.iloc[-1] < range_5m.rolling(40).mean().iloc[-1] * 0.5)
                elif col == 'vol_5m_vs_1h_avg':
                    volume_5m = df_5m['volume'].astype(float).loc[:bar_ts]
                    if len(volume_5m) >= 12:
                        vol_avg = volume_5m.rolling(12).sum().resample('1h').mean()
                        if len(vol_avg.dropna()) > 0 and vol_avg.iloc[-1] != 0:
                            combined[col] = volume_5m.iloc[-1] / vol_avg.iloc[-1]
                elif col.startswith('h1_trend_sma'):
                    period = int(col.replace('h1_trend_sma', ''))
                    if len(available_1h_close) > 0:
                        sma = available_1h_close.rolling(period).mean()
                        if len(sma.dropna()) > 0:
                            trend = 0
                            if sma.iloc[-1] > sma.shift(1).iloc[-1] if not pd.isna(sma.shift(1).iloc[-1]) else False:
                                trend = 1
                            elif sma.iloc[-1] < sma.shift(1).iloc[-1] if not pd.isna(sma.shift(1).iloc[-1]) else False:
                                trend = -1
                            combined[col] = trend

        # ── Build model input ──
        feat_cols = list(combined.columns)
        row_np = np.zeros(n_mf, dtype=np.float64)
        for mf_name, mf_pos in mf_to_idx.items():
            if mf_name in feat_cols:
                val = combined[mf_name].values[0]
                row_np[mf_pos] = float(np.clip(0.0 if np.isnan(val) else val, -10, 10))

        # ── Infer ──
        long_proba = _apply_cal(_infer(long_refs, row_np), 'long')
        short_proba = _apply_cal(_infer(short_refs, row_np), 'short')

        threshold = get_symbol_threshold(symbol)
        if long_proba >= threshold and long_proba >= short_proba:
            signal = 1
        elif short_proba >= threshold:
            signal = -1
        else:
            signal = 0

        timestamps.append(bar_ts)
        signals_list.append(signal)
        long_probas_list.append(long_proba)
        short_probas_list.append(short_proba)

        if verbose:
            log_func(f"  {bar_ts}: signal={signal:>2d} long={long_proba:.4f} short={short_proba:.4f}")

    elapsed = ttime.time() - t0
    log_func(f"✅ Done in {elapsed:.1f}s | {len(timestamps)} bars")

    # ── Trade Simulation (deferred entry, same as run_backtest) ──
    close_map = df_5m['close'].astype(float)
    capital = INITIAL_CAPITAL
    peak_cap = INITIAL_CAPITAL
    pos = 0; ep = 0.0; eq = 0.0; hr = 0; cd = 0; eproba = 0.0
    entry_ts = None; entry_price_bt = 0.0
    ps = 0; pproba = 0.0
    trade_details = []

    for ii in range(len(timestamps)):
        bar_ts = timestamps[ii]
        if bar_ts not in close_map.index:
            continue
        price = float(close_map.loc[bar_ts])
        hr = max(0, hr - 1)
        cd = max(0, cd - 1)

        # Exit
        if pos != 0 and hr <= 0:
            xp = price * (1 - SLIPPAGE) if pos == 1 else price * (1 + SLIPPAGE)
            raw = eq * (xp - ep) if pos == 1 else eq * (ep - xp)
            xc = xp * eq * TAKER_FEE
            net = raw - xc
            capital += raw - xc
            peak_cap = max(peak_cap, capital)
            trade_details.append({
                'direction': 'LONG' if pos == 1 else 'SHORT',
                'entry_proba': eproba, 'pnl': net,
                'entry_time': entry_ts.isoformat() if entry_ts else None,
                'exit_time': bar_ts.isoformat(),
                'entry_price': entry_price_bt,
                'exit_price': float(xp),
            })
            pos = 0; cd = COOLDOWN_BARS

        # Entry (deferred — from previous bar's signal)
        if ps != 0 and pos == 0 and cd <= 0:
            slip = (1 + SLIPPAGE) if ps == 1 else (1 - SLIPPAGE)
            ep = price * slip
            entry_price_bt = ep
            entry_ts = bar_ts
            pos_size_pct = get_dynamic_position_pct(pproba, symbol)
            eq = (capital * pos_size_pct) / ep
            capital -= ep * eq * TAKER_FEE
            peak_cap = max(peak_cap, capital)
            pos = ps; eproba = pproba; hr = get_symbol_hold_bars(symbol)

        # Signal for NEXT bar
        ps = signals_list[ii]
        pproba = long_probas_list[ii] if signals_list[ii] == 1 else short_probas_list[ii]

    # ── Metrics ──
    nt = len(trade_details)
    w_ = sum(1 for t in trade_details if t['pnl'] > 0)
    wr = (w_ / nt * 100) if nt else 0
    tp = sum(t['pnl'] for t in trade_details)
    gp_ = sum(t['pnl'] for t in trade_details if t['pnl'] > 0)
    gl_ = abs(sum(t['pnl'] for t in trade_details if t['pnl'] <= 0))
    pf = gp_ / gl_ if gl_ else float('inf')
    lp = sum(t['pnl'] for t in trade_details if t['direction'] == 'LONG')
    sp_ = sum(t['pnl'] for t in trade_details if t['direction'] == 'SHORT')

    running = INITIAL_CAPITAL; rpeak = INITIAL_CAPITAL; mdd = 0.0
    for t in trade_details:
        running += t['pnl']
        rpeak = max(rpeak, running)
        mdd = max(mdd, (rpeak - running) / rpeak * 100)

    return {
        'symbol': symbol,
        'timestamps': timestamps,
        'signals': signals_list,
        'long_probas': long_probas_list,
        'short_probas': short_probas_list,
        'n_trades': nt, 'win_rate': wr,
        'profit_factor': pf, 'total_pnl': tp,
        'long_pnl': lp, 'short_pnl': sp_,
        'max_dd': mdd,
        'trade_details': trade_details,
        'elapsed': elapsed,
    }


def run_multi_live_aligned(
    symbols: Optional[List[str]] = None,
    test_hours: int = 24,
    verbose: bool = True,
) -> List[Tuple[str, Dict]]:
    if symbols is None:
        symbols = LIVE_SYMBOLS
    results = []
    for sym in symbols:
        r = run_backtest_live_aligned(sym, test_hours=test_hours, verbose=verbose)
        if r:
            results.append((sym, r))
    return results

