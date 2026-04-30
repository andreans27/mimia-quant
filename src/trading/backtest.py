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
    resample_to_timeframes,
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

    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
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

    # Pre-compute 1h OHLCV from full data (for reference, will subset per bar)
    tf_data_full = resample_to_timeframes(df_5m, ['1h'])
    full_1h_ohlcv = tf_data_full['1h']
    log_func(f"  1h OHLCV: {len(full_1h_ohlcv)} bars from full data")

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
        # 1. How many complete 1h bars at bar index bi?
        n_complete_1h = bi // 12  # 12 5m bars = 1 complete 1h bar
        # 2. Worst case: also need warmup. ffilled features need at least some.
        #    If < 1 complete 1h bar: all 1h features are NaN → skip
        if n_complete_1h < 1:
            timestamps.append(bar_ts)
            signals_list.append(0)
            long_probas_list.append(0.5)
            short_probas_list.append(0.5)
            continue

        # 3. Pre-compute 1h features from available 1h bars
        #    (only bars 0 to n_complete_1h-1 are available)
        available_1h = full_1h_ohlcv.iloc[:n_complete_1h].copy()
        if len(available_1h) == 0:
            timestamps.append(bar_ts)
            signals_list.append(0)
            long_probas_list.append(0.5)
            short_probas_list.append(0.5)
            continue

        # 4. Compute 1h features (fast — only on N available bars)
        feats_1h = compute_technical_features(available_1h, prefix='h1_')
        # 5. Align 1h to this bar's 5m index (ffill)
        idx_target = pd.DatetimeIndex([bar_ts])
        feats_1h_aligned = feats_1h.reindex(idx_target, method='ffill', limit=12)

        # 6. Combine 5m features (pre-computed) + 1h features
        combined = pd.concat([feats_5m.loc[[bar_ts]], feats_1h_aligned], axis=1)

        # 7. Cross-TF features (matching compute_5m_features_5tf exactly)
        # Use data available at this point in time
        close_5m_bar = df_5m['close'].astype(float).loc[:bar_ts]  # 5m data up to now
        volume_5m_bar = df_5m['volume'].astype(float).loc[:bar_ts]
        close_1h_bar = available_1h['close'].astype(float)
        
        if 'h1_atr_14' in feats_1h_aligned.columns:
            atr_5m = feats_5m.loc[bar_ts, 'm5_atr_14']
            atr_1h = feats_1h_aligned['h1_atr_14'].values[0]
            combined['vol_ratio_5m_vs_1h'] = atr_5m / atr_1h if not pd.isna(atr_1h) and atr_1h != 0 else np.nan
        
        if 'h1_rsi_14' in feats_1h_aligned.columns:
            rsi_5m = feats_5m.loc[bar_ts, 'm5_rsi_14']
            rsi_1h = feats_1h_aligned['h1_rsi_14'].values[0]
            combined['rsi_div_5m_vs_1h'] = rsi_5m - rsi_1h
        
        # 1h Trend regime
        if len(close_1h_bar) > 0:
            for period in [20, 50]:
                sma = close_1h_bar.rolling(period).mean()
                if len(sma.dropna()) > 0:
                    trend_val = 0
                    if sma.iloc[-1] > sma.shift(1).iloc[-1] if not pd.isna(sma.shift(1).iloc[-1]) else False:
                        trend_val = 1
                    elif sma.iloc[-1] < sma.shift(1).iloc[-1] if not pd.isna(sma.shift(1).iloc[-1]) else False:
                        trend_val = -1
                    combined[f'h1_trend_sma{period}'] = trend_val
        
        # Volume ratio: 5m volume vs 1h average volume
        if len(volume_5m_bar) >= 12:
            vol_1h_avg = volume_5m_bar.rolling(12).sum().resample('1h').mean()
            if len(vol_1h_avg.dropna()) > 0 and not pd.isna(vol_1h_avg.iloc[-1]):
                combined['vol_5m_vs_1h_avg'] = volume_5m_bar.iloc[-1] / vol_1h_avg.iloc[-1] if vol_1h_avg.iloc[-1] != 0 else np.nan
        
        # Consolidation detection
        high_5m_bar = df_5m['high'].astype(float).loc[:bar_ts]
        low_5m_bar = df_5m['low'].astype(float).loc[:bar_ts]
        range_5m_20_bar = (high_5m_bar.rolling(20).max() - low_5m_bar.rolling(20).min())
        if len(available_1h) > 0:
            range_1h_20_bar = (available_1h['high'].astype(float).rolling(20).mean() 
                              - available_1h['low'].astype(float).rolling(20).mean())
            if not pd.isna(range_5m_20_bar.iloc[-1]) and not pd.isna(range_1h_20_bar.iloc[-1]) and range_1h_20_bar.iloc[-1] != 0:
                combined['consolidation_ratio'] = range_5m_20_bar.iloc[-1] / range_1h_20_bar.iloc[-1]
            if not pd.isna(range_5m_20_bar.iloc[-1]):
                combined['is_consolidating'] = float(range_5m_20_bar.iloc[-1] < range_5m_20_bar.rolling(40).mean().iloc[-1] * 0.5)
        
        # 1h bullish alignment
        if len(close_1h_bar) >= 20:
            sma20_1h = close_1h_bar.rolling(20).mean()
            if len(sma20_1h.dropna()) > 0:
                combined['1h_bullish'] = 1.0 if close_1h_bar.iloc[-1] > sma20_1h.iloc[-1] else 0.0

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

