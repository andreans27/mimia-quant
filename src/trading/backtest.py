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
from typing import Optional, Dict, List, Tuple

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_technical_features
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE,
    SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS,
)

# ─── Multi-TF Config ───────────────────────────────────────
TF_ORDER = ['15m', '30m', '1h', '4h']
TF_DUR = {'15m': timedelta(minutes=15), '30m': timedelta(minutes=30),
          '1h': timedelta(hours=1), '4h': timedelta(hours=4)}
TF_LIMIT = {'15m': 3, '30m': 6, '1h': 24, '4h': 96}
TF_PREFIX = {'15m': 'm15_', '30m': 'm30_', '1h': 'h1_', '4h': 'h4_'}
TF_RULE = {'15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h'}
WARMUP_BARS = 200


# ─── Helpers ────────────────────────────────────────────────

def get_last_complete_bar_start(T: datetime, tn: str) -> datetime:
    """Get the START timestamp of the last COMPLETE higher-TF bar at time T.
    
    A higher-TF bar at timestamp TS covers TS to TS+duration.
    It is COMPLETE at time TS+duration.
    At cutoff T, the last complete bar started at T-duration, rounded down.
    """
    T_minus = T - TF_DUR[tn]
    m = T_minus.minute
    if tn == '15m':
        return T_minus.replace(minute=(m // 15) * 15, second=0, microsecond=0)
    elif tn == '30m':
        return T_minus.replace(minute=(m // 30) * 30, second=0, microsecond=0)
    elif tn == '1h':
        return T_minus.replace(minute=0, second=0, microsecond=0)
    elif tn == '4h':
        return T_minus.replace(hour=(T_minus.hour // 4) * 4, minute=0, second=0, microsecond=0)
    return T_minus


def detect_signal(proba: float) -> int:
    if proba is None:
        return 0
    if proba >= THRESHOLD:
        return 1  # LONG
    if proba <= (1 - THRESHOLD):
        return -1  # SHORT
    return 0  # FLAT


# ─── Core Backtest ──────────────────────────────────────────

WalkForwardResult = Dict[str, object]
"""Return type for run_backtest. Keys:
    symbol, trades, win_rate, profit_factor, total_pnl,
    long_pnl, short_pnl, max_dd, elapsed, probas, signals,
    trade_details
"""


def run_backtest(symbol: str, test_hours: int = 24,
                 verbose: bool = False) -> Optional[WalkForwardResult]:
    """Run walk-forward backtest for one symbol.
    
    Args:
        symbol: Trading pair (e.g. 'ENAUSDT')
        test_hours: How many hours of recent data to test
        verbose: Print progress messages
    
    Returns:
        Dict with results, or None on failure
    """
    log_func = (lambda msg: print(f"  {msg}")) if verbose else (lambda _: None)
    t0 = ttime.time()
    log_func(f"{symbol} | {test_hours}h")

    # ── 1. Load models ──
    gen = SignalGenerator(symbol)
    cached = gen._load_models(symbol)
    if cached is None:
        log_func("❌ No models")
        return None
    model_groups = cached['groups']

    # ── 2. Fetch OHLCV ──
    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        log_func("❌ No data")
        return None

    # ── 3. Pre-compute features ──
    feats_5m = compute_technical_features(df_5m, prefix='m5_')
    close_arr = df_5m['close'].astype(float).values

    tf_feats_raw: Dict[str, pd.DataFrame] = {}
    for tn in TF_ORDER:
        td = df_5m.resample(TF_RULE[tn]).agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
        td['low'] = td[['open', 'close', 'low']].min(axis=1)
        td['high'] = td[['open', 'close', 'high']].max(axis=1)

        tf_feats_raw[tn] = compute_technical_features(td, prefix=TF_PREFIX[tn])

        # Cross-TF trend features (used by 'full' group models)
        tf_close = td['close'].astype(float)
        for period in [20, 50]:
            sma = tf_close.rolling(period).mean()
            trend = pd.Series(0, index=tf_close.index)
            trend[sma > sma.shift(1)] = 1
            trend[sma < sma.shift(1)] = -1
            tf_feats_raw[tn][f'{TF_PREFIX[tn]}trend_sma{period}'] = trend

    # Index model features for batch inference
    all_model_feats = sorted(set(
        f for _, models in model_groups.items()
        for _, m, mf in models for f in mf
    ))
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    n_mf = len(all_model_feats)

    # PyXGboost model references + pre-compiled feature indices
    model_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                model_refs.append((m, np.array(avail, dtype=np.int32)))

    # Numpy arrays for fast slicing
    m5_name_to_idx = {f: i for i, f in enumerate(feats_5m.columns)}
    m5_arr = feats_5m.values

    tf_names = {}
    tf_name_to_idx = {}
    tf_arrs = {}
    for tn in TF_ORDER:
        tf_names[tn] = list(tf_feats_raw[tn].columns)
        tf_name_to_idx[tn] = {f: i for i, f in enumerate(tf_names[tn])}
        tf_arrs[tn] = tf_feats_raw[tn].values

    # ── 4. Test range ──
    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    n_total = len(feats_5m)
    test_start = max(WARMUP_BARS, int(np.searchsorted(feats_5m.index, start_dt)))
    test_range = list(range(test_start, n_total))
    n_test = len(test_range)

    # ── 5. Pre-compute TF row mapping ──
    tf_indices_arr = {tn: tf_feats_raw[tn].index.values for tn in TF_ORDER}
    tf_row_map = np.zeros((n_test, len(TF_ORDER)), dtype=np.int32)
    for test_ii, bi in enumerate(test_range):
        bar_close = feats_5m.index[bi] + timedelta(minutes=5)
        for ti, tn in enumerate(TF_ORDER):
            gvs_ts = get_last_complete_bar_start(bar_close, tn)
            pos = int(np.searchsorted(tf_indices_arr[tn], gvs_ts, side='right')) - 1
            tf_row_map[test_ii, ti] = pos

    # ── 6. Build feature matrix ──
    feat_matrix = np.zeros((n_test, n_mf), dtype=np.float64)
    for test_ii, bi in enumerate(test_range):
        # 5m features
        for fname, fi in m5_name_to_idx.items():
            if fname in mf_to_idx:
                val = m5_arr[bi, fi]
                feat_matrix[test_ii, mf_to_idx[fname]] = (
                    0.0 if np.isnan(val) else np.clip(val, -10, 10)
                )
        # TF features
        for ti, tn in enumerate(TF_ORDER):
            row = int(tf_row_map[test_ii, ti])
            if row >= 0:
                for fn, fi in tf_name_to_idx[tn].items():
                    if fn in mf_to_idx:
                        val = tf_arrs[tn][row, fi]
                        feat_matrix[test_ii, mf_to_idx[fn]] = (
                            0.0 if np.isnan(val) else np.clip(val, -10, 10)
                        )
            elif test_ii > 0:
                for fn, fi in tf_name_to_idx[tn].items():
                    if fn in mf_to_idx:
                        feat_matrix[test_ii, mf_to_idx[fn]] = \
                            feat_matrix[test_ii - 1, mf_to_idx[fn]]

    # ── 7. Batch model inference ──
    all_probas = np.zeros(n_test, dtype=np.float64)
    n_valid = np.zeros(n_test, dtype=np.int32)
    for model, feat_idx in model_refs:
        try:
            preds = model.predict_proba(feat_matrix[:, feat_idx])[:, 1]
            all_probas += preds
            n_valid += 1
        except Exception:
            continue
    all_probas = np.where(n_valid > 0, all_probas / n_valid, 0.0)
    all_signals = np.array([detect_signal(p) for p in all_probas])

    # ── 8. Trade simulation (deferred entry) ──
    capital = INITIAL_CAPITAL
    peak_cap = INITIAL_CAPITAL
    pos = 0; ep = 0.0; eq = 0.0; hr = 0; cd = 0; eproba = 0.0
    entry_ts = None; entry_price_bt = 0.0
    ps = 0; pproba = 0.0  # pending from previous bar
    trade_details = []

    for test_ii, bi in enumerate(test_range):
        price = float(close_arr[bi])
        bar_ts = feats_5m.index[bi]
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
            eq = (capital * POSITION_PCT) / ep
            capital -= ep * eq * TAKER_FEE
            peak_cap = max(peak_cap, capital)
            pos = ps; eproba = pproba; hr = HOLD_BARS

        # Signal for NEXT bar
        ps = all_signals[test_ii]
        pproba = all_probas[test_ii]

    # ── 9. Metrics ──
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
        'signals': all_signals.tolist(),
        'timestamps': [feats_5m.index[bi] for bi in test_range],
        'trade_details': trade_details,
    }


def run_multi(symbols: Optional[List[str]] = None, test_hours: int = 24,
              verbose: bool = True) -> List[Tuple[str, WalkForwardResult]]:
    """Run backtest for multiple symbols, return sorted results."""
    if symbols is None:
        symbols = LIVE_SYMBOLS
    results = []
    for sym in symbols:
        r = run_backtest(sym, test_hours=test_hours, verbose=verbose)
        if r:
            results.append((sym, r))
    return results
