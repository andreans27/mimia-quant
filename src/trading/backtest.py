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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE,
    SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS,
)

WARMUP_BARS = 200


# ─── Helpers ────────────────────────────────────────────────

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

    # ── 3. Compute features (same pipeline as live engine) ──
    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
    if feat_df is None or len(feat_df) < 500:
        log_func("❌ Feature computation failed")
        return None
    log_func(f"✅ {len(feat_df)} feature rows | {len(feat_df.columns)} features")
    
    # Align close prices to feat_df's index (feat_df index is subset of df_5m index)
    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values

    # ── 4. Build model refs + feature mapping ──
    all_model_feats = sorted(set(
        f for _, models in model_groups.items()
        for _, m, mf in models for f in mf
    ))
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    n_mf = len(all_model_feats)

    # Map model features to positions in feat_df columns
    feat_cols = list(feat_df.columns)
    feat_to_col = {f: i for i, f in enumerate(feat_cols)}
    
    # Warn about missing features
    missing_feats = [f for f in all_model_feats if f not in feat_to_col]
    if missing_feats:
        log_func(f"⚠️ {len(missing_feats)} model features not in feat_df (will be 0.0)")

    # Pre-compile model refs with mf_to_idx (feat_matrix column indices)
    model_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                model_refs.append((m, np.array(avail, dtype=np.int32)))

    # ── 5. Test range (in feat_df index) ──
    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    feat_index = feat_df.index
    test_start = max(WARMUP_BARS, int(np.searchsorted(feat_index, start_dt)))
    test_range = list(range(test_start, len(feat_index)))
    n_test = len(test_range)

    if n_test == 0:
        log_func("⚠️ No bars in test window")
        return None

    # ── 6. Build feature matrix (vectorized) ──
    feat_np = feat_df.values  # (n_rows, n_features)
    feat_slice = feat_np[test_start:test_start + n_test]
    feat_matrix = np.zeros((n_test, n_mf), dtype=np.float64)
    
    for mf_name, mf_pos in mf_to_idx.items():
        if mf_name in feat_to_col:
            col_pos = feat_to_col[mf_name]
            feat_matrix[:, mf_pos] = np.clip(
                np.nan_to_num(feat_slice[:, col_pos], nan=0.0), -10, 10
            )

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
    
    # Apply Platt scaling calibration (if available)
    cal_path = Path("data/ml_models") / f"{symbol}_calibrator.json"
    if cal_path.exists():
        try:
            import json
            with open(cal_path) as f:
                cal = json.load(f)
            # Vectorized Platt scaling
            z = cal['coef'] * all_probas + cal['intercept']
            all_probas = np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)
        except Exception:
            pass  # No calibration available — use raw probas
    
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
        bar_ts = feat_df.index[bi]
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
        'timestamps': [feat_df.index[bi] for bi in test_range],
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
