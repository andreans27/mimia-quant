#!/usr/bin/env python3
"""
Compare Baseline (THRESHOLD=0.60, HOLD_BARS=9, POSITION_PCT=0.15) vs Enhanced
(per-symbol threshold, hold bars, dynamic sizing) backtest results.

Runs both on the SAME data and compares trade-by-trade + aggregate metrics.
"""
import sys; sys.path.insert(0, '.')
import time as ttime
import warnings; warnings.filterwarnings('ignore')
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Imports ───────────────────────────────────────
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE, SLIPPAGE,
    POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS,
    get_symbol_threshold, get_symbol_hold_bars, get_dynamic_position_pct,
)

WARMUP_BARS = 200

def simulate_trades(
    symbol: str, close_arr: np.ndarray, feat_index,
    test_range, all_signals, all_probas,
    hold_bars_fn, position_size_fn,
    label: str = ""
) -> Tuple[list, float, float, float]:
    """Run trade simulation with given hold_bars and position_size strategies."""
    capital = INITIAL_CAPITAL
    peak_cap = INITIAL_CAPITAL
    pos = 0; ep = 0.0; eq = 0.0; hr = 0; cd = 0; eproba = 0.0
    entry_ts = None; entry_price_bt = 0.0
    ps = 0; pproba = 0.0
    trade_details = []
    n_test = len(test_range)

    for test_ii, bi in enumerate(test_range):
        price = float(close_arr[bi])
        bar_ts = feat_index[bi]
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

        # Entry (deferred)
        if ps != 0 and pos == 0 and cd <= 0:
            slip = (1 + SLIPPAGE) if ps == 1 else (1 - SLIPPAGE)
            ep = price * slip
            entry_price_bt = ep
            entry_ts = bar_ts
            pos_size_pct = position_size_fn(pproba, symbol)
            eq = (capital * pos_size_pct) / ep
            capital -= ep * eq * TAKER_FEE
            peak_cap = max(peak_cap, capital)
            pos = ps; eproba = pproba; hr = hold_bars_fn(symbol)

        # Signal for NEXT bar
        ps = all_signals[test_ii]
        pproba = all_probas[test_ii]

    # Metrics
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

    return trade_details, wr, tp, mdd, pf, lp, sp_


def backtest_symbol(symbol: str, test_hours: int = 168, verbose: bool = False) -> Optional[Dict]:
    """Run backtest for one symbol, returning BOTH baseline and enhanced results."""
    log_func = (lambda msg: print(f"  {msg}")) if verbose else (lambda _: None)
    t0 = ttime.time()
    log_func(f"\n{'='*50}\n{symbol} | {test_hours}h")

    # 1. Load models
    gen = SignalGenerator(symbol)
    cached = gen._load_models(symbol)
    if cached is None:
        log_func("❌ No models")
        return None
    model_groups = cached['groups']

    # 2. Fetch OHLCV
    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        log_func("❌ No data")
        return None

    # 3. Compute features
    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
    if feat_df is None or len(feat_df) < 500:
        log_func("❌ Feature computation failed")
        return None
    log_func(f"✅ {len(feat_df)} feature rows | {len(feat_df.columns)} features")

    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values

    # 4. Build model refs
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
        log_func(f"⚠️ {len(missing_feats)} model features not in feat_df")

    long_refs = []; short_refs = []
    for tg, models in model_groups.items():
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                ref = (m, np.array(avail, dtype=np.int32))
                if tg == 'long': long_refs.append(ref)
                elif tg == 'short': short_refs.append(ref)

    # 5. Test range
    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    feat_index = feat_df.index
    test_start = max(WARMUP_BARS, int(np.searchsorted(feat_index, start_dt)))
    test_range = list(range(test_start, len(feat_index)))
    n_test = len(test_range)
    if n_test == 0:
        log_func("⚠️ No bars in test window")
        return None

    # 6. Feature matrix
    feat_np = feat_df.values
    feat_slice = feat_np[test_start:test_start + n_test]
    feat_matrix = np.zeros((n_test, n_mf), dtype=np.float64)
    for mf_name, mf_pos in mf_to_idx.items():
        if mf_name in feat_to_col:
            col_pos = feat_to_col[mf_name]
            feat_matrix[:, mf_pos] = np.clip(
                np.nan_to_num(feat_slice[:, col_pos], nan=0.0), -10, 10
            )

    # 7. Batch inference
    def _batch_infer(refs, fm):
        probas = np.zeros(n_test, dtype=np.float64)
        nv = np.zeros(n_test, dtype=np.int32)
        for model, feat_idx in refs:
            try:
                preds = model.predict_proba(fm[:, feat_idx])[:, 1]
                probas += preds
                nv += 1
            except Exception:
                continue
        return np.where(nv > 0, probas / nv, 0.0)

    def _apply_cal(probas, side):
        cal_path = Path("data/ml_models") / f"{symbol}_{side}_calibrator.json"
        if cal_path.exists():
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
                z = cal['coef'] * probas + cal['intercept']
                return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)
            except Exception:
                pass
        return probas

    long_probas = _batch_infer(long_refs, feat_matrix)
    long_probas = _apply_cal(long_probas, 'long')
    short_probas = _batch_infer(short_refs, feat_matrix)
    short_probas = _apply_cal(short_probas, 'short')

    # 8. ENHANCED signal (per-symbol threshold)
    enhanced_threshold = get_symbol_threshold(symbol)
    all_signals = np.zeros(n_test, dtype=np.int32)
    long_mask = long_probas >= enhanced_threshold
    short_mask = short_probas >= enhanced_threshold
    both = long_mask & short_mask
    pick_long = long_probas >= short_probas
    all_signals[both & pick_long] = 1
    all_signals[both & ~pick_long] = -1
    all_signals[long_mask & ~short_mask] = 1
    all_signals[short_mask & ~long_mask] = -1
    all_probas = np.where(all_signals == 1, long_probas,
                          np.where(all_signals == -1, short_probas,
                                   np.maximum(long_probas, short_probas)))

    # Generate BASELINE signals (global THRESHOLD=0.60 for all)
    all_signals_baseline = np.zeros(n_test, dtype=np.int32)
    long_mask_b = long_probas >= THRESHOLD
    short_mask_b = short_probas >= THRESHOLD
    both_b = long_mask_b & short_mask_b
    all_signals_baseline[both_b & (long_probas >= short_probas)] = 1
    all_signals_baseline[both_b & ~(long_probas >= short_probas)] = -1
    all_signals_baseline[long_mask_b & ~short_mask_b] = 1
    all_signals_baseline[short_mask_b & ~long_mask_b] = -1
    all_probas_b = np.where(all_signals_baseline == 1, long_probas,
                            np.where(all_signals_baseline == -1, short_probas,
                                     np.maximum(long_probas, short_probas)))

    # 9. Simulate trades — BASELINE
    base_trades, base_wr, base_pnl, base_dd, base_pf, base_lp, base_sp = simulate_trades(
        symbol, close_arr, feat_index, test_range,
        all_signals_baseline, all_probas_b,
        hold_bars_fn=lambda s: HOLD_BARS,   # global 9
        position_size_fn=lambda p, s: POSITION_PCT,  # global 0.15
        label="baseline"
    )

    # 10. Simulate trades — ENHANCED
    enh_trades, enh_wr, enh_pnl, enh_dd, enh_pf, enh_lp, enh_sp = simulate_trades(
        symbol, close_arr, feat_index, test_range,
        all_signals_baseline if enhanced_threshold == THRESHOLD else all_signals,
        all_probas_b if enhanced_threshold == THRESHOLD else all_probas,
        hold_bars_fn=get_symbol_hold_bars,
        position_size_fn=get_dynamic_position_pct,
        label="enhanced"
    )

    elapsed = ttime.time() - t0
    log_func(f"  Baseline:  {len(base_trades):3d} trades | WR {base_wr:5.1f}% | PnL ${base_pnl:+8.2f} | DD {base_dd:.2f}%")
    log_func(f"  Enhanced:  {len(enh_trades):3d} trades | WR {enh_wr:5.1f}% | PnL ${enh_pnl:+8.2f} | DD {enh_dd:.2f}%")
    log_func(f"  ⏱ {elapsed:.1f}s")

    return {
        'symbol': symbol,
        'baseline': {
            'n_trades': len(base_trades), 'win_rate': base_wr, 'total_pnl': base_pnl,
            'max_dd': base_dd, 'profit_factor': base_pf,
            'long_pnl': base_lp, 'short_pnl': base_sp,
            'trade_details': base_trades,
        },
        'enhanced': {
            'n_trades': len(enh_trades), 'win_rate': enh_wr, 'total_pnl': enh_pnl,
            'max_dd': enh_dd, 'profit_factor': enh_pf,
            'long_pnl': enh_lp, 'short_pnl': enh_sp,
            'trade_details': enh_trades,
        },
        'elapsed': elapsed,
    }


def print_comparison(results: List[dict]):
    """Print formatted comparison table."""
    print(f"\n{'='*85}")
    print(f"  COMPARISON: BASELINE vs ENHANCED (1 Week = 168h)")
    print(f"{'='*85}")
    print(f"  {'Symbol':<12} {'Trades':>6} {'WR%':>6} {'PnL':>10} {'DD%':>6} {'PF':>6} | "
          f"{'Trades':>6} {'WR%':>6} {'PnL':>10} {'DD%':>6} {'PF':>6} | {'Δ PnL':>10}")
    print(f"  {'':-<12} {'':->6} {'':->6} {'':->10} {'':->6} {'':->6} | "
          f"{'':->6} {'':->6} {'':->10} {'':->6} {'':->6} | {'':->10}")
    print(f"     {'BASELINE':^44} |     {'ENHANCED':^44} |")

    agg_base = {'n_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'max_dd': 0.0, 'profit_factor': 0.0}
    agg_enh = {'n_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0, 'max_dd': 0.0, 'profit_factor': 0.0}
    n_profitable_base = 0
    n_profitable_enh = 0

    for r in results:
        b = r['baseline']; e = r['enhanced']
        print(f"  {r['symbol']:<12} {b['n_trades']:>6} {b['win_rate']:>5.1f}% "
              f"${b['total_pnl']:>+8.2f} {b['max_dd']:>5.2f}% "
              f"{b['profit_factor']:>5.1f} | "
              f"{e['n_trades']:>6} {e['win_rate']:>5.1f}% "
              f"${e['total_pnl']:>+8.2f} {e['max_dd']:>5.2f}% "
              f"{e['profit_factor']:>5.1f} | "
              f"${e['total_pnl'] - b['total_pnl']:>+9.2f}")
        agg_base['n_trades'] += b['n_trades']
        agg_base['total_pnl'] += b['total_pnl']
        agg_base['max_dd'] = max(agg_base['max_dd'], b['max_dd'])
        if b['total_pnl'] > 0: n_profitable_base += 1
        agg_enh['n_trades'] += e['n_trades']
        agg_enh['total_pnl'] += e['total_pnl']
        agg_enh['max_dd'] = max(agg_enh['max_dd'], e['max_dd'])
        if e['total_pnl'] > 0: n_profitable_enh += 1

    # Totals
    tot_base_wr = sum(1 for t in [tr for r in results for tr in r['baseline']['trade_details']] if t['pnl'] > 0)
    tot_base_n = sum(r['baseline']['n_trades'] for r in results)
    tot_enh_wr = sum(1 for t in [tr for r in results for tr in r['enhanced']['trade_details']] if t['pnl'] > 0)
    tot_enh_n = sum(r['enhanced']['n_trades'] for r in results)

    # Aggregate PF
    base_gp = sum(t['pnl'] for r in results for t in r['baseline']['trade_details'] if t['pnl'] > 0)
    base_gl = abs(sum(t['pnl'] for r in results for t in r['baseline']['trade_details'] if t['pnl'] <= 0))
    enh_gp = sum(t['pnl'] for r in results for t in r['enhanced']['trade_details'] if t['pnl'] > 0)
    enh_gl = abs(sum(t['pnl'] for r in results for t in r['enhanced']['trade_details'] if t['pnl'] <= 0))

    print(f"  {'':-<85}")
    print(f"  {'TOTAL':<12} {agg_base['n_trades']:>6} "
          f"{tot_base_wr/tot_base_n*100 if tot_base_n else 0:>5.1f}% "
          f"${agg_base['total_pnl']:>+8.2f} {agg_base['max_dd']:>5.2f}% "
          f"{base_gp/base_gl if base_gl else float('inf'):>5.1f} | "
          f"{agg_enh['n_trades']:>6} "
          f"{tot_enh_wr/tot_enh_n*100 if tot_enh_n else 0:>5.1f}% "
          f"${agg_enh['total_pnl']:>+8.2f} {agg_enh['max_dd']:>5.2f}% "
          f"{enh_gp/enh_gl if enh_gl else float('inf'):>5.1f} | "
          f"${agg_enh['total_pnl'] - agg_base['total_pnl']:>+9.2f}")

    # Additional stats
    delta_pnl = agg_enh['total_pnl'] - agg_base['total_pnl']
    delta_trades = agg_enh['n_trades'] - agg_base['n_trades']
    delta_wr = (tot_enh_wr/tot_enh_n*100 if tot_enh_n else 0) - (tot_base_wr/tot_base_n*100 if tot_base_n else 0)
    print(f"\n  📊 SUMMARY")
    print(f"  {'Trades:':>20} {agg_base['n_trades']} → {agg_enh['n_trades']} ({delta_trades:+d}, {delta_trades/agg_base['n_trades']*100:+>.1f}%)")
    print(f"  {'Win Rate:':>20} {tot_base_wr/tot_base_n*100 if tot_base_n else 0:.1f}% → {tot_enh_wr/tot_enh_n*100 if tot_enh_n else 0:.1f}% ({delta_wr:+.1f}pp)")
    print(f"  {'Total PnL:':>20} ${agg_base['total_pnl']:>+.2f} → ${agg_enh['total_pnl']:>+.2f} ({delta_pnl:+.2f}, {delta_pnl/abs(agg_base['total_pnl'])*100:+>.1f}%)")
    print(f"  {'Max DD:':>20} {agg_base['max_dd']:.2f}% → {agg_enh['max_dd']:.2f}%")
    print(f"  {'Profit Factor:':>20} {base_gp/base_gl if base_gl else float('inf'):.1f} → {enh_gp/enh_gl if enh_gl else float('inf'):.1f}")
    print(f"  {'Profitable Symbols:':>20} {n_profitable_base}/{len(results)} → {n_profitable_enh}/{len(results)}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare baseline vs enhanced backtest')
    parser.add_argument('--hours', type=int, default=168, help='Test window hours')
    parser.add_argument('--symbols', nargs='*', default=None, help='Specific symbols')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    symbols = args.symbols or LIVE_SYMBOLS
    print(f"🔬 Comparison: {' + '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}")
    print(f"📊 Test window: {args.hours}h ({args.hours/24:.1f} days)")
    print()

    results = []
    for i, sym in enumerate(symbols):
        r = backtest_symbol(sym, test_hours=args.hours, verbose=args.verbose)
        if r:
            results.append(r)

    if results:
        print_comparison(results)
        # Save results
        out_path = f"data/backtest_compare_{args.hours}h.json"
        with open(out_path, 'w') as f:
            # Strip trade_details for file size
            export = []
            for r in results:
                er = {
                    'symbol': r['symbol'],
                    'baseline': {k: v for k, v in r['baseline'].items() if k != 'trade_details'},
                    'enhanced': {k: v for k, v in r['enhanced'].items() if k != 'trade_details'},
                    'elapsed': r['elapsed'],
                }
                export.append(er)
            json.dump(export, f, indent=2, default=str)
        print(f"\n💾 Results saved to {out_path}")
