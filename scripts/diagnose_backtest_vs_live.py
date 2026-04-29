#!/usr/bin/env python3
"""
Mimia — Final Backtest vs Live Verification v4
===============================================
One-shot comparison: load OHLCV + models ONCE, then run BOTH
inference methods on the SAME feature matrix.

This isolates the ONLY difference: batch all-models-mean (backtest)
vs per-bar TF-group mean (live signal).

Method C already proved both averaging methods are identical (diff=0.0).
So the result should be 100% identical proba values.

Usage:
  python scripts/diagnose_backtest_vs_live.py --quick
  python scripts/diagnose_backtest_vs_live.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import THRESHOLD

WARMUP_BARS = 200  # must match backtest.py

import json
CALIBRATOR_CACHE = {}


def _load_calibrator(symbol):
    if symbol in CALIBRATOR_CACHE:
        return CALIBRATOR_CACHE[symbol]
    cal_path = Path("data/ml_models") / f"{symbol}_calibrator.json"
    if cal_path.exists():
        try:
            with open(cal_path) as f:
                cal = json.load(f)
            CALIBRATOR_CACHE[symbol] = cal
            return cal
        except Exception:
            return None
    return None


def _apply_calibration(probas, symbol):
    """Vectorized Platt scaling calibration."""
    cal = _load_calibrator(symbol)
    if cal is None:
        return probas
    z = cal['coef'] * probas + cal['intercept']
    return np.clip(1.0 / (1.0 + np.exp(-z)), 0.0, 1.0)


def detect_signal(proba):
    if proba is None: return 0
    if proba >= THRESHOLD: return 1
    if proba <= (1 - THRESHOLD): return -1
    return 0


def compare_one_symbol(symbol, hours=24, verbose=True):
    """One-shot comparison: shared OHLCV + features, two inference methods."""
    t0 = time.time()
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"  {symbol} — One-Shot Verification ({hours}h)")
        print(f"{'='*70}")
    
    # ── 1. Load OHLCV ONCE ──
    gen = SignalGenerator(symbol)
    log = lambda msg: print(f"    {msg}")
    
    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000:
        log(f"❌ No OHLCV")
        return None
    log(f"OHLCV: {len(df_5m)} bars ({df_5m.index[0]} → {df_5m.index[-1]})")
    
    # ── 2. Load models ONCE ──
    cached = gen._load_models(symbol)
    if cached is None:
        log(f"❌ No models")
        return None
    groups = cached['groups']
    
    # ── 3. Compute features ONCE (same pipeline as both engines) ──
    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
    if feat_df is None or len(feat_df) < 500:
        log(f"❌ Feature computation failed")
        return None
    
    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values
    log(f"Features: {len(feat_df)} rows × {len(feat_df.columns)} cols")
    
    # ── 4. Build shared structures ──
    all_model_feats = sorted(set(
        f for _, models in groups.items()
        for _, m, mf in models for f in mf
    ))
    n_mf = len(all_model_feats)
    mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
    feat_cols = list(feat_df.columns)
    feat_to_col = {f: i for i, f in enumerate(feat_cols)}
    
    # Warn about missing features
    missing = [f for f in all_model_feats if f not in feat_to_col]
    if missing:
        log(f"⚠️ {len(missing)} model features missing (will be 0.0)")
    
    # Pre-compile model refs (feat_matrix column indices via mf_to_idx)
    model_refs = []
    tf_model_refs = {}  # {tf_group: [(model, feat_matrix_col_indices)]}
    for tg, models in groups.items():
        tf_list = []
        for seed, m, mf in models:
            avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
            if len(avail) >= 5:
                arr = np.array(avail, dtype=np.int32)
                model_refs.append((m, arr))
                tf_list.append((m, arr))
        if tf_list:
            tf_model_refs[tg] = tf_list
    
    log(f"Models: {len(model_refs)} | Features: {n_mf}")
    
    # ── 5. Test range ──
    now = datetime.utcnow().replace(second=0, microsecond=0)
    start_dt = now - timedelta(hours=hours)
    feat_index = feat_df.index
    test_start = max(WARMUP_BARS, int(np.searchsorted(feat_index, start_dt)))
    test_range = list(range(test_start, len(feat_index)))
    n_test = len(test_range)
    
    if n_test == 0:
        log("⚠️ No bars in test window")
        return None
    
    test_timestamps = [feat_index[bi] for bi in test_range]
    log(f"Test window: {n_test} bars ({test_timestamps[0]} → {test_timestamps[-1]})")
    
    # ── 6. Build shared feature matrix ──
    feat_np = feat_df.values
    feat_slice = feat_np[test_start:test_start + n_test]
    feat_matrix = np.zeros((n_test, n_mf), dtype=np.float64)
    
    for mf_name, mf_pos in mf_to_idx.items():
        if mf_name in feat_to_col:
            col_pos = feat_to_col[mf_name]
            feat_matrix[:, mf_pos] = np.clip(
                np.nan_to_num(feat_slice[:, col_pos], nan=0.0), -10, 10
            )
    
    # ── 7. Method A: Batch all-models-mean (backtest) ──
    probas_bt = np.zeros(n_test, dtype=np.float64)
    nv = np.zeros(n_test, dtype=np.int32)
    for model, feat_idx in model_refs:
        try:
            preds = model.predict_proba(feat_matrix[:, feat_idx])[:, 1]
            probas_bt += preds
            nv += 1
        except Exception:
            continue
    probas_bt = np.where(nv > 0, probas_bt / nv, 0.0)
    # Apply calibration (same as backtest.py)
    probas_bt = _apply_calibration(probas_bt, symbol)
    signals_bt = np.array([detect_signal(p) for p in probas_bt])
    
    # ── 8. Method B: TF-group mean per bar (live) ──
    probas_live = np.zeros(n_test, dtype=np.float64)
    
    for i in range(n_test):
        row = feat_matrix[i:i+1]
        group_probs = []
        
        for tg, tf_models in tf_model_refs.items():
            tf_probs = []
            for model, feat_idx in tf_models:
                try:
                    preds = model.predict_proba(row[:, feat_idx])[:, 1]
                    tf_probs.append(preds[0])
                except Exception:
                    continue
            if tf_probs:
                group_probs.append(np.mean(tf_probs))
        
        if len(group_probs) >= 1:
            probas_live[i] = np.mean(group_probs)
        else:
            probas_live[i] = probas_live[i-1] if i > 0 else probas_bt[i]
    # Apply calibration (same as signals.py)
    probas_live = _apply_calibration(probas_live, symbol)
    
    signals_live = np.array([detect_signal(p) for p in probas_live])
    
    # ── 9. Compare ──
    diffs = np.abs(probas_bt - probas_live)
    exact = np.isclose(probas_bt, probas_live, rtol=1e-10, atol=1e-10)
    close_001 = diffs < 0.01
    sig_match = signals_bt == signals_live
    
    stats = {
        'symbol': symbol,
        'n_bars': n_test,
        'n_models': len(model_refs),
        'n_tf_groups': len(tf_model_refs),
        'n_missing_features': len(missing),
        'exact_pct': float(np.mean(exact) * 100),
        'close_001_pct': float(np.mean(close_001) * 100),
        'sig_match_pct': float(np.mean(sig_match) * 100),
        'mean_diff': float(np.mean(diffs)),
        'max_diff': float(np.max(diffs)),
        'bt_signals': {'L': int((signals_bt==1).sum()), 'S': int((signals_bt==-1).sum())},
        'live_signals': {'L': int((signals_live==1).sum()), 'S': int((signals_live==-1).sum())},
        'n_sig_mismatch': int(n_test - sig_match.sum()),
        'elapsed': time.time() - t0,
    }
    
    if verbose:
        print(f"\n  📊 RESULTS ({n_test} bars):")
        print(f"    Exact equal proba:      {stats['exact_pct']:5.1f}%")
        print(f"    Proba diff < 0.01:      {stats['close_001_pct']:5.1f}%")
        print(f"    Signal direction match: {stats['sig_match_pct']:5.1f}%")
        print(f"    Mean|Max proba diff:    {stats['mean_diff']:.6f} | {stats['max_diff']:.6f}")
        bt_n = stats['bt_signals']['L'] + stats['bt_signals']['S']
        live_n = stats['live_signals']['L'] + stats['live_signals']['S']
        print(f"    BT signals:  {bt_n} (L:{stats['bt_signals']['L']} S:{stats['bt_signals']['S']})")
        print(f"    Live signals: {live_n} (L:{stats['live_signals']['L']} S:{stats['live_signals']['S']})")
        
        if stats['n_sig_mismatch'] > 0:
            print(f"\n  ⚠️  Signal mismatches ({stats['n_sig_mismatch']} bars):")
            mis = np.where(~sig_match)[0]
            for idx in mis[:5]:
                ts = test_timestamps[idx]
                print(f"    @ {ts} | BT_sig={signals_bt[idx]:+d}(p={probas_bt[idx]:.4f}) "
                      f"Live_sig={signals_live[idx]:+d}(p={probas_live[idx]:.4f})")
    
    return stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', type=str)
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--out', type=str,
                        default='data/bt_vs_live_verified.json')
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    
    from src.trading.state import LIVE_SYMBOLS
    
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.quick:
        symbols = ['ENAUSDT', 'SUIUSDT', 'OPUSDT']
    else:
        symbols = LIVE_SYMBOLS
    
    print(f"{'='*70}")
    print(f"BACKTEST vs LIVE — ONE-SHOT VERIFICATION")
    print(f"Window: {args.hours}h | Symbols: {len(symbols)}")
    print(f"{'='*70}")
    
    results = {}
    summary = []
    
    for sym in symbols:
        print(f"\n  🔄 {sym}...")
        r = compare_one_symbol(sym, hours=args.hours, verbose=True)
        if r:
            results[sym] = r
            summary.append(r)
        else:
            print(f"  ⏭️")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    hdr = (f"{'Symbol':>10s} | {'Bars':>5s} | {'Exact%':>6s} | {'<.01%':>5s} | "
           f"{'SigMat%':>6s} | {'AvgDiff':>7s} | {'MaxDiff':>7s} | {'BT|LiveSig':>10s}")
    print(hdr)
    print('-' * len(hdr))
    
    for r in summary:
        bt_n = r['bt_signals']['L'] + r['bt_signals']['S']
        live_n = r['live_signals']['L'] + r['live_signals']['S']
        print(f"{r['symbol']:>10s} | {r['n_bars']:5d} | "
              f"{r['exact_pct']:5.1f}% | {r['close_001_pct']:4.1f}% | "
              f"{r['sig_match_pct']:5.1f}% | "
              f"{r['mean_diff']:6.4f} | {r['max_diff']:6.4f} | "
              f"{bt_n:3d}|{live_n:<3d}")
    
    sig_matches = [r['sig_match_pct'] for r in summary]
    exacts = [r['exact_pct'] for r in summary]
    avg_sig = np.mean(sig_matches) if sig_matches else 0
    avg_exact = np.mean(exacts) if exacts else 0
    
    print(f"\n{'='*70}")
    if avg_exact > 99:
        print("✅ VERDICT: 100% IDENTICAL — backtest == live")
    elif avg_sig > 99:
        print("✅ VERDICT: Signals identical — tiny proba drift only")
    elif avg_sig > 95:
        print("⚠️  NEAR-IDENTICAL — minor residual differences")
    else:
        print("🔴 STILL DIFFERENT — investigation continues")
    print(f"  Avg exact: {avg_exact:.1f}% | Avg sig match: {avg_sig:.1f}%")
    
    if args.out:
        with open(args.out, 'w') as f:
            json.dump({
                'meta': {
                    'hours': args.hours,
                    'timestamp': datetime.utcnow().isoformat(),
                    'avg_exact_pct': round(avg_exact, 1),
                    'avg_sig_match_pct': round(avg_sig, 1),
                },
                'results': results,
            }, f, indent=2, default=str)
        print(f"\n✅ Saved → {args.out}")


if __name__ == '__main__':
    main()
