#!/usr/bin/env python3
"""
Mimia — Proba-Level Comparison Script (V3)
===========================================
Focuses on comparing model probabilities at the BAR level, not just signal direction.
This tells us if the model outputs the SAME proba for the SAME 5m bar,
regardless of when the live trader evaluated it.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.trading.signals import SignalGenerator
from src.trading.state import LIVE_SYMBOLS, THRESHOLD, HOLD_BARS, COOLDOWN_BARS, DB_PATH
from src.trading.state import TAKER_FEE, SLIPPAGE, MODEL_DIR, TF_GROUPS, SEEDS


def compute_bar_probas(symbol: str, days_back: float = 1.5) -> Optional[pd.Series]:
    """Same as V2 — compute proba for every bar."""
    gen = SignalGenerator(symbol)
    
    spot_symbol = symbol
    if symbol.startswith("1000"):
        for prefix in ["1000", "10000", "100000"]:
            if symbol.startswith(prefix):
                spot_symbol = symbol[len(prefix):]
                break
    
    cached = gen._load_models(symbol)
    if cached is None:
        return None
    
    feat_df = cached['features']
    groups = cached['groups']
    if len(feat_df) < 100:
        return None
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    feat_df = feat_df[feat_df.index >= cutoff].copy()
    if len(feat_df) < 50:
        return None
    
    group_probs_all = {}
    for tf, models in groups.items():
        tf_probs_all = []
        for seed, m, mf in models:
            available = [c for c in mf if c in feat_df.columns]
            if len(available) < 5:
                continue
            X = feat_df[available].fillna(0).clip(-10, 10).values
            probs = m.predict_proba(X)[:, 1]
            tf_probs_all.append(probs)
        if tf_probs_all:
            group_probs_all[tf] = np.nanmean(tf_probs_all, axis=0)
    
    if len(group_probs_all) < 2:
        return None
    
    prob_stack = np.column_stack([group_probs_all[tf] for tf in group_probs_all])
    return pd.Series(np.mean(prob_stack, axis=1), index=feat_df.index)


def analyze_symbol(symbol: str) -> Dict:
    """Comprehensive proba-level analysis."""
    print(f"\n  {'─'*60}")
    print(f"  {symbol}")
    print(f"  {'─'*60}")
    
    # 1. Compute bar-level probas
    bar_probas = compute_bar_probas(symbol)
    if bar_probas is None:
        print(f"    ❌ No proba data")
        return None
    
    print(f"    Model bars: {len(bar_probas)}")
    print(f"    Date range: {bar_probas.index[0]} → {bar_probas.index[-1]}")
    
    # 2. Get live signals
    cutoff_ms = int((time.time() - 36 * 3600) * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.execute('''
        SELECT timestamp, proba, signal FROM live_signals
        WHERE symbol = ? AND timestamp >= ? ORDER BY timestamp
    ''', (symbol, cutoff_ms))
    live_sigs = [{
        'ts_ms': r[0],
        'ts': datetime.utcfromtimestamp(r[0]/1000),
        'proba': r[1],
        'signal': r[2],
    } for r in c.fetchall()]
    conn.close()
    
    print(f"    Live signals: {len(live_sigs)}")
    
    if not live_sigs:
        return {'symbol': symbol, 'error': 'No live signals'}
    
    # 3. Map each live signal to nearest 5m bar
    matched = []
    proba_diffs = []
    signal_mismatches = 0
    missed_bars = 0
    
    for sig in live_sigs:
        sig_ts = sig['ts']
        nearest_idx = bar_probas.index.get_indexer([sig_ts], method='nearest')[0]
        if nearest_idx < 0 or nearest_idx >= len(bar_probas):
            missed_bars += 1
            continue
        
        bar_ts = bar_probas.index[nearest_idx]
        bar_proba = bar_probas.iloc[nearest_idx]
        time_diff = abs((sig_ts - bar_ts).total_seconds())
        
        if time_diff > 600:  # Skip if > 10 min away (shouldn't happen for 5-min cycle)
            missed_bars += 1
            continue
        
        # Align to bar timestamp
        bar_signal = 0
        if bar_proba >= THRESHOLD:
            bar_signal = 1
        elif bar_proba <= (1 - THRESHOLD):
            bar_signal = -1
        
        proba_diff = abs(sig['proba'] - bar_proba)
        proba_diffs.append(proba_diff)
        signal_match = (sig['signal'] == bar_signal)
        
        if not signal_match:
            signal_mismatches += 1
        
        matched.append({
            'sig_ts': sig_ts,
            'bar_ts': bar_ts,
            'sig_proba': sig['proba'],
            'bar_proba': bar_proba,
            'proba_diff': proba_diff,
            'sig_signal': sig['signal'],
            'bar_signal': bar_signal,
            'signal_match': signal_match,
            'time_diff_s': time_diff,
        })
    
    n_matched = len(matched)
    n_signal_mismatches = signal_mismatches
    n_missed = missed_bars
    
    # 4. Statistics
    if proba_diffs:
        avg_proba_diff = np.mean(proba_diffs)
        max_proba_diff = max(proba_diffs)
        std_proba_diff = np.std(proba_diffs)
        p50 = np.percentile(proba_diffs, 50)
        p95 = np.percentile(proba_diffs, 95)
    else:
        avg_proba_diff = max_proba_diff = std_proba_diff = p50 = p95 = 0
    
    signal_match_rate = (n_matched - n_signal_mismatches) / n_matched * 100 if n_matched > 0 else 0
    
    # 5. Analyze by proximity
    # Are mismatches concentrated near the threshold boundary (0.55-0.65 where small proba changes flip signal)?
    boundary_mismatches = 0
    deep_mismatches = 0
    for m in matched:
        if not m['signal_match']:
            # Is the signal near threshold boundary?
            near_boundary = (0.55 <= m['sig_proba'] <= 0.65) or (0.35 <= m['sig_proba'] <= 0.45)
            if near_boundary:
                boundary_mismatches += 1
            else:
                deep_mismatches += 1
    
    print(f"\n    {'─'*50}")
    print(f"    PROBA-LEVEL ANALYSIS")
    print(f"    {'─'*50}")
    print(f"    Matched samples:         {n_matched}")
    print(f"    Missed (no bar):         {n_missed}")
    print(f"    Signal match rate:       {signal_match_rate:.1f}")
    print(f"    Avg |proba diff|:        {avg_proba_diff:.4f}")
    print(f"    Std |proba diff|:        {std_proba_diff:.4f}")
    print(f"    P50 |proba diff|:        {p50:.4f}")
    print(f"    P95 |proba diff|:        {p95:.4f}")
    print(f"    Max |proba diff|:        {max_proba_diff:.4f}")
    print(f"    Signal mismatches:       {n_signal_mismatches}")
    print(f"      → Near boundary (0.55-0.65 or 0.35-0.45): {boundary_mismatches}")
    print(f"      → Deep (far from boundary):              {deep_mismatches}")
    
    # 6. Show worst mismatches
    mismatches_sorted = sorted([m for m in matched if not m['signal_match']],
                               key=lambda m: m['proba_diff'], reverse=True)
    
    if mismatches_sorted:
        print(f"\n    ── Signal Mismatches (worst 10 by proba diff) ──")
        print(f"    {'Time':6s} {'LiveP':6s} {'LiveS':6s} {'BarP':6s} {'BarS':6s} {'ΔProb':7s} {'Δ(s)':6s} {'Boundary':8s}")
        print(f"    {'-'*55}")
        for m in mismatches_sorted[:10]:
            near = "YES" if (0.55 <= m['sig_proba'] <= 0.65) or (0.35 <= m['sig_proba'] <= 0.45) else "no"
            print(f"    {m['sig_ts'].strftime('%H:%M'):6s} {m['sig_proba']:.4f} {m['sig_signal']:4d} {m['bar_proba']:.4f} {m['bar_signal']:4d} {m['proba_diff']:.4f} {m['time_diff_s']:4.0f}s {near:>8s}")
    
    # 7. Show matched samples with largest proba diff (even if same signal)
    all_sorted = sorted(matched, key=lambda m: m['proba_diff'], reverse=True)
    if all_sorted:
        print(f"\n    ── Largest |proba diff| (top 10) ──")
        print(f"    {'Time':6s} {'LiveP':6s} {'BarP':6s} {'ΔProb':7s} {'Match':4s} {'Δ(s)':6s}")
        print(f"    {'-'*40}")
        for m in all_sorted[:10]:
            icon = '✅' if m['signal_match'] else '❌'
            print(f"    {m['sig_ts'].strftime('%H:%M'):6s} {m['sig_proba']:.4f} {m['bar_proba']:.4f} {m['proba_diff']:.4f} {icon:4s} {m['time_diff_s']:4.0f}s")
    
    return {
        'symbol': symbol,
        'n_matched': n_matched,
        'signal_match_pct': signal_match_rate,
        'avg_proba_diff': avg_proba_diff,
        'p50_proba_diff': p50,
        'p95_proba_diff': p95,
        'max_proba_diff': max_proba_diff,
        'n_mismatches': n_signal_mismatches,
        'boundary_mismatches': boundary_mismatches,
        'deep_mismatches': deep_mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', '-s', default=None)
    args = parser.parse_args()
    
    symbols = [args.symbol] if args.symbol else LIVE_SYMBOLS
    
    print(f"{'='*70}")
    print(f"  PROBA-LEVEL BACKTEST vs LIVE COMPARISON")
    print(f"{'='*70}")
    
    all_results = []
    for sym in symbols:
        result = analyze_symbol(sym)
        if result and not result.get('error'):
            all_results.append(result)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        for r in all_results:
            print(f"  {r['symbol']:12s}: signal_match={r['signal_match_pct']:5.1f}% | "
                  f"avg|Δproba|={r['avg_proba_diff']:.4f} | "
                  f"p50={r['p50_proba_diff']:.4f} p95={r['p95_proba_diff']:.4f} | "
                  f"mismatches={r['n_mismatches']:3d} "
                  f"({r['boundary_mismatches']} boundary, {r['deep_mismatches']} deep)")
        
        avg_signal = np.mean([r['signal_match_pct'] for r in all_results])
        avg_proba = np.mean([r['avg_proba_diff'] for r in all_results])
        avg_p95 = np.mean([r['p95_proba_diff'] for r in all_results])
        
        total_mismatches = sum(r['n_mismatches'] for r in all_results)
        total_boundary = sum(r['boundary_mismatches'] for r in all_results)
        total_deep = sum(r['deep_mismatches'] for r in all_results)
        
        print(f"\n  {'─'*60}")
        print(f"  Avg Signal Match:  {avg_signal:.1f}%")
        print(f"  Avg |Δproba|:      {avg_proba:.4f}")
        print(f"  Avg P95 |Δproba|:  {avg_p95:.4f}")
        print(f"  Total Mismatches:  {total_mismatches} ({total_boundary} boundary, {total_deep} deep)")
    
    print()


if __name__ == '__main__':
    main()
