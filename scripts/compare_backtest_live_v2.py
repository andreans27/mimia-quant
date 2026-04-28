#!/usr/bin/env python3
"""
Mimia — Backtest vs Live Trader Comparison V2
==============================================
Improved comparison:
1. Only compares model-driven trades (exit_reason='hold_expiry', proba != 0.5)
2. Directly compares probabilities at each daemon cycle (live_signals table)
3. Simulates trade-by-trade based on exact bar-level proba

Usage:
    python scripts/compare_backtest_live.py [--symbol SYMBOL] [--list]
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import json
import sqlite3
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.trading.signals import SignalGenerator
from src.trading.state import (
    LIVE_SYMBOLS, THRESHOLD, HOLD_BARS, COOLDOWN_BARS,
    DB_PATH, TAKER_FEE, SLIPPAGE, MODEL_DIR, TF_GROUPS, SEEDS,
)
from src.strategies.ml_features import OHLCV_CACHE_DIR


def compute_bar_probas(symbol: str, days_back: float = 1.5) -> Optional[pd.Series]:
    """
    Compute model probabilities for EVERY 5m bar in the test window.
    Uses the exact same SignalGenerator._load_models() code path.
    Returns a pd.Series with proba values indexed by timestamp.
    """
    gen = SignalGenerator(symbol)
    
    spot_symbol = symbol
    if symbol.startswith("1000"):
        for prefix in ["1000", "10000", "100000"]:
            if symbol.startswith(prefix):
                spot_symbol = symbol[len(prefix):]
                break
    
    # Load models + features (same as live trader)
    cached = gen._load_models(symbol)
    if cached is None:
        print(f"    ❌ {symbol}: Cannot load models")
        return None
    
    feat_df = cached['features']
    groups = cached['groups']
    
    if len(feat_df) < 100:
        print(f"    ❌ {symbol}: Too few features ({len(feat_df)})")
        return None
    
    cutoff = datetime.utcnow() - timedelta(days=days_back)
    feat_df = feat_df[feat_df.index >= cutoff].copy()
    if len(feat_df) < 50:
        print(f"    ⚠️ {symbol}: Too few feature rows in window ({len(feat_df)})")
        return None
    
    # Compute proba for each bar
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
    avg_probs = np.mean(prob_stack, axis=1)
    
    return pd.Series(avg_probs, index=feat_df.index)


def get_live_trades(symbol: str, hours: int = 36) -> List[Dict]:
    """Get model-driven live trades (NOT history_sync)."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    
    c = conn.execute('''
        SELECT direction, entry_time, exit_time, entry_price, exit_price,
               qty, pnl_net, pnl_pct, entry_proba, hold_bars, exit_reason
        FROM live_trades
        WHERE symbol = ? AND entry_time >= ?
          AND exit_reason = 'hold_expiry'
          AND entry_proba != 0.5
        ORDER BY entry_time
    ''', (symbol, cutoff_ms))
    
    trades = []
    for row in c.fetchall():
        trades.append({
            'symbol': symbol,
            'direction': row[0],
            'entry_time_ms': row[1],
            'entry_time': datetime.utcfromtimestamp(row[1]/1000),
            'exit_time_ms': row[2],
            'exit_time': datetime.utcfromtimestamp(row[2]/1000) if row[2] else None,
            'entry_price': row[3],
            'exit_price': row[4],
            'qty': row[5],
            'pnl_net': row[6],
            'pnl_pct': row[7],
            'entry_proba': row[8],
            'hold_bars': row[9],
            'exit_reason': row[10],
        })
    
    conn.close()
    return trades


def get_live_signals(symbol: str, hours: int = 36) -> List[Dict]:
    """Get live trader signals for comparison."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    
    c = conn.execute('''
        SELECT timestamp, proba, signal, capital
        FROM live_signals
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp
    ''', (symbol, cutoff_ms))
    
    signals = []
    for row in c.fetchall():
        signals.append({
            'timestamp_ms': row[0],
            'timestamp': datetime.utcfromtimestamp(row[0]/1000),
            'proba': row[1],
            'signal': row[2],
        })
    
    conn.close()
    return signals


def simulate_trades(proba_series: pd.Series,
                    threshold: float = THRESHOLD,
                    hold_bars: int = HOLD_BARS,
                    cooldown_bars: int = COOLDOWN_BARS) -> List[Dict]:
    """
    Simulate trades based on proba series (bar-by-bar, same logic as backtest).
    Returns list of trade dicts.
    """
    trades = []
    position = 0
    hold_remaining = 0
    cooldown = 0
    entry_time = None
    entry_proba = 0.0
    
    for i, (ts, proba) in enumerate(proba_series.items()):
        # Decrement
        hold_remaining = max(0, hold_remaining - 1)
        cooldown = max(0, cooldown - 1)
        
        # Exit check
        if position != 0 and hold_remaining <= 0:
            trades.append({
                'direction': 'long' if position == 1 else 'short',
                'entry_time': entry_time,
                'entry_proba': entry_proba,
                'exit_time': ts,
                'hold_bars': hold_bars,
            })
            position = 0
            cooldown = cooldown_bars
        
        # Entry check
        if position == 0 and cooldown <= 0:
            signal = 0
            if proba >= threshold:
                signal = 1
            elif proba <= (1 - threshold):
                signal = -1
            
            if signal != 0:
                position = signal
                entry_time = ts
                entry_proba = proba
                hold_remaining = hold_bars
    
    return trades


def compare_for_symbol(symbol: str) -> Dict:
    """Full comparison for one symbol."""
    print(f"\n  {'─'*60}")
    print(f"  {symbol}")
    print(f"  {'─'*60}")
    
    # 1. Compute bar-level probas
    proba_series = compute_bar_probas(symbol)
    if proba_series is None:
        return {'symbol': symbol, 'error': 'No proba data'}
    
    # 2. Get live trader data
    live_trades = get_live_trades(symbol, hours=36)
    live_signals = get_live_signals(symbol, hours=36)
    
    print(f"  Model bars:   {len(proba_series)}")
    print(f"  Live signals: {len(live_signals)}")
    print(f"  Live trades:  {len(live_trades)} (hold_expiry, non-backfill)")
    
    # 3. Simulate trades from model probas
    bt_trades = simulate_trades(proba_series)
    print(f"  BT trades:    {len(bt_trades)}")
    
    # 4. Compare signal probabilities at matching timestamps
    #    For each live signal, find the nearest BT proba bar
    signal_comparison = []
    unmatched_signal_bars = []
    
    for sig in live_signals:
        sig_ts = sig['timestamp']
        # Find nearest bar in proba_series
        nearest_idx = proba_series.index.get_indexer([sig_ts], method='nearest')[0]
        if nearest_idx >= 0 and len(proba_series) > nearest_idx:
            bar_ts = proba_series.index[nearest_idx]
            bar_proba = proba_series.iloc[nearest_idx]
            time_diff = abs((sig_ts - bar_ts).total_seconds())
            
            if time_diff <= 360:  # Within 6 min (1 bar + margin)
                bt_signal = 0
                if bar_proba >= THRESHOLD:
                    bt_signal = 1
                elif bar_proba <= (1 - THRESHOLD):
                    bt_signal = -1
                
                signal_match = (sig['signal'] == bt_signal)
                
                signal_comparison.append({
                    'time': sig_ts.strftime('%H:%M'),
                    'live_proba': sig['proba'],
                    'live_signal': sig['signal'],
                    'bar_proba': bar_proba,
                    'bar_signal': bt_signal,
                    'match': signal_match,
                    'time_diff_s': time_diff,
                })
            else:
                unmatched_signal_bars.append({
                    'time': sig_ts.strftime('%H:%M'),
                    'live_proba': sig['proba'],
                    'live_signal': sig['signal'],
                    'reason': f'No bar within 360s (nearest: {time_diff:.0f}s)',
                })
    
    # 5. Compare trade entries
    trade_comparison = []
    bt_trade_times = [t['entry_time'] for t in bt_trades]
    live_trade_times = [t['entry_time'] for t in live_trades]
    
    bt_matched = [False] * len(bt_trades)
    live_matched = [False] * len(live_trades)
    
    for li, lt in enumerate(live_trades):
        lt_entry = lt['entry_time']
        best_match = None
        best_diff = timedelta(minutes=10)
        
        for bi, bt_t in enumerate(bt_trades):
            if bt_matched[bi]:
                continue
            diff = abs(lt_entry - bt_t['entry_time'])
            if diff < best_diff:
                best_diff = diff
                best_match = bi
        
        if best_match is not None:
            bt_matched[best_match] = True
            live_matched[li] = True
            bt_t = bt_trades[best_match]
            dir_match = lt['direction'] == bt_t['direction']
            trade_comparison.append({
                'direction_match': dir_match,
                'live_dir': lt['direction'],
                'bt_dir': bt_t['direction'],
                'live_entry': lt['entry_time'].strftime('%H:%M'),
                'bt_entry': bt_t['entry_time'].strftime('%H:%M'),
                'time_diff_min': best_diff.total_seconds() / 60,
                'live_proba': lt['entry_proba'],
                'bt_proba': bt_t['entry_proba'],
            })
    
    bt_unmatched = [bt_trades[i] for i in range(len(bt_trades)) if not bt_matched[i]]
    live_unmatched = [live_trades[i] for i in range(len(live_trades)) if not live_matched[i]]
    
    # 6. Statistics
    n_live = len(live_trades)
    n_bt = len(bt_trades)
    n_matched = len(trade_comparison)
    
    if signal_comparison:
        signal_match_rate = sum(1 for s in signal_comparison if s['match']) / len(signal_comparison) * 100
    else:
        signal_match_rate = 0
    
    if n_matched > 0:
        dir_match_rate = sum(1 for t in trade_comparison if t['direction_match']) / n_matched * 100
    else:
        dir_match_rate = 0
    
    match_rate = n_matched / max(n_live, n_bt) * 100 if max(n_live, n_bt) > 0 else 0
    
    # Print summary
    print(f"\n  📊 Signal Match: {sum(1 for s in signal_comparison if s['match'])}/{len(signal_comparison)} ({signal_match_rate:.1f}%)")
    print(f"  📊 Trade Match:  {n_matched}/{max(n_live,n_bt)} ({match_rate:.1f}%)")
    print(f"  📊 Direction Match: {sum(1 for t in trade_comparison if t['direction_match'])}/{n_matched} ({dir_match_rate:.1f}%)")
    
    if signal_comparison:
        print(f"\n  ── Signal Detail (first 15) ──")
        print(f"  {'Time':6s} {'LiveP':6s} {'LiveS':6s} {'BarP':6s} {'BarS':6s} {'Match':4s} {'Δ(s)':6s}")
        print(f"  {'-'*40}")
        for sc in signal_comparison[:15]:
            icon = '✅' if sc['match'] else '❌'
            print(f"  {sc['time']:6s} {sc['live_proba']:.4f} {sc['live_signal']:4d} {sc['bar_proba']:.4f} {sc['bar_signal']:4d} {icon:4s} {sc['time_diff_s']:4.0f}s")
    
    if trade_comparison:
        print(f"\n  ── Trade Detail (first 10) ──")
        print(f"  {'LiveDir':7s} {'BTDir':7s} {'Dir?':4s} {'LiveEntry':9s} {'BTEntry':9s} {'Δ(min)':7s} {'LivePro':7s} {'BTPro':7s}")
        print(f"  {'-'*55}")
        for tc in trade_comparison[:10]:
            icon = '✅' if tc['direction_match'] else '❌'
            print(f"  {tc['live_dir']:>7s} {tc['bt_dir']:>7s} {icon:4s} {tc['live_entry']:>9s} {tc['bt_entry']:>9s} {tc['time_diff_min']:6.1f}m {tc['live_proba']:.4f} {tc['bt_proba']:.4f}")
    
    return {
        'symbol': symbol,
        'n_signal_samples': len(signal_comparison),
        'signal_match_pct': signal_match_rate,
        'n_live_trades': n_live,
        'n_bt_trades': n_bt,
        'n_matched_trades': n_matched,
        'trade_match_pct': match_rate,
        'trade_dir_match_pct': dir_match_rate,
        'bt_unmatched': len(bt_unmatched),
        'live_unmatched': len(live_unmatched),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', '-s', default=None)
    parser.add_argument('--list', action='store_true', help='List symbols with data ready')
    args = parser.parse_args()
    
    if args.list:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.execute('''
            SELECT symbol, COUNT(*) as cnt
            FROM live_trades
            WHERE exit_reason = 'hold_expiry' AND entry_proba != 0.5
            GROUP BY symbol ORDER BY cnt DESC
        ''')
        print("Symbols with model-driven trades:")
        for r in c:
            print(f"  {r[0]:12s}: {r[1]} trades")
        print()
        conn.close()
        return
    
    symbols = [args.symbol] if args.symbol else LIVE_SYMBOLS
    
    print(f"{'='*70}")
    print(f"  BACKTEST vs LIVE TRADER — COMPARISON V2")
    print(f"{'='*70}")
    
    all_results = []
    for sym in symbols:
        result = compare_for_symbol(sym)
        if result and not result.get('error'):
            all_results.append(result)
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    
    if all_results:
        avg_signal_match = np.mean([r['signal_match_pct'] for r in all_results])
        avg_trade_match = np.mean([r['trade_match_pct'] for r in all_results])
        avg_dir_match = np.mean([r['trade_dir_match_pct'] for r in all_results])
        
        total_live = sum(r['n_live_trades'] for r in all_results)
        total_bt = sum(r['n_bt_trades'] for r in all_results)
        total_matched = sum(r['n_matched_trades'] for r in all_results)
        
        print(f"  Symbols: {len(all_results)}")
        print(f"  Avg Signal Match Rate:   {avg_signal_match:.1f}%")
        print(f"  Avg Trade Match Rate:    {avg_trade_match:.1f}%")
        print(f"  Avg Direction Match Rate: {avg_dir_match:.1f}%")
        print(f"  Total Live Trades (model): {total_live}")
        print(f"  Total BT Trades:           {total_bt}")
        print(f"  Total Matched:             {total_matched}")
        print(f"  Overall Match Rate:        {total_matched/max(total_live,total_bt)*100:.1f}%" if max(total_live,total_bt) > 0 else "")
    
    print()


if __name__ == '__main__':
    main()
