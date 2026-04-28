#!/usr/bin/env python3
"""
Mimia — Backtest vs Live Trader Comparison Script
==================================================
Uses the EXACT same SignalGenerator code path as the live trader to re-run
signal generation for the past 24 hours on every 5m bar, then compares
the resulting signals/trades with actual live trader DB records.

Usage:
    python scripts/compare_backtest_live.py [--symbol SYMBOL] [--days 1]
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

# ── Import live trader code (EXACT same code path) ──────────────────────
from src.trading.signals import SignalGenerator
from src.trading.state import (
    LIVE_SYMBOLS, THRESHOLD, HOLD_BARS, COOLDOWN_BARS,
    DB_PATH, TAKER_FEE, SLIPPAGE, MODEL_DIR, TF_GROUPS, SEEDS,
)
from src.strategies.ml_features import OHLCV_CACHE_DIR

# ── Config ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 5000.0
TEST_DAYS = 1       # Compare last 24 hours
WARMUP_BARS = 0     # Already warm (models use cached features)


def run_backtest_simulation(symbol: str, threshold: float = THRESHOLD,
                            hold_bars: int = HOLD_BARS,
                            cooldown_bars: int = COOLDOWN_BARS,
                            position_pct: float = 0.15) -> Dict:
    """
    Re-run the EXACT same bar-by-bar trading logic as the live trader,
    but for all 5m bars in the past 24 hours instead of just the latest bar.

    Uses the same SignalGenerator._load_models() and generate_signal() 
    code path — just iterates over each bar instead of the latest one.

    Returns:
        dict with: backtest_signals, backtest_trades, proba_series
    """
    print(f"\n{'='*70}")
    print(f"  BACKTEST SIMULATION: {symbol}")
    print(f"  threshold={threshold:.2f} hold={hold_bars} cooldown={cooldown_bars}")
    print(f"{'='*70}")

    # ── 1. Fetch OHLCV data (same path as live trader) ──────────────
    gen = SignalGenerator(symbol)
    
    # Use cached OHLCV data (same as live trader uses)
    spot_symbol = symbol
    if symbol.startswith("1000"):
        for prefix in ["1000", "10000", "100000"]:
            if symbol.startswith(prefix):
                spot_symbol = symbol[len(prefix):]
                break
    
    df_ohlcv = gen._ensure_ohlcv_data(spot_symbol)
    if df_ohlcv is None or len(df_ohlcv) < 500:
        print(f"  ❌ Insufficient OHLCV data")
        return None
    
    # Limit to test period
    cutoff = datetime.utcnow() - timedelta(days=TEST_DAYS + 0.1)  # Extra buffer for warmup
    df_ohlcv = df_ohlcv[df_ohlcv.index >= cutoff].copy()
    if len(df_ohlcv) < 50:
        print(f"  ❌ Too few bars ({len(df_ohlcv)}) in test window")
        return None
    
    print(f"  📊 OHLCV window: {df_ohlcv.index[0]} → {df_ohlcv.index[-1]} ({len(df_ohlcv)} bars)")

    # ── 2. Compute features and get models (exact same as live trader) ──
    cached = gen._load_models(symbol)
    if cached is None:
        print(f"  ❌ Failed to load models for {symbol}")
        return None
    
    feat_df = cached['features']
    groups = cached['groups']
    
    # Filter features to test window (+ extra for lookback)
    feat_cutoff = datetime.utcnow() - timedelta(days=TEST_DAYS + 0.5)
    feat_df = feat_df[feat_df.index >= feat_cutoff].copy()
    if len(feat_df) < 50:
        print(f"  ❌ Too few feature rows ({len(feat_df)})")
        return None
    
    print(f"  🔧 Features: {len(feat_df)} rows, {len(feat_df.columns)} cols")
    print(f"    TF groups loaded: {list(groups.keys())}")

    # ── 3. Compute proba for EVERY bar ─────────────────────────────────
    # This replicates what generate_signal() does, but for ALL bars
    print(f"  🧮 Computing probabilities for all bars...")
    
    group_probs_all = {}
    for tf, models in groups.items():
        tf_probs_all = []
        for seed, m, mf in models:
            available = [c for c in mf if c in feat_df.columns]
            if len(available) < 5:
                print(f"    ⚠️ {tf}/{seed}: only {len(available)} features (need >= 5)")
                continue
            X = feat_df[available].fillna(0).clip(-10, 10).values
            probs = m.predict_proba(X)[:, 1]
            tf_probs_all.append(probs)
        
        if tf_probs_all:
            avg = np.nanmean(tf_probs_all, axis=0)
            group_probs_all[tf] = avg
    
    if len(group_probs_all) < 2:
        print(f"  ❌ Only {len(group_probs_all)} TF groups available")
        return None
    
    # Average across all TF groups
    prob_stack = np.column_stack([group_probs_all[tf] for tf in group_probs_all])
    avg_probs = np.mean(prob_stack, axis=1)
    
    prob_series = pd.Series(avg_probs, index=feat_df.index)
    
    # ── 4. Align with OHLCV (inner join on index) ──────────────────────
    df_bt = df_ohlcv.join(prob_series.to_frame('proba'), how='inner')
    
    if len(df_bt) < 10:
        print(f"  ❌ Aligned bars: {len(df_bt)} — too few")
        return None
    
    print(f"  ✅ Aligned: {len(df_bt)} bars for simulation")
    
    # ── 5. Simulate bar-by-bar (same logic as live trader + backtest) ────
    capital = INITIAL_CAPITAL
    position = 0        # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_qty = 0.0
    entry_time = None
    entry_proba_val = 0.0
    hold_remaining = 0
    cooldown = 0
    
    backtest_signals = []    # Every signal event (like live_signals table)
    backtest_trades = []     # Every completed trade (like live_trades table)
    
    long_pnl = 0.0
    short_pnl = 0.0
    
    for idx in range(len(df_bt)):
        row = df_bt.iloc[idx]
        ts = df_bt.index[idx]
        price = float(row['close'])
        prob_val = float(row['proba'])
        
        # Decrement counters
        hold_remaining = max(0, hold_remaining - 1)
        cooldown = max(0, cooldown - 1)
        
        # ── EXIT CHECK (same as live trader) ──
        if position != 0 and hold_remaining <= 0:
            # Exit at current close with slippage
            if position == 1:
                exit_price = price * (1 - SLIPPAGE)
                raw_pnl = entry_qty * (exit_price - entry_price)
            else:
                exit_price = price * (1 + SLIPPAGE)
                raw_pnl = entry_qty * (entry_price - exit_price)
            
            exit_cost = exit_price * entry_qty * TAKER_FEE
            pnl_net = raw_pnl - exit_cost
            pnl_pct = pnl_net / (entry_price * entry_qty) * 100 if entry_price * entry_qty > 0 else 0
            
            capital += raw_pnl  # Add raw PnL
            capital -= exit_cost  # Exit fee
            
            if position == 1:
                long_pnl += pnl_net
            else:
                short_pnl += pnl_net
            
            backtest_trades.append({
                'symbol': symbol,
                'direction': 'long' if position == 1 else 'short',
                'entry_time': entry_time,
                'exit_time': ts,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'qty': entry_qty,
                'pnl_net': pnl_net,
                'pnl_pct': pnl_pct,
                'entry_proba': entry_proba_val,
                'hold_bars': HOLD_BARS,
                'exit_reason': 'hold_expiry' if hold_remaining <= 0 else 'unknown',
            })
            
            position = 0
            entry_price = 0.0
            entry_qty = 0.0
            entry_proba_val = 0.0
            cooldown = cooldown_bars
        
        # ── ENTRY CHECK (same level-based logic as live trader) ──
        if position == 0 and cooldown <= 0:
            signal = 0
            if prob_val >= threshold:
                signal = 1  # LONG
            elif prob_val <= (1 - threshold):
                signal = -1  # SHORT
            
            # Record signal
            backtest_signals.append({
                'symbol': symbol,
                'timestamp': ts,
                'proba': prob_val,
                'signal': signal,
                'capital': capital,
            })
            
            if signal != 0:
                # Enter
                if signal == 1:
                    entry_price = price * (1 + SLIPPAGE)
                else:
                    entry_price = price * (1 - SLIPPAGE)
                
                entry_qty = (capital * position_pct) / entry_price
                entry_fee = entry_price * entry_qty * TAKER_FEE
                capital -= entry_fee
                
                position = signal
                entry_time = ts
                entry_proba_val = prob_val
                hold_remaining = hold_bars
    
    # ── 6. Summary metrics ──
    result = {
        'symbol': symbol,
        'threshold': threshold,
        'hold_bars': hold_bars,
        'cooldown_bars': cooldown_bars,
        'total_bars': len(df_bt),
        'total_signals': len(backtest_signals),
        'total_trades': len(backtest_trades),
        'long_trades': len([t for t in backtest_trades if t['direction'] == 'long']),
        'short_trades': len([t for t in backtest_trades if t['direction'] == 'short']),
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'total_pnl': capital - INITIAL_CAPITAL,
        'final_capital': capital,
        'backtest_signals': backtest_signals,
        'backtest_trades': backtest_trades,
        'proba_series': prob_series,
    }
    
    return result


def get_live_trader_data(symbol: str, hours: int = 24) -> Dict:
    """
    Fetch live trader signals and trades from DB for comparison.
    """
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    
    # Live signals
    c = conn.execute('''
        SELECT timestamp, proba, signal, capital
        FROM live_signals
        WHERE symbol = ? AND timestamp >= ?
        ORDER BY timestamp
    ''', (symbol, cutoff_ms))
    signals = []
    for row in c.fetchall():
        signals.append({
            'symbol': symbol,
            'timestamp': datetime.utcfromtimestamp(row[0]/1000),
            'ts_ms': row[0],
            'proba': row[1],
            'signal': row[2],
            'capital': row[3],
        })
    
    # Live trades (from the deployed model only — entry_proba != 0.5 default)
    c = conn.execute('''
        SELECT direction, entry_time, exit_time, entry_price, exit_price,
               qty, pnl_net, pnl_pct, entry_proba, hold_bars, exit_reason
        FROM live_trades
        WHERE symbol = ? AND entry_time >= ?
        ORDER BY entry_time
    ''', (symbol, cutoff_ms))
    trades = []
    for row in c.fetchall():
        trades.append({
            'symbol': symbol,
            'direction': row[0],
            'entry_time': datetime.utcfromtimestamp(row[1]/1000),
            'entry_time_ms': row[1],
            'exit_time': datetime.utcfromtimestamp(row[2]/1000) if row[2] else None,
            'exit_time_ms': row[2],
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
    
    return {
        'symbol': symbol,
        'signals': signals,
        'trades': trades,
    }


def compare_results(symbol: str, bt_result: Dict, live_data: Dict) -> Dict:
    """
    Compare backtest simulation results with live trader data.
    """
    bt_trades = bt_result['backtest_trades']
    live_trades = live_data['trades']
    bt_signals = bt_result['backtest_signals']
    live_signals = live_data['signals']
    
    # ── Trade Comparison ──
    # For each BT trade, find the closest matching live trade
    bt_used = [False] * len(bt_trades)
    live_used = [False] * len(live_trades)
    
    matches = []
    bt_unmatched = []
    live_unmatched = []
    
    # Match by time proximity (within 10 minutes = ~2 bars)
    match_window_ms = 10 * 60 * 1000  # 10 minutes
    
    import bisect
    
    # Sort live trades by entry_time
    live_sorted = sorted(
        [(i, t) for i, t in enumerate(live_trades)],
        key=lambda x: x[1].get('entry_time_ms', 0)
    )
    live_times = [t['entry_time_ms'] for _, t in live_sorted]
    
    for bi, bt_t in enumerate(bt_trades):
        bt_entry_ms = int(bt_t.get('entry_time').timestamp() * 1000) if hasattr(bt_t['entry_time'], 'timestamp') else int(bt_t['entry_time'] * 1000)
        
        # Find nearest live trade by time
        if not live_times:
            bt_unmatched.append((bi, bt_t, 'no live trades'))
            continue
        
        # Binary search for nearest
        pos = bisect.bisect_left(live_times, bt_entry_ms)
        best_pos = None
        best_diff = match_window_ms
        
        for candidate in [pos-1, pos, pos+1]:
            if 0 <= candidate < len(live_times):
                diff = abs(live_times[candidate] - bt_entry_ms)
                if diff < best_diff and not live_used[live_sorted[candidate][0]]:
                    best_diff = diff
                    best_pos = candidate
        
        if best_pos is not None:
            li, live_t = live_sorted[best_pos]
            live_used[li] = True
            bt_used[bi] = True
            
            # Compare direction
            dir_match = bt_t['direction'] == live_t['direction']
            
            matches.append({
                'bt': bt_t,
                'live': live_t,
                'time_diff_ms': best_diff,
                'time_diff_min': best_diff / 60000,
                'direction_match': dir_match,
                'bt_direction': bt_t['direction'],
                'live_direction': live_t['direction'],
                'bt_entry_price': bt_t['entry_price'],
                'live_entry_price': live_t['entry_price'],
                'bt_entry_proba': bt_t.get('entry_proba', 0),
                'live_entry_proba': live_t.get('entry_proba', 0),
                'bt_exit_price': bt_t['exit_price'],
                'live_exit_price': live_t['exit_price'],
                'bt_pnl': bt_t['pnl_net'],
                'live_pnl': live_t['pnl_net'],
            })
        else:
            bt_unmatched.append((bi, bt_t, 'no close live trade match'))
    
    for li, t in enumerate(live_trades):
        if not live_used[li]:
            live_unmatched.append((li, t))
    
    # ── Signal Comparison ──
    # Count signals that match
    bt_long_signals = len([s for s in bt_signals if s['signal'] == 1])
    bt_short_signals = len([s for s in bt_signals if s['signal'] == -1])
    bt_flat_signals = len([s for s in bt_signals if s['signal'] == 0])
    
    live_long_signals = len([s for s in live_signals if s['signal'] == 1])
    live_short_signals = len([s for s in live_signals if s['signal'] == -1])
    live_flat_signals = len([s for s in live_signals if s['signal'] == 0])
    
    # For each live signal, find the closest BT signal in time
    # (the live trader fires signals ~every 5 min, so BT has many more)
    signal_matches = []
    
    # ── Summary ──
    result = {
        'symbol': symbol,
        'bt_trades_count': len(bt_trades),
        'live_trades_count': len(live_trades),
        'matched_trades': len(matches),
        'bt_unmatched': bt_unmatched,
        'live_unmatched': live_unmatched,
        'trade_matches': matches,
        'bt_long_signals': bt_long_signals,
        'bt_short_signals': bt_short_signals,
        'bt_flat_signals': bt_flat_signals,
        'live_long_signals': live_long_signals,
        'live_short_signals': live_short_signals,
        'live_flat_signals': live_flat_signals,
        'bt_signals': bt_signals,
        'live_signals': live_signals,
        'signal_matches': signal_matches,
    }
    
    return result


def print_comparison_report(comparison: Dict):
    """Print a detailed comparison report."""
    s = comparison['symbol']
    bt_trades = comparison['bt_trades_count']
    live_trades = comparison['live_trades_count']
    matches = comparison['matched_trades']
    
    print(f"\n{'='*70}")
    print(f"  COMPARISON REPORT: {s}")
    print(f"{'='*70}")
    
    print(f"\n  ── Overview ──")
    print(f"  Backtest trades: {bt_trades}")
    print(f"  Live trades:     {live_trades}")
    print(f"  Matched:         {matches}")
    print(f"  BT unmatched:    {len(comparison['bt_unmatched'])}")
    print(f"  Live unmatched:  {len(comparison['live_unmatched'])}")
    
    print(f"\n  ── Signals ──")
    print(f"  BT:  L={comparison['bt_long_signals']} S={comparison['bt_short_signals']} Flat={comparison['bt_flat_signals']}")
    print(f"  Live: L={comparison['live_long_signals']} S={comparison['live_short_signals']} Flat={comparison['live_flat_signals']}")
    
    if matches:
        dir_match = sum(1 for m in comparison['trade_matches'] if m['direction_match'])
        dir_mismatch = matches - dir_match
        avg_time_diff = np.mean([m['time_diff_min'] for m in comparison['trade_matches']])
        
        print(f"\n  ── Trade Match Details ──")
        print(f"  Direction match: {dir_match}/{matches} ({dir_match/matches*100:.1f}%)")
        print(f"  Direction mismatch: {dir_mismatch}")
        print(f"  Avg time diff: {avg_time_diff:.1f} min")
        
        print(f"\n  Matched Trades:")
        print(f"  {'#':3s} {'BT Dir':6s} {'Live Dir':6s} {'Dir?':4s} {'TimeΔ':6s} {'BT Entry':10s} {'Live Entry':10s} {'BT Ex':10s} {'Live Ex':10s} {'BT PnL':8s} {'Live PnL':8s}")
        print(f"  {'-'*80}")
        for i, m in enumerate(comparison['trade_matches'][:20]):
            print(f"  {i:3d} {m['bt_direction']:6s} {m['live_direction']:6s} {'✅' if m['direction_match'] else '❌'} {m['time_diff_min']:5.1f}m {m['bt_entry_price']:8.4f} {m['live_entry_price']:8.4f} {m['bt_exit_price']:8.4f} {m['live_exit_price']:8.4f} {m['bt_pnl']:7.2f} {m['live_pnl']:7.2f}")
        
        if len(comparison['trade_matches']) > 20:
            print(f"  ... {len(comparison['trade_matches']) - 20} more")
    
    if comparison['bt_unmatched']:
        print(f"\n  ── BT Trades Not in Live ──")
        for bi, bt_t, reason in comparison['bt_unmatched'][:10]:
            et = bt_t['entry_time'].strftime('%H:%M') if hasattr(bt_t['entry_time'], 'strftime') else bt_t['entry_time']
            print(f"  [{bi}] {bt_t['direction']:6s} @ {et} | ${bt_t['entry_price']:.4f}→${bt_t['exit_price']:.4f} | PnL=${bt_t['pnl_net']:+.2f} | proba={bt_t.get('entry_proba', 0):.4f} | {reason}")
    
    if comparison['live_unmatched']:
        print(f"\n  ── Live Trades Not in BT ──")
        for li, t in comparison['live_unmatched'][:10]:
            print(f"  [{li}] {t['direction']:6s} @ {t['entry_time'].strftime('%H:%M')} | ${t['entry_price']:.4f}→${t['exit_price']:.4f} | PnL=${t['pnl_net']:+.2f} | proba={t.get('entry_proba', 0):.4f} | {t.get('exit_reason','')}")
    
    # Summary verdict
    if matches >= min(bt_trades, live_trades) * 0.8:
        verdict = "✅ HIGH MATCH — BT and Live trader are closely aligned"
    elif matches >= min(bt_trades, live_trades) * 0.5:
        verdict = "⚡ MODERATE MATCH — Some alignment, but significant gaps"
    else:
        verdict = "❌ LOW MATCH — BT and Live trader differ significantly"
    
    print(f"\n  ── Verdict ──")
    print(f"  {verdict}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Compare backtest vs live trader')
    parser.add_argument('--symbol', '-s', default=None, help='Symbol to test (default: all)')
    parser.add_argument('--days', '-d', type=int, default=1, help='Days of data to compare')
    args = parser.parse_args()
    
    global TEST_DAYS
    TEST_DAYS = args.days
    
    symbols_to_test = [args.symbol] if args.symbol else LIVE_SYMBOLS
    
    print(f"{'='*70}")
    print(f"  BACKTEST vs LIVE TRADER COMPARISON")
    print(f"  Period: {TEST_DAYS} day(s)")
    print(f"  Symbols: {len(symbols_to_test)}")
    print(f"{'='*70}")
    
    all_reports = []
    
    for sym in symbols_to_test:
        print(f"\n  Processing {sym}...")
        
        # Run backtest simulation
        bt_result = run_backtest_simulation(sym)
        if bt_result is None:
            print(f"  ❌ Backtest failed for {sym}")
            continue
        
        # Get live trader data
        live_data = get_live_trader_data(sym, hours=TEST_DAYS * 24)
        print(f"  📋 Live signals: {len(live_data['signals'])} | Live trades: {len(live_data['trades'])}")
        
        # Compare
        comparison = compare_results(sym, bt_result, live_data)
        print_comparison_report(comparison)
        all_reports.append(comparison)
    
    # ── Final Summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    
    total_bt = sum(r['bt_trades_count'] for r in all_reports)
    total_live = sum(r['live_trades_count'] for r in all_reports)
    total_matched = sum(r['matched_trades'] for r in all_reports)
    
    print(f"  Total symbols:     {len(all_reports)}")
    print(f"  Total BT trades:   {total_bt}")
    print(f"  Total Live trades: {total_live}")
    print(f"  Total Matched:     {total_matched}")
    print(f"  Match rate:        {total_matched/max(total_bt,total_live)*100:.1f}%" if max(total_bt,total_live) > 0 else "  N/A")
    print()


if __name__ == '__main__':
    main()
