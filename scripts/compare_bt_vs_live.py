#!/usr/bin/env python3
"""
Backtest vs Live Trader — Signal Comparison
Compare probas & direction for last 10 real trades.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import sqlite3
from datetime import datetime, timedelta
from src.trading.state import DB_PATH
from src.trading.backtest import run_backtest

# Get live trades
conn = sqlite3.connect(str(DB_PATH))
cur = conn.cursor()
cur.execute('''
    SELECT symbol, direction, entry_time, entry_proba
    FROM live_trades 
    WHERE entry_proba NOT IN (0, 0.5) AND exit_reason != 'history_sync'
    ORDER BY entry_time DESC LIMIT 10
''')
live = cur.fetchall()
conn.close()

# Group by symbol and find the 6h window that covers all trades
now = datetime.utcnow()
earliest = min(datetime.utcfromtimestamp(t[2]/1000) for t in live)
# Make sure we cover the earliest trade + 30min buffer
bt_hours = int((now - earliest).total_seconds() / 3600) + 1
bt_hours = max(bt_hours, 3)  # min 3h

print(f"{'='*80}")
print(f"  BACKTEST vs LIVE TRADER — Last {len(live)} Trades")
print(f"  Backtest window: {bt_hours}h (from {earliest.strftime('%H:%M')} to now)")
print(f"{'='*80}")
print(f"{'T#':>3s} {'Symbol':>10s} {'LiveDir':>7s} {'LiveProb':>8s} {'BTProb':>8s} {'BTDir':>7s} {'Diff':>8s} {'Match?':>7s}")
print(f"{'-'*65}")

# Group by symbol to minimize backtest runs
by_symbol = {}
for t in live:
    by_symbol.setdefault(t[0], []).append(t)

match_cnt = 0
total = 0
for sym, trades in by_symbol.items():
    r = run_backtest(sym, test_hours=bt_hours, verbose=False)
    if r is None:
        for t in trades:
            print(f"{'--':>3s} {sym:>10s} {'--':>7s} {'--':>8s} {'--':>8s} {'--':>7s} {'--':>8s} {'NO MODELS':>7s}")
        continue
    
    ts_list = r['timestamps']
    probas = r['probas']
    signals = r['signals']
    
    for t in trades:
        total += 1
        dir_, entry_ms, lp = t[1], t[2], t[3]
        entry_dt = datetime.utcfromtimestamp(entry_ms / 1000)
        # Signal that caused entry at ET was generated at the PREVIOUS bar's close.
        # Live trader: bar closes → generate signal → stored as pending → next bar close → execute entry
        # So signal_time = ET - 5min (the close of the bar before entry)
        # The bar whose close = signal_time has open = signal_time - 5min = ET - 10min
        # In backtest timestamps, timestamps[i] = bar open time
        # So we need probas[i] where timestamps[i] ≈ ET - 10min
        signal_bar_open = entry_dt - timedelta(minutes=10)
        
        # Find the bar in backtest closest to signal_bar_open
        best_idx = None
        best_dist = timedelta(hours=1)
        for idx, ts in enumerate(ts_list):
            dist = abs(ts - signal_bar_open)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        
        if best_idx is None:
            print(f"{'--':>3s} {sym:>10s} {dir_:>7s} {lp:>.4f} {'--':>8s} {'--':>7s} {'--':>8s} {'NO BAR':>7s}")
            continue
        
        bt_proba = probas[best_idx]
        bt_sig = signals[best_idx]
        bt_dir = 'LONG' if bt_sig == 1 else ('SHORT' if bt_sig == -1 else 'FLAT')
        live_dir_short = {'long': 'LONG', 'short': 'SHORT'}.get(dir_.lower(), dir_)
        
        diff = abs(bt_proba - lp)
        dir_match = (bt_sig == 1 and dir_.lower() == 'long') or \
                    (bt_sig == -1 and dir_.lower() == 'short')
        
        match_icon = '✅' if dir_match else '❌'
        if dir_match:
            match_cnt += 1
        
        time_str = entry_dt.strftime('%H:%M')
        print(f"{total:3d} {sym:>10s} {live_dir_short:>7s} {lp:>.4f}    {bt_proba:>.4f}    {bt_dir:>7s} {diff:>.4f}    {match_icon:>7s}")

print(f"{'-'*65}")
print(f"  Direction match: {match_cnt}/{total} ({match_cnt/total*100:.0f}%)")
if match_cnt < total:
    print(f"  ⚠️  {total - match_cnt} trade(s) have DIFFERENT signals!")
else:
    print(f"  ✅ ALL signals match!")
