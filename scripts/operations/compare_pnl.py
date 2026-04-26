#!/usr/bin/env python3
"""Compare DB entry_price vs Binance actual for all trades."""
import sys, os, sqlite3
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = ROOT / 'data' / 'live_trading.db'

# Get DB trades
db = sqlite3.connect(str(DB_PATH))
cur = db.execute('SELECT symbol, direction, entry_price, exit_price, qty, pnl_net, entry_time, exit_time FROM live_trades WHERE exit_time IS NOT NULL ORDER BY symbol, entry_time')
cols = [d[0] for d in cur.description]
db_trades = [dict(zip(cols, r)) for r in cur.fetchall()]
db.close()

print(f"=== Comparing DB entry_price vs Binance actual ===\n")

from src.utils.binance_client import BinanceRESTClient
client = BinanceRESTClient(testnet=True)

start_ms = int((datetime.now() - timedelta(days=2)).timestamp() * 1000)

# Fetch all Binance trades per symbol
binance_by_sym = {}
for sym in sorted(set(t['symbol'] for t in db_trades)):
    try:
        trades = client.get_account_trades(sym, limit=100, start_time=start_ms)
        # Sort by time
        trades.sort(key=lambda x: x.get('time', 0))
        binance_by_sym[sym] = trades
    except Exception as e:
        print(f"  {sym}: Binance fetch failed: {e}")
        binance_by_sym[sym] = []

# For each DB trade, find matching Binance trades
total_pnl_diff = 0.0
mismatch_count = 0

for t in db_trades:
    sym = t['symbol']
    db_entry = t['entry_price']
    db_exit = t['exit_price']
    db_pnl = t['pnl_net']
    db_qty = t['qty']
    entry_ms = t['entry_time']
    exit_ms = t['exit_time']
    direction = t['direction']
    
    binance_trades = binance_by_sym.get(sym, [])
    
    # Find entry trade on Binance (side SELL for short, BUY for long, matching qty and close time)
    best_entry = None
    best_exit = None
    
    for bt in binance_trades:
        bt_time = bt.get('time', 0)
        bt_side = bt.get('side', '')
        bt_price = float(bt.get('price', 0))
        bt_qty = float(bt.get('qty', 0))
        
        # Entry: SELL for short, BUY for long
        expected_entry_side = 'SELL' if direction == 'short' else 'BUY'
        expected_exit_side = 'BUY' if direction == 'short' else 'SELL'
        
        # Match within 30s window and similar qty
        if bt_side == expected_entry_side and abs(bt_time - entry_ms) < 30000 and abs(bt_qty - db_qty) / max(db_qty, 1) < 0.05:
            if best_entry is None or abs(bt_time - entry_ms) < abs(best_entry['time'] - entry_ms):
                best_entry = bt
        
        if bt_side == expected_exit_side and abs(bt_time - exit_ms) < 30000 and abs(bt_qty - db_qty) / max(db_qty, 1) < 0.05:
            if best_exit is None or abs(bt_time - exit_ms) < abs(best_exit['time'] - exit_ms):
                best_exit = bt
    
    # Calculate actual PnL
    if best_entry and best_exit:
        be_price = float(best_entry['price'])
        bx_price = float(best_exit['price'])
        actual_qty = abs(float(best_entry['qty']))
        
        # PnL for short: (entry - exit) * qty, for long: (exit - entry) * qty
        if direction == 'short':
            raw_pnl = (be_price - bx_price) * actual_qty
        else:
            raw_pnl = (bx_price - be_price) * actual_qty
        
        # Fees: 0.04% per side
        entry_fee = be_price * actual_qty * 0.0004
        exit_fee = bx_price * actual_qty * 0.0004
        actual_pnl = raw_pnl - entry_fee - exit_fee
        
        diff = db_pnl - actual_pnl
        total_pnl_diff += diff
        
        if abs(diff) > 0.10:  # Only show trades with >$0.10 discrepancy
            mismatch_count += 1
            et = datetime.fromtimestamp(t['entry_time']/1000).strftime('%m-%d %H:%M')
            xt = datetime.fromtimestamp(t['exit_time']/1000).strftime('%m-%d %H:%M')
            print(f"  {sym:<12s} DB entry=${db_entry:<8.4f} BINANCE=${be_price:<8.4f}  "
                  f"DB pnl=${db_pnl:<+7.2f} ACTUAL=${actual_pnl:<+7.2f}  "
                  f"DIFF=${diff:<+6.2f}  [{et}→{xt}]")
    else:
        if best_entry is None:
            print(f"  {sym:<12s} ❌ No matching entry found for DB entry=${db_entry:.4f}")
        if best_exit is None:
            print(f"  {sym:<12s} ❌ No matching exit found for DB exit=${db_exit:.4f}")

print(f"\n--- Summary ---")
print(f"  Total DB PnL: ${sum(t['pnl_net'] for t in db_trades):.2f}")
print(f"  Mismatched trades (>$0.10): {mismatch_count}/{len(db_trades)}")
print(f"  Total PnL discrepancy: ${total_pnl_diff:.2f}")
print()

# Show per-symbol comparison
print("=== Per-symbol Actual PnL ===")
cur_sym = None
actual_by_sym = defaultdict(float)
db_by_sym = defaultdict(float)
for t in db_trades:
    sym = t['symbol']
    db_entry = t['entry_price']
    db_qty = t['qty']
    direction = t['direction']
    
    binance_trades = binance_by_sym.get(sym, [])
    expected_entry_side = 'SELL' if direction == 'short' else 'BUY'
    expected_exit_side = 'BUY' if direction == 'short' else 'SELL'
    
    entry_ms = t['entry_time']
    exit_ms = t['exit_time']
    
    best_entry = None
    best_exit = None
    for bt in binance_trades:
        bt_side = bt.get('side', '')
        bt_time = bt.get('time', 0)
        bt_price = float(bt.get('price', 0))
        bt_qty = float(bt.get('qty', 0))
        
        if bt_side == expected_entry_side and abs(bt_time - entry_ms) < 30000 and abs(bt_qty - db_qty) / max(db_qty, 1) < 0.05:
            if best_entry is None or abs(bt_time - entry_ms) < abs(best_entry['time'] - entry_ms):
                best_entry = bt
        if bt_side == expected_exit_side and abs(bt_time - exit_ms) < 30000 and abs(bt_qty - db_qty) / max(db_qty, 1) < 0.05:
            if best_exit is None or abs(bt_time - exit_ms) < abs(best_exit['time'] - exit_ms):
                best_exit = bt
    
    if best_entry and best_exit:
        be_price = float(best_entry['price'])
        bx_price = float(best_exit['price'])
        actual_qty = abs(float(best_entry['qty']))
        
        if direction == 'short':
            raw_pnl = (be_price - bx_price) * actual_qty
        else:
            raw_pnl = (bx_price - be_price) * actual_qty
        
        entry_fee = be_price * actual_qty * 0.0004
        exit_fee = bx_price * actual_qty * 0.0004
        actual_pnl = raw_pnl - entry_fee - exit_fee
    else:
        actual_pnl = 0
    
    db_by_sym[sym] += t['pnl_net']
    actual_by_sym[sym] += actual_pnl

total_actual = sum(actual_by_sym.values())
total_db_total = sum(db_by_sym.values())
for sym in sorted(db_by_sym.keys()):
    diff_sym = db_by_sym[sym] - actual_by_sym[sym]
    print(f"  {sym:<12s} DB=${db_by_sym[sym]:<+8.2f} Actual=${actual_by_sym[sym]:<+8.2f} Diff=${diff_sym:<+8.2f}")

print(f"\n  {'TOTAL':<12s} DB=${total_db_total:<+8.2f} Actual=${total_actual:<+8.2f} Diff=${total_db_total-total_actual:<+8.2f}")
