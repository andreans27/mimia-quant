#!/usr/bin/env python3
"""Detailed PnL reconciliation - compare DB vs Binance within matching time ranges."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

DB_PATH = ROOT / 'data' / 'live_trading.db'

# 1. Get DB trade time range
db = sqlite3.connect(str(DB_PATH))
cur = db.execute('SELECT MIN(entry_time), MAX(exit_time), COUNT(*) FROM live_trades WHERE exit_time IS NOT NULL')
min_et, max_et, n_trades = cur.fetchone()
print(f"DB trade time range:")
print(f"  Earliest entry: {datetime.fromtimestamp(min_et/1000).strftime('%Y-%m-%d %H:%M:%S') if min_et else 'N/A'}")
print(f"  Latest exit:    {datetime.fromtimestamp(max_et/1000).strftime('%Y-%m-%d %H:%M:%S') if max_et else 'N/A'}")
print(f"  Trade count:    {n_trades}")
print()

# 2. Get DB total PnL
cur = db.execute("SELECT symbol, COUNT(*), SUM(pnl_net), SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) FROM live_trades WHERE exit_time IS NOT NULL AND pnl_net IS NOT NULL GROUP BY symbol ORDER BY symbol")
rows = cur.fetchall()
db_pnl = {}
for r in rows:
    db_pnl[r[0]] = {'count': r[1], 'pnl': r[2], 'wins': r[3]}
    print(f"  DB {r[0]:<12s} {r[1]:>3} trades | PnL: ${r[2]:<+8.2f}")
db.close()

total_db_pnl = sum(v['pnl'] for v in db_pnl.values())
print(f"\n  DB TOTAL PnL: ${total_db_pnl:.2f}")
print()

# 3. Get Binance income history within DB's time range
# Use the DB's entry_time as start (with buffer) and exit_time as end (with buffer)
start_ms = max(0, (min_et or 0) - 3600000)  # 1h buffer before earliest entry
end_ms = (max_et or 9999999999999) + 3600000  # 1h buffer after latest exit

print(f"Filtering Binance data from {datetime.fromtimestamp(start_ms/1000)} to {datetime.fromtimestamp(end_ms/1000)}")
print()

from src.utils.binance_client import BinanceRESTClient
client = BinanceRESTClient(testnet=True)

# Fetch income history with time filters
print("=== BINANCE REALIZED_PNL (time-filtered) ===")
try:
    income = client.get_income_history(
        income_type='REALIZED_PNL', limit=500,
        start_time=start_ms, end_time=end_ms
    )
    print(f"  Records: {len(income)}")
    
    inc_by_sym = defaultdict(float)
    inc_count = defaultdict(int)
    total_binance_pnl = 0.0
    
    # Sort by time for display
    income_sorted = sorted(income, key=lambda x: x.get('time', 0))
    for i in income_sorted:
        sym = i.get('symbol', 'UNKNOWN')
        pnl = float(i.get('income', 0))
        inc_by_sym[sym] += pnl
        inc_count[sym] += 1
        total_binance_pnl += pnl
    
    for sym in sorted(inc_by_sym.keys()):
        print(f"  {sym:<12s} {inc_count[sym]:>3} records | PnL: ${inc_by_sym[sym]:<+10.2f}")
    
    print(f"\n  BINANCE TOTAL (time-filtered): ${total_binance_pnl:.2f}")
    
except Exception as e:
    print(f"  ❌ {e}")
    income = []
    total_binance_pnl = 0

# 4. Compare
print(f"\n{'='*60}")
print(f"  RECONCILIATION (same time period)")
print(f"{'='*60}")
print(f"  DB PnL:           ${total_db_pnl:<+10.2f}")
print(f"  Binance PnL:      ${total_binance_pnl:<+10.2f}")
diff = total_db_pnl - total_binance_pnl
print(f"  DISCREPANCY:      ${diff:<+10.2f}")

# 5. Per-symbol comparison
print(f"\n{'='*60}")
print(f"  PER-SYMBOL COMPARISON")
print(f"{'='*60}")
print(f"  {'SYMBOL':<12s} {'DB PnL':>10s} {'BINANCE':>10s} {'DIFF':>10s} {'DB Trades':>10s} {'BIN Recs':>10s}")
print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for sym in sorted(set(list(db_pnl.keys()) + list(inc_by_sym.keys()))):
    db_val = db_pnl.get(sym, {}).get('pnl', 0)
    bi_val = inc_by_sym.get(sym, 0)
    d = db_val - bi_val
    db_cnt = db_pnl.get(sym, {}).get('count', 0)
    bi_cnt = inc_count.get(sym, 0)
    print(f"  {sym:<12s} ${db_val:<+8.2f} ${bi_val:<+8.2f} ${d:<+8.2f} {db_cnt:>10} {bi_cnt:>10}")

# 6. Show sample trades for a few symbols to investigate the granular mismatch
print(f"\n{'='*60}")
print(f"  SAMPLE DB TRADES (last 10)")
print(f"{'='*60}")
db = sqlite3.connect(str(DB_PATH))
cur = db.execute('''
    SELECT symbol, direction, entry_price, exit_price, qty, pnl_net, pnl_pct, exit_time, entry_time
    FROM live_trades WHERE exit_time IS NOT NULL AND pnl_net IS NOT NULL
    ORDER BY exit_time DESC LIMIT 10
''')
cols = [d[0] for d in cur.description]
for r in cur.fetchall():
    d = dict(zip(cols, r))
    et = datetime.fromtimestamp(d['exit_time']/1000).strftime('%m-%d %H:%M')
    print(f"  {d['symbol']:<12s} {d['direction']:>4s} ${d['entry_price']:<8.4f} → ${d['exit_price']:<8.4f} qty={d['qty']:<8.2f} pnl=${d['pnl_net']:<+7.2f} [{et}]")
db.close()

# 7. Show sample Binance income records
print(f"\n{'='*60}")
print(f"  SAMPLE BINANCE INCOME RECORDS (last 10)")
print(f"{'='*60}")
for i in sorted(income, key=lambda x: x.get('time', 0))[-10:]:
    ts = datetime.fromtimestamp(i.get('time', 0)/1000).strftime('%m-%d %H:%M:%S')
    pnl = float(i.get('income', 0))
    sym = i.get('symbol', '?')
    print(f"  {sym:<12s} ${pnl:<+8.2f} [{ts}]")

# 8. Check wallet
print(f"\n{'='*60}")
print(f"  WALLET RECONCILIATION")
print(f"{'='*60}")
acct = client.get_account_info()
wallet = float(acct.get('total_margin_balance', 0))
print(f"  Current wallet: ${wallet:.2f}")
print(f"  Expected (5000 + Binance PnL): ${5000 + total_binance_pnl:.2f}")
# Also get unrealized PnL
print(f"\n  Open positions:")
pos = client.get_position_info()
for p in pos:
    upnl = float(p.get('unRealizedProfit', 0))
    sym = p.get('symbol', '?')
    amt = float(p.get('positionAmt', 0))
    if abs(amt) > 0:
        print(f"    {sym:<12s} {amt:<+10.3f} unrealized: ${upnl:<+8.2f}")
