#!/usr/bin/env python3
"""Check PnL consistency between Mimia DB and Binance testnet."""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(str(ROOT / '.env'))

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict

# Connect DB
DB_PATH = ROOT / 'data' / 'live_trading.db'
db = sqlite3.connect(str(DB_PATH))
cur = db.execute('''
    SELECT symbol, direction, entry_price, exit_price, qty, pnl_net, pnl_pct, 
           entry_time, exit_time, exit_reason 
    FROM live_trades WHERE exit_time IS NOT NULL AND pnl_net IS NOT NULL 
    ORDER BY exit_time
''')
cols = [d[0] for d in cur.description]
db_trades = [dict(zip(cols, r)) for r in cur.fetchall()]
db.close()

print(f"=== MIMIA DB: {len(db_trades)} closed trades ===")
print()

# Group by symbol
by_sym = defaultdict(list)
for t in db_trades:
    by_sym[t['symbol']].append(t)

# Per-symbol stats
total_db_pnl = 0.0
for sym in sorted(by_sym.keys()):
    trades = by_sym[sym]
    tot_pnl = sum(t['pnl_net'] for t in trades)
    total_db_pnl += tot_pnl
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    wr = wins / len(trades) * 100 if trades else 0
    print(f"  {sym:<12s} {len(trades):>3} trades | PnL: ${tot_pnl:<+8.2f} | WR: {wr:.0f}%")

print(f"\n  {'TOTAL':<12s} {len(db_trades):>3} trades | PnL: ${total_db_pnl:<+8.2f}")
print()

# Now get Binance data
print("=== BINANCE TESTNET DATA ===")
from src.utils.binance_client import BinanceRESTClient
client = BinanceRESTClient(testnet=True)

# Wallet balance
acct = client.get_account_info()
wallet = float(acct.get('total_margin_balance', 0))
print(f"  Wallet balance (total_margin_balance): ${wallet:.2f}")

# Income history for REALIZED_PNL
print(f"\n  Fetching income history...")
try:
    income = client.get_income_history(income_type='REALIZED_PNL', limit=500)
    binance_realized = sum(float(i.get('income', 0)) for i in income)
    print(f"  Binance REALIZED_PNL total: ${binance_realized:.2f}")
    print(f"  ({len(income)} income records)")
    
    # Per-symbol breakdown
    inc_by_sym = defaultdict(float)
    for i in income:
        inc_by_sym[i.get('symbol', 'UNKNOWN')] += float(i.get('income', 0))
    
    print(f"\n  Per-symbol REALIZED_PNL (Binance):")
    for sym in sorted(inc_by_sym.keys()):
        print(f"    {sym:<12s} ${inc_by_sym[sym]:<+10.2f}")
except Exception as e:
    print(f"  ❌ Income history failed: {e}")
    binance_realized = None

# Commission (fees)
try:
    fee_income = client.get_income_history(income_type='COMMISSION', limit=500)
    total_commission = sum(float(i.get('income', 0)) for i in fee_income)
    print(f"\n  Total COMMISSION (fees paid): ${total_commission:.2f}")
    
    funding_income = client.get_income_history(income_type='FUNDING_FEE', limit=500)
    total_funding = sum(float(i.get('income', 0)) for i in funding_income)
    print(f"  Total FUNDING_FEE (paid/earned): ${total_funding:.2f}")
except Exception as e:
    print(f"  ❌ Fee fetch failed: {e}")
    total_commission = 0
    total_funding = 0

# Account trades for details
print(f"\n  Fetching account_trade_list for recent trades...")
try:
    start_ms = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
    all_trades = []
    for sym in sorted(by_sym.keys()):
        try:
            trades = client.get_account_trades(sym, limit=100, start_time=start_ms)
            all_trades.extend(trades)
            print(f"    {sym:<12s} {len(trades)} trades")
        except Exception as e:
            print(f"    {sym:<12s} ❌ {e}")
    
    realized_from_trades = sum(float(t.get('realizedPnl', 0)) for t in all_trades)
    print(f"\n  REALIZED PnL from account_trade_list: ${realized_from_trades:.2f}")
except Exception as e:
    print(f"  ❌ Account trades failed: {e}")

# Summary
print(f"\n{'='*60}")
print(f"  PnL RECONCILIATION")
print(f"{'='*60}")
print(f"  Starting capital:                        $5,000.00")
print(f"  DB total PnL (from live_trades):         ${total_db_pnl:<+10.2f}")
print(f"  Current wallet (total_margin_balance):   ${wallet:<+10.2f}")
print(f"  Wallet change from $5,000:               ${wallet - 5000:<+10.2f}")
print()
if binance_realized is not None:
    diff = total_db_pnl - binance_realized
    print(f"  DB PnL vs Binance REALIZED_PNL:")
    print(f"    DB:             ${total_db_pnl:<+10.2f}")
    print(f"    Binance:        ${binance_realized:<+10.2f}")
    print(f"    DISCREPANCY:    ${diff:<+10.2f}")
    print()
    print(f"  With fees accounted:")
    print(f"    DB PnL:                             ${total_db_pnl:<+10.2f}")
    print(f"    Commissions:                        ${total_commission:<+10.2f}")
    print(f"    Funding fees:                       ${total_funding:<+10.2f}")
    adj_db = total_db_pnl - total_commission - total_funding
    print(f"    DB PnL - fees:                      ${adj_db:<+10.2f}")
    print(f"    Binance REALIZED_PNL:               ${binance_realized:<+10.2f}")
    print(f"    ADJUSTED DISCREPANCY:               ${adj_db - binance_realized:<+10.2f}")

print(f"\n  Note: Binance REALIZED_PNL = realized PnL from closed positions")
print(f"  Note: Commissions are already netted inside REALIZED_PNL on Binance")
print(f"  Note: Funding fees are separate from trade PnL on Binance")
