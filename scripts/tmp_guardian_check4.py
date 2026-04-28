#!/usr/bin/env python3
"""Check #4: Wallet & Drawdown"""
import sqlite3, json, os, sys
sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv('/root/projects/mimia-quant/.env')
from datetime import datetime

db = sqlite3.connect("data/live_trading.db")

# DB capital
cap = db.execute("SELECT capital, peak_capital FROM live_capital ORDER BY id DESC LIMIT 1").fetchone()
if cap:
    dd = (cap[1] - cap[0]) / cap[1] * 100 if cap[1] > 0 else 0
    print(f"Capital: ${cap[0]:.2f} | Peak: ${cap[1]:.2f} | DD: {dd:.2f}%")

# Today's PnL
import time
today_ms = int(time.time() * 1000) - 24*3600*1000
today = db.execute("SELECT COUNT(*), SUM(pnl_net), SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) FROM live_trades WHERE entry_time > ?", (today_ms,)).fetchone()
if today and today[0] > 0:
    wr = today[2]/today[0]*100 if today[0] > 0 else 0
    print(f"Today: {today[0]} trades | WR={wr:.0f}% | PnL=${today[1]:+.2f}")

# Wallet via BinanceRESTClient
from src.utils.binance_client import BinanceRESTClient
client = BinanceRESTClient(testnet=True)
try:
    balances = client.get_balance()
    usdt = next((float(b.get('balance',0)) for b in balances if b.get('asset')=='USDT'), None)
    if usdt is not None:
        print(f"Wallet: ${usdt:.2f}")
    else:
        print("Wallet: not found")
except Exception as e:
    print(f"Wallet error: {e}")

# Funding rate
try:
    fr = client.get_funding_rate('BTCUSDT', limit=1)
    if isinstance(fr, list) and fr:
        rate = float(fr[0].get('fundingRate', 0)) * 100
        print(f"BTC FR: {rate:.4f}%")
    else:
        print("BTC FR: no data")
except Exception as e:
    print(f"FR error: {e}")

db.close()
