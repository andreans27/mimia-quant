#!/usr/bin/env python3
"""Guardian check: Signal Quality, Wallet, Retrain Status"""
import json, sqlite3, sys, os
sys.path.insert(0, "/root/projects/mimia-quant")
os.chdir("/root/projects/mimia-quant")
from dotenv import load_dotenv
load_dotenv("/root/projects/mimia-quant/.env")
from datetime import datetime, timezone

# Retrain status
print("=== CHECK 3: SIGNAL QUALITY & RETRAIN ===")
try:
    with open("data/ml_models/_retrain_status.json") as f:
        status = json.load(f)
except Exception as e:
    print(f"Could not load retrain status: {e}")
    status = {'symbols': {}, 'runs': []}

deployed_at = status.get('deployed_at', '')
if deployed_at:
    dt = datetime.fromisoformat(deployed_at.replace('Z','+00:00'))
    hours_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    print(f"Deployed: {hours_ago:.0f}h ago")

symbols = sorted(status.get('symbols', {}).keys())
if not symbols:
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']
print(f"Tracked symbols: {len(symbols)}")

# Retrain history
runs = status.get('runs', [])
for r in runs[-5:]:
    dt = r.get('time','')
    sym = r.get('symbol','')
    wr = r.get('metrics',{}).get('wr','?')
    pf = r.get('metrics',{}).get('pf','?')
    print(f"  Retrain {sym}: WR={wr}% PF={pf} [{dt}]")

# Live WR dari DB
db = sqlite3.connect("data/live_trading.db")
for sym in symbols[:20]:
    trades = db.execute("SELECT pnl_net FROM live_trades WHERE symbol=? ORDER BY entry_time DESC LIMIT 200", (sym,)).fetchall()
    if len(trades) >= 10:
        wins = sum(1 for t in trades if t[0] > 0)
        wr = wins / len(trades)
        profit = sum(t[0] for t in trades if t[0] > 0)
        loss = abs(sum(t[0] for t in trades if t[0] < 0))
        pf = profit / loss if loss > 0 else 99
        bt_wr_raw = status.get('symbols',{}).get(sym,{}).get('wr', 70)
        bt_wr = bt_wr_raw / 100.0 if bt_wr_raw > 1 else bt_wr_raw
        action = 'GREEN' if wr >= 0.60 and pf >= 2.0 else ('YELLOW' if wr >= 0.55 else 'RED')
        print(f"  {sym}: WR={wr:.0%} PF={pf:.2f} BT={bt_wr:.0%} trades={len(trades)} {action}")
    else:
        print(f"  {sym}: trades={len(trades)} (insufficient)")
db.close()

# Check 4: WALLET & DRAWDOWN
print("\n=== CHECK 4: WALLET & DRAWDOWN ===")
db = sqlite3.connect("data/live_trading.db")
cap = db.execute("SELECT capital, peak_capital FROM live_capital ORDER BY id DESC LIMIT 1").fetchone()
if cap:
    dd = (cap[1] - cap[0]) / cap[1] * 100 if cap[1] > 0 else 0
    print(f"Capital: ${cap[0]:.2f} | Peak: ${cap[1]:.2f} | DD: {dd:.2f}%")

today_ms = int(__import__('time').time() * 1000) - 24*3600*1000
today = db.execute("SELECT COUNT(*), SUM(pnl_net), SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) FROM live_trades WHERE entry_time > ?", (today_ms,)).fetchone()
if today and today[0] > 0:
    wr = today[2]/today[0]*100 if today[0] > 0 else 0
    print(f"Today: {today[0]} trades | WR={wr:.0f}% | PnL=${today[1]:+.2f}")
db.close()

# Wallet
try:
    from src.utils.binance_client import BinanceRESTClient
    client = BinanceRESTClient(testnet=True)
    balances = client.get_balance()
    usdt = next((float(b.get('balance',0)) for b in balances if b.get('asset')=='USDT'), None)
    if usdt: print(f"Wallet: ${usdt:.2f}")
    fr = client.get_funding_rate('BTCUSDT', limit=1)
    if isinstance(fr, list) and fr:
        rate = float(fr[0].get('fundingRate', 0)) * 100
        print(f"BTC FR: {rate:.4f}%")
except Exception as e:
    print(f"Wallet check error: {e}")
