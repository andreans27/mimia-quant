#!/usr/bin/env python3
"""Check #3: Signal Quality & Retrain Decisions"""
import json, sqlite3
from datetime import datetime, timezone

# Retrain status
try:
    with open("data/ml_models/_retrain_status.json") as f:
        status = json.load(f)
except:
    status = {'symbols': {}, 'runs': []}

# Deployed time
deployed_at = status.get('deployed_at', '')
if deployed_at:
    dt = datetime.fromisoformat(deployed_at.replace('Z','+00:00'))
    hours_ago = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    print(f"Deployed: {hours_ago:.0f}h ago")

# Symbol list (from status)
symbols = sorted(status.get('symbols', {}).keys())
if not symbols:
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']

# Retrain history
runs = status.get('runs', [])
print(f"Total retrain runs: {len(runs)}")
for r in runs[-5:]:
    dt = r.get('time','')
    sym = r.get('symbol','')
    wr = r.get('metrics',{}).get('wr','?')
    pf = r.get('metrics',{}).get('pf','?')
    print(f"  Retrain {sym}: WR={wr}% PF={pf} [{dt}]")

# Live WR dari DB
db = sqlite3.connect("data/live_trading.db")
print(f"\nSignal Quality per Symbol:")
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
        print(f"  {sym}: trades={len(trades)} (insufficient data)")
db.close()
