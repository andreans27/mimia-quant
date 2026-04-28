#!/usr/bin/env python3
"""Check #2: Run History & Gap Analysis"""
import sqlite3, json
from datetime import datetime
db = sqlite3.connect("data/live_trading.db")
total = db.execute("SELECT COUNT(*) FROM live_runs").fetchone()[0]
print(f"Total runs: {total}")
runs = db.execute("SELECT id, timestamp, duration_ms, signals_generated, trades_opened, trades_closed, capital, drawdown FROM live_runs ORDER BY id DESC LIMIT 10").fetchall()
for r in reversed(runs):
    ts = datetime.fromtimestamp(r[1]/1000).strftime('%H:%M')
    print(f"  #{r[0]} @ {ts} ({r[2]}ms) sig={r[3]} open={r[4]} close={r[5]} cap=${r[6]:.2f} DD={r[7]:.2f}%")
# Gap analysis
all_runs = db.execute("SELECT id, timestamp FROM live_runs ORDER BY id").fetchall()
prev_ts = None
gaps = []
for rid, rts in all_runs:
    cur_ts = datetime.fromtimestamp(rts/1000)
    if prev_ts:
        gap_min = (cur_ts - prev_ts).total_seconds() / 60
        if gap_min > 10:
            gaps.append((rid, gap_min))
            print(f"  GAP #{rid}: {gap_min:.0f} min")
    prev_ts = cur_ts
print(f"Total gaps >10min: {len(gaps)}")
# Last run time
if all_runs:
    last_ts = datetime.fromtimestamp(all_runs[-1][1]/1000)
    minutes_ago = (datetime.now() - last_ts).total_seconds() / 60
    print(f"Last run: {minutes_ago:.0f} min ago")
db.close()
