#!/usr/bin/env python3
"""Guardian check: Run History, Gap Analysis, Trade Quality"""
import sqlite3, json
from datetime import datetime

db = sqlite3.connect("/root/projects/mimia-quant/data/live_trading.db")

# Check 2: RUN HISTORY & GAP ANALYSIS
total = db.execute("SELECT COUNT(*) FROM live_runs").fetchone()[0]
print("=== CHECK 2: RUN HISTORY ===")
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
            print(f"  WARNING GAP #{rid}: {gap_min:.0f} min")
    prev_ts = cur_ts
print(f"Total gaps >10min: {len(gaps)}")

# Check 5: TRADE QUALITY CHECK
print("\n=== CHECK 5: TRADE QUALITY ===")
trades = db.execute("""
    SELECT symbol, direction, entry_price, exit_price, pnl_net, pnl_pct, exit_reason
    FROM live_trades ORDER BY entry_time DESC LIMIT 10
""").fetchall()
if trades:
    print(f"Last {len(trades)} trades:")
    for t in trades:
        d = 'LNG' if t[1]==1 else 'SHT'
        pnl = f"${t[4]:+.2f}" if t[4] else '?'
        print(f"  {d} {t[0]}: ${t[2]:.4f}->${t[3]:.4f} ({t[6]}) PnL={pnl}")

reasons = db.execute("SELECT exit_reason, COUNT(*) FROM live_trades GROUP BY exit_reason ORDER BY COUNT(*) DESC").fetchall()
print(f"Exit reasons: {dict(reasons)}" if reasons else "No trades yet")

open_pos = db.execute("SELECT symbol, position, qty, hold_remaining FROM live_state WHERE position != 0").fetchall()
if open_pos:
    for p in open_pos:
        d = 'LNG' if p[1]==1 else 'SHT'
        print(f"  Open: {d} {p[0]} qty={p[2]} hold={p[3]}")
else:
    print("Open positions: none")

capital_hist = db.execute("SELECT timestamp, capital FROM live_capital ORDER BY id DESC LIMIT 48").fetchall()
if len(capital_hist) >= 2:
    low = min(c[1] for c in capital_hist)
    high = max(c[1] for c in capital_hist)
    change = capital_hist[0][1] - capital_hist[-1][1]
    print(f"12h range: ${low:.2f}->${high:.2f} (change: ${change:+.2f})")

db.close()
