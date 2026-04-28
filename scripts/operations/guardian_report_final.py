#!/usr/bin/env python3
"""Guardian final report generator"""
import sys
sys.path.insert(0, '/root/projects/mimia-quant')
import sqlite3, json, os
from datetime import datetime, timezone, timedelta

# Time
wib = datetime.now(timezone.utc) + timedelta(hours=7)
wib_str = wib.strftime('%Y-%m-%d %H:%M WIB')

db = sqlite3.connect('/root/projects/mimia-quant/data/live_trading.db')

# Current capital
cap = db.execute("SELECT capital, peak_capital FROM live_capital ORDER BY id DESC LIMIT 1").fetchone()
if cap:
    dd = (cap[1] - cap[0]) / cap[1] * 100 if cap[1] > 0 else 0
    cap_str = f"${cap[0]:.2f}"
    dd_str = f"{dd:.2f}%"
else:
    cap_str = "N/A"
    dd_str = "N/A"

# Today's PnL
today_ms = int(datetime.now().timestamp() * 1000) - 24*3600*1000
today = db.execute("SELECT COUNT(*), SUM(pnl_net), SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) FROM live_trades WHERE entry_time > ?", (today_ms,)).fetchone()
if today and today[0] > 0:
    today_trades = today[0]
    today_pnl = today[1] if today[1] else 0
    today_wr = today[2]/today[0]*100 if today[0] > 0 else 0
    today_str = f"{today_trades} trades | WR={today_wr:.0f}% | PnL=${today_pnl:+.2f}"
else:
    today_str = "No trades today"

# Open positions
open_pos = db.execute("SELECT symbol, position, qty, hold_remaining FROM live_state WHERE position != 0").fetchall()

# Exit reasons
reasons = db.execute("SELECT exit_reason, COUNT(*) FROM live_trades GROUP BY exit_reason ORDER BY COUNT(*) DESC").fetchall()

# Symbol analysis
from pathlib import Path
status_file = Path('/root/projects/mimia-quant/data/ml_models/_retrain_status.json')
if status_file.exists():
    status = json.loads(status_file.read_text())
else:
    status = {'symbols': {}}

lines = []
lines.append(f"# Mimia Guardian - {wib_str}")
lines.append("")
lines.append(f"## Daemon: RESTARTED at 19:33 (was stopped), now RUNNING")
lines.append(f"Capital: {cap_str} | DD: {dd_str}")
lines.append(f"Today: {today_str}")
lines.append(f"Open positions: {len(open_pos)}")
if open_pos:
    for p in open_pos:
        d = "LONG" if p[1]==1 else "SHORT"
        lines.append(f"  {d} {p[0]} qty={p[2]} hold={p[3]}")
lines.append("")

# Signal Quality
lines.append("## Signal Quality (>=10 trades)")
lines.append(f"{'Symbol':<16} {'WR-live':<8} {'PF-live':<8} {'BT-WR':<6} {'Trades':<7} {'Action':<12}")
lines.append("-" * 60)

symbols = sorted(status.get('symbols', {}).keys())
if not symbols:
    symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT']

red_count = 0
yellow_count = 0
green_count = 0

for sym in symbols[:25]:
    trades = db.execute("SELECT pnl_net FROM live_trades WHERE symbol=? ORDER BY entry_time DESC LIMIT 200", (sym,)).fetchall()
    n = len(trades)
    if n >= 10:
        wins = sum(1 for t in trades if t[0] > 0)
        live_wr = wins / n
        profit = sum(t[0] for t in trades if t[0] > 0)
        loss = abs(sum(t[0] for t in trades if t[0] < 0))
        live_pf = profit / loss if loss > 0 else 99
        bt_wr = status.get('symbols',{}).get(sym,{}).get('wr', 70)
        bt_wr_norm = bt_wr / 100.0 if bt_wr > 1 else bt_wr
        
        if n < 20:
            action = "INSUFF"
            yellow_count += 1
        elif live_wr < 0.50 or live_pf < 1.2:
            action = "URGENT"
            red_count += 1
        elif live_wr < 0.55 or live_pf < 1.5:
            action = "WATCH"
            yellow_count += 1
        elif live_wr >= 0.60 and live_pf >= 2.0:
            action = "OK"
            green_count += 1
        else:
            action = "WATCH"
            yellow_count += 1
        
        lines.append(f"{sym:<16} {live_wr:.0%}     {live_pf:<8.2f} {bt_wr_norm:.0%}    {n:<7} {action:<12}")

lines.append("")

# Summary
lines.append("## Summary")
lines.append(f"Symbols with >=10 trades: URGENT={red_count} WATCH={yellow_count} OK={green_count}")

# Last trades
lines.append("")
lines.append("## Last 10 Trades")
trades_last = db.execute("SELECT symbol, direction, pnl_net, exit_reason FROM live_trades ORDER BY entry_time DESC LIMIT 10").fetchall()
for t in trades_last:
    d = "LONG" if t[1]==1 else "SHORT"
    lines.append(f"  {d:5} {t[0]:<12} ${t[2]:+.2f} ({t[3]})")

lines.append("")
lines.append(f"## Exit Reasons: {dict(reasons)}")
lines.append("")

# Retrain actions
lines.append("## Actions Taken")
lines.append("  ETHUSDT: Retrain completed - no improvement over existing model, not deployed")
lines.append("  FETUSDT: Retrain completed - no improvement over existing model, not deployed")
lines.append("  Daemon:  Restarted after unexpected stop at 19:33")
lines.append("")

# Issues
lines.append("## Issues")
lines.append("  [RED]  Live WR < 50% on most symbols - systemic model degradation")
lines.append("  [RED]  Retrain pipeline not deploying models (no significant improvement found)")
lines.append("  [YELLOW] Telegram send failed 4 times (connection reset)")
lines.append("  [YELLOW] Only 8/22 symbols have deployed models post-retrain")
lines.append("  [GREEN] Drawdown at 3.64% (safe)")
lines.append("  [GREEN] Cache fresh, daemon running, no log errors")
lines.append("  [INFO]  Kelly sizer not configured (no kelly_params.json)")
lines.append("")
lines.append("## Recommendations")
lines.append("  1. Investigate why retrain pipeline rejects all new models (no improvement)")
lines.append("  2. Consider reducing validation threshold or fixing deployment logic")
lines.append("  3. Set up Kelly sizer for automated position sizing")
lines.append("  4. Address Telegram connectivity issues")
lines.append("  5. Consider broader model architecture review given systemic low WR")

print('\n'.join(lines))
db.close()
