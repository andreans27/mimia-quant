#!/usr/bin/env python3
"""Check #5: Trade Quality Check"""
import sqlite3
db = sqlite3.connect("data/live_trading.db")

# Last 10 trades
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

# Exit reasons
reasons = db.execute("SELECT exit_reason, COUNT(*) FROM live_trades GROUP BY exit_reason ORDER BY COUNT(*) DESC").fetchall()
if reasons:
    print(f"Exit reasons: {dict(reasons)}")
else:
    print("No trades yet")

# Open positions
open_pos = db.execute("SELECT symbol, position, qty, hold_remaining FROM live_state WHERE position != 0").fetchall()
if open_pos:
    for p in open_pos:
        d = 'LNG' if p[1]==1 else 'SHT'
        print(f"  Open: {d} {p[0]} qty={p[2]} hold={p[3]}")
else:
    print("Open: none")

# Equity curve (last 48 points)
capital_hist = db.execute("SELECT timestamp, capital FROM live_capital ORDER BY id DESC LIMIT 48").fetchall()
if len(capital_hist) >= 2:
    low = min(c[1] for c in capital_hist)
    high = max(c[1] for c in capital_hist)
    change = capital_hist[0][1] - capital_hist[-1][1]
    print(f"12h range: ${low:.2f}->${high:.2f} (change: ${change:+.2f})")

# Summary stats
total_trades = db.execute("SELECT COUNT(*) FROM live_trades").fetchone()[0]
total_pnl = db.execute("SELECT SUM(pnl_net) FROM live_trades").fetchone()[0]
winning = db.execute("SELECT COUNT(*) FROM live_trades WHERE pnl_net > 0").fetchone()[0]
losing = db.execute("SELECT COUNT(*) FROM live_trades WHERE pnl_net < 0").fetchone()[0]
print(f"\nAll-time: {total_trades} trades | PnL=${total_pnl:+.2f} | Wins: {winning} Losses: {losing}")

# Top 5 worst symbols by PnL
worst = db.execute("SELECT symbol, COUNT(*), SUM(pnl_net) FROM live_trades GROUP BY symbol ORDER BY SUM(pnl_net) ASC LIMIT 5").fetchall()
print(f"\nWorst 5 symbols by PnL:")
for sym, n, pnl in worst:
    print(f"  {sym}: {n} trades | PnL=${pnl:+.2f}")

db.close()
