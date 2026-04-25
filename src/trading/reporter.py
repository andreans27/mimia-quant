#!/usr/bin/env python3
"""
Mimia Quant - Live Trading Reporter
=====================================
Status display and Telegram daily report generation for live trading.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3
from datetime import datetime, timedelta

from src.trading.state import (
    init_db, get_capital,
    DB_PATH, INITIAL_CAPITAL,
)
from src.trading.engine import LiveTrader


def show_status():
    """Show current live trading status."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))

    capital, peak_capital = get_capital(conn)
    pnl = capital - INITIAL_CAPITAL
    pnl_pct = pnl / INITIAL_CAPITAL * 100
    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

    print(f"\n📊 Live Trading Status")
    print(f"{'='*50}")
    print(f"  Capital: ${capital:.2f}")
    print(f"  Peak:    ${peak_capital:.2f}")
    print(f"  PnL:     ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    print(f"  DD:      {dd:.2f}%")

    # Trades summary
    c = conn.cursor()
    c.execute("SELECT COUNT(*), SUM(pnl_net) FROM live_trades")
    cnt, total_pnl = c.fetchone()
    total_pnl = total_pnl or 0
    print(f"\n  Total Trades: {cnt}")
    print(f"  Cumulative PnL: ${total_pnl:.2f}")

    if cnt > 0:
        c.execute("SELECT COUNT(*), SUM(pnl_net) FROM live_trades WHERE pnl_net > 0")
        wins, win_pnl = c.fetchone()
        win_pnl = win_pnl or 0
        win_rate = wins / cnt * 100
        print(f"  Wins: {wins}/{cnt} ({win_rate:.1f}%)")

        c.execute("SELECT SUM(pnl_net) FROM live_trades WHERE pnl_net < 0")
        loss_pnl = c.fetchone()[0] or 0
        pf = win_pnl / abs(loss_pnl) if loss_pnl != 0 else float('inf')
        print(f"  Profit Factor: {pf:.2f}")

    # Open positions
    print(f"\n  Open Positions:")
    c.execute("SELECT symbol, position, entry_price, hold_remaining FROM live_state WHERE position != 0")
    positions = c.fetchall()
    if positions:
        for sym, pos, price, hold in positions:
            dir_str = "🟢 LONG" if pos == 1 else "🔴 SHORT"
            print(f"    {dir_str} {sym} @ ${price:.2f} | hold_remaining={hold}")
    else:
        print(f"    None (all flat)")

    # Last runs
    print(f"\n  Last 5 Runs:")
    c.execute("SELECT timestamp, signals_generated, trades_opened, trades_closed FROM live_runs ORDER BY id DESC LIMIT 5")
    for ts, sigs, opened, closed in c.fetchall():
        t = datetime.fromtimestamp(ts / 1000).strftime('%m-%d %H:%M')
        print(f"    {t}: {sigs} signals, {opened} opened, {closed} closed")

    conn.close()


def show_report(testnet: bool = True):
    """Generate and send a full daily report."""
    init_db()
    pt = LiveTrader(report=True, testnet=testnet)

    capital, peak_capital = get_capital(pt.conn)
    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

    # Get today's trades
    today_start = int((datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=1)).timestamp() * 1000)
    c = pt.conn.cursor()
    c.execute("""
        SELECT symbol, direction, entry_time, exit_time, entry_price, exit_price,
               pnl_net, pnl_pct, entry_proba
        FROM live_trades
        WHERE entry_time >= ?
        ORDER BY entry_time DESC
    """, (today_start,))
    trades = c.fetchall()

    trade_lines = []
    for t in trades:
        sym, direction, entry_ts, exit_ts, entry_p, exit_p, pnl_net, pnl_pct, proba = t
        pnl_icon = "🟢" if pnl_net > 0 else "🔴"
        trade_lines.append(
            f"{pnl_icon} {sym}: {direction.upper()} @ ${entry_p:.2f}→${exit_p:.2f} "
            f"| PnL=${pnl_net:.2f}({pnl_pct:+.2f}%) | proba={proba:.3f}"
        )

    # Today's PnL
    today_pnl = sum(t[6] for t in trades) if trades else 0

    # Weekly stats
    week_start = int((datetime.utcnow() - timedelta(days=7)).timestamp() * 1000)
    c.execute("SELECT COUNT(*), SUM(pnl_net) FROM live_trades WHERE entry_time >= ?", (week_start,))
    week_cnt, week_pnl = c.fetchone()
    week_pnl = week_pnl or 0
    c.execute("SELECT COUNT(*) FROM live_trades WHERE entry_time >= ? AND pnl_net > 0", (week_start,))
    week_wins = c.fetchone()[0] or 0
    week_wr = week_wins / week_cnt * 100 if week_cnt > 0 else 0

    # Open positions
    c.execute("SELECT symbol, position, entry_price FROM live_state WHERE position != 0")
    open_pos = c.fetchall()

    msg = (
        f"📈 *Mimia Daily Report*\n"
        f"`{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}`\n\n"
        f"*Portfolio*\n"
        f"  Capital: `${capital:.2f}` | PnL: `${today_pnl:.2f}` (today)\n"
        f"  Week: `${week_pnl:.2f}` ({week_wr:.1f}% WR, {week_cnt} trades)\n"
        f"  Peak: `${peak_capital:.2f}` | DD: `{dd:.2f}%`\n\n"
    )

    if trade_lines:
        msg += f"*Today's Trades ({len(trade_lines)})*\n"
        for line in trade_lines[-10:]:
            msg += line + "\n"
        msg += "\n"

    if open_pos:
        msg += f"*Open Positions*\n"
        for sym, pos, price in open_pos:
            msg += f"  {'🟢' if pos==1 else '🔴'} {sym} @ ${price:.2f}\n"
        msg += "\n"

    # Best/worst symbol
    c.execute("""
        SELECT symbol, COUNT(*) as cnt, SUM(pnl_net) as pnl
        FROM live_trades GROUP BY symbol ORDER BY pnl DESC
    """)
    symbol_pnl = c.fetchall()
    if symbol_pnl:
        msg += "*Performance by Symbol*\n"
        for sym, cnt, pnl in symbol_pnl:
            icon = "🟢" if (pnl or 0) > 0 else "🔴"
            msg += f"  {icon} {sym}: ${pnl:.2f} ({cnt} trades)\n"

    pt.send_telegram(msg)
    pt.close()
