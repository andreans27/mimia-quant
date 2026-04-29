#!/usr/bin/env python3
"""Reset database for fresh start — preserves meta table (db_initialized_at)."""
import sqlite3
import time
from datetime import datetime

DB_PATH = 'data/live_trading.db'


def main():
    ts = int(time.time() * 1000)
    now_str = datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M UTC')

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Create meta if not exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)

    # Store reset timestamp BEFORE clearing trades
    c.execute('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)',
              ('db_initialized_at', str(ts)))

    # Clear all trade tables
    for table in ['live_trades', 'live_signals', 'live_runs', 'pending_signals']:
        c.execute(f'DELETE FROM {table}')

    # Reset capital
    c.execute('DELETE FROM live_capital')
    c.execute('INSERT INTO live_capital (timestamp, capital, peak_capital) VALUES (?, 5000.00, 5000.00)', (ts,))

    # Reset state
    from src.config import LIVE_SYMBOLS
    c.execute('DELETE FROM live_state')
    for sym in LIVE_SYMBOLS:
        c.execute("""
            INSERT INTO live_state (symbol, position, entry_price, entry_time,
                entry_proba, hold_remaining, cooldown_remaining, qty, last_signal)
            VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0)
        """, (sym,))

    conn.commit()
    conn.close()

    print(f'✅ DB Reset — {now_str}')
    print(f'   meta.db_initialized_at = {ts}')
    print(f'   capital = $5000.00')
    print(f'   state = {len(LIVE_SYMBOLS)} symbols')


if __name__ == '__main__':
    main()
