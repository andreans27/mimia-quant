#!/usr/bin/env python3
"""
Mimia Quant - Live Trading State & Database Functions
======================================================
Database initialization, state management, capital tracking, and logging
for the live trading system.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional

from dotenv import load_dotenv
load_dotenv()  # Load .env for Binance API keys

# ─── Constants ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")
DB_PATH = Path("data/live_trading.db")

# Top 10 pairs sorted by win rate from full backtest
LIVE_SYMBOLS = [
    "APTUSDT", "UNIUSDT", "FETUSDT", "TIAUSDT", "SOLUSDT",
    "OPUSDT", "1000PEPEUSDT", "SUIUSDT", "ARBUSDT", "INJUSDT",
]

# Trading parameters (from optimal backtest sweep)
THRESHOLD = 0.60
HOLD_BARS = 9       # Hold position for ~45 min (9 × 5m)
COOLDOWN_BARS = 3   # Wait 15 min between trades
POSITION_PCT = 0.15 # Risk 15% of capital per trade (deprecated — use MARGIN_PCT * LEVERAGE)
MARGIN_PCT = 0.01   # 1% of total balance per position
LEVERAGE_X = 10     # 10x leverage
INITIAL_CAPITAL = 5000.0
TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE = 0.0005   # 0.05%

TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']
SEEDS = [42, 101, 202, 303, 404]
WARMUP_BARS = 200
FETCH_DAYS = 130

# Telegram reporting
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    print("⚠️  WARNING: TELEGRAM_BOT_TOKEN not set — Telegram reporting disabled")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "766684679")


# ─── Database ───────────────────────────────────────────────────────────────
def init_db():
    """Initialize the live trading database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS live_state (
            symbol TEXT PRIMARY KEY,
            position INTEGER DEFAULT 0,
            entry_price REAL,
            entry_time INTEGER,
            entry_proba REAL,
            hold_remaining INTEGER DEFAULT 0,
            cooldown_remaining INTEGER DEFAULT 0,
            qty REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS live_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_time INTEGER NOT NULL,
            exit_time INTEGER,
            entry_price REAL NOT NULL,
            exit_price REAL,
            qty REAL NOT NULL,
            pnl_net REAL,
            pnl_pct REAL,
            entry_proba REAL,
            hold_bars INTEGER,
            exit_reason TEXT DEFAULT 'hold_expiry'
        );

        CREATE TABLE IF NOT EXISTS live_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            proba REAL NOT NULL,
            signal INTEGER NOT NULL,
            capital REAL
        );

        CREATE TABLE IF NOT EXISTS live_capital (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            capital REAL NOT NULL,
            peak_capital REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS live_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            duration_ms INTEGER,
            signals_generated INTEGER,
            trades_opened INTEGER,
            trades_closed INTEGER,
            capital REAL,
            peak REAL,
            drawdown REAL
        );
    """)

    # Initialize state for all symbols if not exists
    for sym in LIVE_SYMBOLS:
        c.execute(
            "INSERT OR IGNORE INTO live_state (symbol) VALUES (?)",
            (sym,)
        )

    # Initialize capital tracker
    c.execute("SELECT COUNT(*) FROM live_capital")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO live_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
            (int(time.time() * 1000), INITIAL_CAPITAL, INITIAL_CAPITAL)
        )

    conn.commit()
    conn.close()


def get_state(conn) -> Dict[str, Dict]:
    """Load current state for all symbols."""
    c = conn.cursor()
    c.execute("SELECT * FROM live_state")
    state = {}
    for row in c.fetchall():
        state[row[0]] = {
            'position': row[1],
            'entry_price': row[2],
            'entry_time': row[3],
            'entry_proba': row[4],
            'hold_remaining': row[5],
            'cooldown_remaining': row[6],
            'qty': row[7],
        }
    return state


def save_state(conn, states: Dict[str, Dict]):
    """Save current state for all symbols."""
    c = conn.cursor()
    for sym, s in states.items():
        c.execute("""
            UPDATE live_state SET
                position=?, entry_price=?, entry_time=?, entry_proba=?,
                hold_remaining=?, cooldown_remaining=?, qty=?
            WHERE symbol=?
        """, (
            s['position'], s['entry_price'], s['entry_time'], s['entry_proba'],
            s['hold_remaining'], s['cooldown_remaining'], s['qty'], sym
        ))
    conn.commit()


def get_capital(conn) -> Tuple[float, float]:
    """Get current capital and peak capital."""
    c = conn.cursor()
    c.execute("SELECT capital FROM live_capital ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    if not row:
        return INITIAL_CAPITAL, INITIAL_CAPITAL
    c.execute("SELECT MAX(peak_capital) FROM live_capital")
    peak = c.fetchone()[0] or INITIAL_CAPITAL
    return row[0], peak


def update_capital(conn, capital: float, peak: float):
    """Update capital tracker."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO live_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
        (int(time.time() * 1000), capital, peak)
    )
    conn.commit()


def log_signal(conn, symbol: str, timestamp: int, proba: float, signal: int, capital: float):
    """Log a signal event."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO live_signals (symbol, timestamp, proba, signal, capital) VALUES (?, ?, ?, ?, ?)",
        (symbol, timestamp, proba, signal, capital)
    )
    conn.commit()


def log_trade(conn, trade: Dict):
    """Log a completed trade."""
    c = conn.cursor()
    c.execute("""
        INSERT INTO live_trades
            (symbol, direction, entry_time, exit_time, entry_price, exit_price,
             qty, pnl_net, pnl_pct, entry_proba, hold_bars, exit_reason)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade['symbol'], trade['direction'], trade['entry_time'],
        trade['exit_time'], trade['entry_price'], trade['exit_price'],
        trade['qty'], trade['pnl_net'], trade['pnl_pct'],
        trade['entry_proba'], trade['hold_bars'], trade.get('exit_reason', 'hold_expiry')
    ))
    conn.commit()


def log_run(conn, duration_ms: int, n_signals: int, n_opened: int, n_closed: int,
            capital: float, peak: float, dd: float):
    """Log a run cycle."""
    c = conn.cursor()
    c.execute("""
        INSERT INTO live_runs
            (timestamp, duration_ms, signals_generated, trades_opened,
             trades_closed, capital, peak, drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(time.time() * 1000), duration_ms, n_signals, n_opened,
          n_closed, capital, peak, dd))
    conn.commit()


def reset_state():
    """Reset all live trading state (positions, but keep trade history)."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("UPDATE live_state SET position=0, entry_price=0, entry_time=0, "
              "entry_proba=0, hold_remaining=0, cooldown_remaining=0, qty=0")
    c.execute("DELETE FROM live_signals")
    c.execute("DELETE FROM live_capital")
    c.execute("INSERT INTO live_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
              (int(time.time() * 1000), INITIAL_CAPITAL, INITIAL_CAPITAL))
    conn.commit()
    conn.close()
    print("✅ State reset. All positions set to flat, capital reset to $5000.")
