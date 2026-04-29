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

# All 20 pairs fully retrained & deployed with WR > 70%
# (replaced legacy pairs APT-60.3%, UNI-54.8% → all 20 retrained)
LIVE_SYMBOLS = [
    # WR >= 85%
    "ENAUSDT",     # WR=87.4% PF=26.96 DD=0.28%
    "SUIUSDT",     # WR=86.9% PF=30.33 DD=0.08%
    "OPUSDT",      # WR=86.1% PF=20.74 DD=0.33%
    "FETUSDT",     # WR=86.0% PF=21.70 DD=0.32%
    "TIAUSDT",     # WR=86.6% PF=25.90 DD=0.13%
    "WIFUSDT",     # WR=83.6% PF=16.56 DD=0.18%
    "DOGEUSDT",    # WR=80.6% PF=13.72 DD=0.41%
    "SOLUSDT",     # WR=76.9% PF=12.10 DD=0.14%
    "SEIUSDT",     # WR=81.8% PF=14.00 DD=0.17%
    "1000PEPEUSDT",# WR=88.7% PF=22.05 DD=0.25%
    "ARBUSDT",     # WR=86.4% PF=27.55 DD=0.14%
    "INJUSDT",     # WR=86.7% PF=26.52 DD=0.17%
    "AVAXUSDT",    # WR=82.9% PF=18.75 DD=0.11%
    "BNBUSDT",     # WR=70.2% PF=8.13 DD=0.09%
    "ETHUSDT",     # WR=72.7% PF=11.24 DD=0.16%
    "LINKUSDT",    # WR=80.8% PF=13.55 DD=0.19%
    "NEARUSDT",    # WR=87.4% PF=27.35 DD=0.15%
    "ADAUSDT",     # WR=87.0% PF=28.89 DD=0.10%
    "AAVEUSDT",    # WR=81.5% PF=16.19 DD=0.22%
    "WLDUSDT",     # WR=84.2% PF=17.02 DD=0.39%
]

# Trading parameters (from optimal parameter sweep v4 — Apr 29)
THRESHOLD = 0.50      # Optimal: lower = more trades, still 91% WR
HOLD_BARS = 10         # Optimal: 10 bars (50 min) gives highest PnL
COOLDOWN_BARS = 3      # Wait 15 min between trades
POSITION_PCT = 0.15    # Risk 15% of capital per trade (baseline)
MARGIN_PCT = 0.01      # 1% of total balance per position
LEVERAGE_X = 10        # 10x leverage

# ─── Enhanced Parameters ───────────────────────────────────────────

# Per-symbol thresholds: lower for low-freq symbols to increase trade count
# Optimal from sweep v4: global THRESHOLD=0.50 is optimal for ALL symbols
# Per-symbol adjustments not needed — empty dict, all use THRESHOLD
SYMBOL_THRESHOLDS = {}

# Per-symbol hold bars based on volatility regime
# Optimal from sweep v4: high-vol = 9 (vs global 10), produces marginal +$23 improvement
# Default HOLD_BARS (10) for all others
HOLD_BARS_PER_SYMBOL = {
    'WIFUSDT': 9,
    'DOGEUSDT': 9,
    '1000PEPEUSDT': 9,
    'INJUSDT': 9,
}

# Dynamic position sizing: proba → position_pct fraction of capital
# Optimal from param sweep v4 (aggressive variant = best +44.5% over fixed 15%):
#   proba 0.50-0.64 → 0.15  (same as baseline — baseline 15% is already optimal)
#   proba 0.65-0.69 → 0.18  (moderate confidence → slight increase)
#   proba 0.70-0.74 → 0.22  (high confidence → larger position)
#   proba 0.75-0.79 → 0.28  (very high confidence → much larger)
#   proba ≥ 0.80    → 0.35  (extremely high confidence → max deployment)
SIZE_BY_PROBA = {
    0.65: 0.18,
    0.70: 0.22,
    0.75: 0.28,
    0.80: 0.35,
}

def get_symbol_threshold(symbol: str) -> float:
    """Get per-symbol threshold, fallback to global THRESHOLD."""
    return SYMBOL_THRESHOLDS.get(symbol, THRESHOLD)

def get_symbol_hold_bars(symbol: str) -> int:
    """Get per-symbol hold bars, fallback to global HOLD_BARS."""
    return HOLD_BARS_PER_SYMBOL.get(symbol, HOLD_BARS)

def get_dynamic_position_pct(proba: float, symbol: str = '') -> float:
    """Get position size based on entry probability.
    
    Uses SIZE_BY_PROBA mapping. Falls back to POSITION_PCT if
    proba is below threshold or not in map.
    """
    if proba < THRESHOLD:
        return POSITION_PCT
    # Find the closest proba bucket (floor)
    buckets = sorted(SIZE_BY_PROBA.keys())
    for b in reversed(buckets):
        if proba >= b:
            return SIZE_BY_PROBA[b]
    return POSITION_PCT
INITIAL_CAPITAL = 5000.0
TAKER_FEE = 0.0004  # 0.04%
SLIPPAGE = 0.0005   # 0.05%

TF_GROUPS = ['long', 'short']  # Dual models: long (predict UP) + short (predict DOWN)
SEEDS = [42, 101, 202, 303, 404]
WARMUP_BARS = 200
FETCH_DAYS = 130

# Telegram reporting
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not TELEGRAM_BOT_TOKEN:
    print("WARNING: TELEGRAM_BOT_TOKEN not set — Telegram reporting disabled")
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
            capital REAL,
            data_cutoff INTEGER,
            model_info TEXT
        );

        CREATE TABLE IF NOT EXISTS live_capital (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            capital REAL NOT NULL,
            peak_capital REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pending_signals (
            symbol TEXT PRIMARY KEY,
            signal INTEGER NOT NULL DEFAULT 0,
            proba REAL NOT NULL DEFAULT 0.0,
            timestamp INTEGER NOT NULL DEFAULT 0,
            bar_index TEXT
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

    # Migration: add columns if table already exists (duck-typed ALTER TABLE)
    for table_col in [
        ('live_signals', 'data_cutoff', 'INTEGER'),
        ('live_signals', 'model_info', 'TEXT'),
        ('live_trades', 'data_cutoff', 'INTEGER'),
    ]:
        try:
            c.execute(f"ALTER TABLE {table_col[0]} ADD COLUMN {table_col[1]} {table_col[2]}")
        except:
            pass
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


def log_signal(conn, symbol: str, timestamp: int, proba: float, signal: int, capital: float,
               data_cutoff: int = 0, model_info: str = ""):
    """Log a signal event with snapshot metadata."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO live_signals (symbol, timestamp, proba, signal, capital, data_cutoff, model_info) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (symbol, timestamp, proba, signal, capital, data_cutoff, model_info)
    )
    conn.commit()


def get_model_info() -> str:
    """Get JSON string of model file timestamps for current models.
    
    Returns JSON dict: {model_filename: mtime_timestamp, ...}
    Used by backtest replay to detect model drift.
    """
    import json
    from pathlib import Path
    model_dir = Path("data/ml_models")
    info = {}
    if model_dir.exists():
        for f in sorted(model_dir.glob("*_xgb_ens_*.json")):
            if f.is_file():
                info[f.name] = int(f.stat().st_mtime)
    return json.dumps(info)


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
    c.execute("DELETE FROM pending_signals")
    c.execute("DELETE FROM live_capital")
    c.execute("INSERT INTO live_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
              (int(time.time() * 1000), INITIAL_CAPITAL, INITIAL_CAPITAL))
    conn.commit()
    conn.close()
    print("State reset. All positions set to flat, capital reset to $5000.")


# ─── Pending Signals (Deferred Entry — Candle N → Candle N+1) ───────

def save_pending_signals(conn, signals: Dict[str, Dict]):
    """Save pending signals to DB for deferred execution.

    Signals generated at candle N close are stored here and executed
    at candle N+1 close. This ensures backtest <-> live alignment.

    signals dict: {symbol: {'signal': int, 'proba': float, 'timestamp': ms, 'bar_index': str}}
    """
    c = conn.cursor()
    for symbol, sig in signals.items():
        s = sig.get('signal', 0)
        p = sig.get('proba', 0.0)
        ts = sig.get('timestamp', int(time.time() * 1000))
        bi = sig.get('bar_index', '')
        c.execute("""
            INSERT INTO pending_signals (symbol, signal, proba, timestamp, bar_index)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                signal=excluded.signal, proba=excluded.proba,
                timestamp=excluded.timestamp, bar_index=excluded.bar_index
        """, (symbol, s, p, ts, bi))
    conn.commit()


def load_pending_signals(conn) -> Dict[str, Dict]:
    """Load all pending signals from DB.

    Returns dict of {symbol: {'signal': int, 'proba': float, 'timestamp': ms, 'bar_index': str}}
    """
    c = conn.cursor()
    c.execute("SELECT symbol, signal, proba, timestamp, bar_index FROM pending_signals WHERE signal != 0")
    signals = {}
    for row in c.fetchall():
        signals[row[0]] = {
            'signal': row[1],
            'proba': row[2],
            'timestamp': row[3] or 0,
            'bar_index': row[4] or '',
        }
    return signals


def clear_pending_signals(conn):
    """Clear all pending signals after execution."""
    c = conn.cursor()
    c.execute("DELETE FROM pending_signals")
    conn.commit()
