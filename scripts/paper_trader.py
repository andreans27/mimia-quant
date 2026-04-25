#!/usr/bin/env python3
"""
Mimia Quant - Paper Trading Engine
====================================
Live paper trading on Binance Futures testnet using the multi-TF XGBoost ensemble.
Simulates trading with 10 pairs, logs everything to SQLite, reports via Telegram.

Usage:
    python scripts/paper_trader.py                      # Run once (for cron, every 5min)
    python scripts/paper_trader.py --init               # Initialize DB + state
    python scripts/paper_trader.py --status             # Show current state
    python scripts/paper_trader.py --report             # Generate & send daily report
"""

import sys
sys.path.insert(0, ".")

import os
from dotenv import load_dotenv
load_dotenv()  # Load .env for Binance API keys

import json
import time
import sqlite3
import argparse
import warnings
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

warnings.filterwarnings('ignore')

# ─── Constants ──────────────────────────────────────────────────────────────
CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")
DB_PATH = Path("data/paper_trading.db")

# Top 10 pairs sorted by win rate from full backtest
PAPER_SYMBOLS = [
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
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "766684679")


# ─── Database ───────────────────────────────────────────────────────────────
def init_db():
    """Initialize the paper trading database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS paper_state (
            symbol TEXT PRIMARY KEY,
            position INTEGER DEFAULT 0,
            entry_price REAL,
            entry_time INTEGER,
            entry_proba REAL,
            hold_remaining INTEGER DEFAULT 0,
            cooldown_remaining INTEGER DEFAULT 0,
            qty REAL DEFAULT 0.0
        );

        CREATE TABLE IF NOT EXISTS paper_trades (
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

        CREATE TABLE IF NOT EXISTS paper_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timestamp INTEGER NOT NULL,
            proba REAL NOT NULL,
            signal INTEGER NOT NULL,
            capital REAL
        );

        CREATE TABLE IF NOT EXISTS paper_capital (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            capital REAL NOT NULL,
            peak_capital REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS paper_runs (
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
    for sym in PAPER_SYMBOLS:
        c.execute(
            "INSERT OR IGNORE INTO paper_state (symbol) VALUES (?)",
            (sym,)
        )

    # Initialize capital tracker
    c.execute("SELECT COUNT(*) FROM paper_capital")
    if c.fetchone()[0] == 0:
        c.execute(
            "INSERT INTO paper_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
            (int(time.time() * 1000), INITIAL_CAPITAL, INITIAL_CAPITAL)
        )

    conn.commit()
    conn.close()


def get_state(conn) -> Dict[str, Dict]:
    """Load current state for all symbols."""
    c = conn.cursor()
    c.execute("SELECT * FROM paper_state")
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
            UPDATE paper_state SET
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
    c.execute("SELECT capital FROM paper_capital ORDER BY id DESC LIMIT 1")
    row = c.fetchone()
    if not row:
        return INITIAL_CAPITAL, INITIAL_CAPITAL
    c.execute("SELECT MAX(peak_capital) FROM paper_capital")
    peak = c.fetchone()[0] or INITIAL_CAPITAL
    return row[0], peak


def update_capital(conn, capital: float, peak: float):
    """Update capital tracker."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO paper_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
        (int(time.time() * 1000), capital, peak)
    )
    conn.commit()


def log_signal(conn, symbol: str, timestamp: int, proba: float, signal: int, capital: float):
    """Log a signal event."""
    c = conn.cursor()
    c.execute(
        "INSERT INTO paper_signals (symbol, timestamp, proba, signal, capital) VALUES (?, ?, ?, ?, ?)",
        (symbol, timestamp, proba, signal, capital)
    )
    conn.commit()


def log_trade(conn, trade: Dict):
    """Log a completed trade."""
    c = conn.cursor()
    c.execute("""
        INSERT INTO paper_trades
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
        INSERT INTO paper_runs
            (timestamp, duration_ms, signals_generated, trades_opened,
             trades_closed, capital, peak, drawdown)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(time.time() * 1000), duration_ms, n_signals, n_opened,
          n_closed, capital, peak, dd))
    conn.commit()


# ─── Signal Generation ──────────────────────────────────────────────────────
class SignalGenerator:
    """Generates trading signals using the multi-TF XGBoost ensemble."""

    def __init__(self):
        self._cache: Dict[str, Any] = {}  # symbol -> loaded data

    def _fetch_5m_ohlcv(self, symbol: str, days: int = FETCH_DAYS) -> Optional[pd.DataFrame]:
        """Fetch 5m OHLCV from Binance public API (faster than testnet)."""
        end = datetime.now()
        start = end - timedelta(days=days)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        limit = 1000
        all_bars = []
        last_ts = start_ms
        while last_ts < end_ms:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '5m',
                'limit': limit,
                'startTime': last_ts,
                'endTime': end_ms,
            }
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code != 200:
                    break
                batch = r.json()
                if not batch:
                    break
                all_bars.extend(batch)
                last_ts = batch[-1][0] + 1
                if len(batch) < limit:
                    break
            except Exception:
                break

        if len(all_bars) < 1000:
            return None

        df = pd.DataFrame(all_bars, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

    def _load_models(self, symbol: str):
        """Load all models for a symbol. Cache models permanently; compute fresh features from live data."""
        if symbol in self._cache and 'groups' in self._cache[symbol]:
            # Models are cached — but features are stale (parquet), so we always recompute
            cached = self._cache[symbol]
            # Recompute fresh features
            fresh_features = self._compute_fresh_features(symbol)
            if fresh_features is not None:
                cached['features'] = fresh_features
                return cached
            # Fallback: try stale cache if fresh fails
            if 'features' in cached:
                print(f"    ⚠️ Fresh features failed, using stale for {symbol}")
                return cached
            return None

        # First load: compute fresh features + load models
        group_models = {}
        for tf in TF_GROUPS:
            models = self._load_tf_group(symbol, tf)
            if models:
                group_models[tf] = models

        if len(group_models) < 2:
            return None

        fresh_features = self._compute_fresh_features(symbol)
        if fresh_features is None:
            return None

        result = {
            'features': fresh_features,
            'groups': group_models,
        }
        self._cache[symbol] = result
        return result

    def _compute_fresh_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch live 5m data and compute all features for inference."""
        from src.strategies.ml_features import compute_5m_features_5tf

        # Map 1000x symbols to spot symbols for OHLCV (Binance Spot API)
        spot_symbol = symbol
        if symbol.startswith("1000"):
            for prefix in ["1000", "10000", "100000"]:
                if symbol.startswith(prefix):
                    spot_symbol = symbol[len(prefix):]
                    break

        from datetime import datetime, timedelta

        try:
            print(f"    📡 Fetching live 5m data for {symbol} (Spot: {spot_symbol})...")
            df_5m = self._fetch_5m_ohlcv(spot_symbol, days=5)
            if df_5m is None or len(df_5m) < 500:
                print(f"    ⚠️ Insufficient data for {symbol} (got {len(df_5m) if df_5m is not None else 0} rows)")
                return None

            print(f"    🔧 Computing features...")
            feat_df = compute_5m_features_5tf(df_5m, for_inference=True)

            if len(feat_df) == 0:
                print(f"    ⚠️ No feature rows for {symbol}")
                return None

            # Print latest timestamp for debugging
            latest = feat_df.index[-1]
            print(f"    ✅ {len(feat_df)} feature rows | Latest: {latest} | {len(feat_df.columns)} features")

            return feat_df

        except Exception as e:
            print(f"    ⚠️ Feature computation error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_tf_group(self, symbol: str, tf_group: str) -> Optional[List]:
        """Load models for one TF group."""
        models = []
        if tf_group == 'full':
            meta_path = MODEL_DIR / f"{symbol}_ensemble_meta.json"
            if not meta_path.exists():
                return None
            with open(meta_path) as f:
                meta = json.load(f)
            feature_cols = meta.get('features', meta.get('full_feature_set', []))

            for seed in SEEDS:
                path = MODEL_DIR / f"{symbol}_xgb_ens_{seed}.json"
                if not path.exists():
                    continue
                mf = meta.get('model_features', {}).get(str(seed), meta.get('features', []))
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                models.append((str(seed), m, mf))
        else:
            prefix = f"{tf_group}_"
            for seed in SEEDS:
                path = MODEL_DIR / f"{symbol}_{tf_group}_xgb_ens_{seed}.json"
                if not path.exists():
                    continue
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                model_features = m.get_booster().feature_names
                if model_features and not model_features[0].startswith(prefix):
                    model_features = [
                        f"{prefix}{f}" if not f.startswith(prefix) else f
                        for f in model_features
                    ]
                models.append((str(seed), m, model_features))

        return models if len(models) >= 2 else None

    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate a signal for a symbol using the latest data.
        
        Returns:
            dict with keys: proba, signal (1=long, -1=short, 0=flat), 
            previous_proba, or None if error
        """
        # Handle 1000x symbols -> use spot symbol for OHLCV
        spot_symbol = symbol
        if symbol.startswith("1000"):
            # Map to spot symbol (e.g., 1000PEPEUSDT -> PEPEUSDT)
            for prefix in ["1000", "10000", "100000"]:
                if symbol.startswith(prefix):
                    remainder = symbol[len(prefix):]
                    spot_symbol = remainder
                    break

        try:
            cached = self._load_models(symbol)
            if cached is None:
                return None

            feat_df = cached['features']
            groups = cached['groups']

            # Get latest feature row
            latest_features = feat_df.iloc[-1:]
            if len(latest_features) == 0:
                return None

            # Compute probabilities from all groups
            group_probs = []
            for tf, models in groups.items():
                tf_probs = []
                for seed, m, mf in models:
                    available = [c for c in mf if c in feat_df.columns]
                    if len(available) < 5:
                        continue
                    X = feat_df[available].fillna(0).clip(-10, 10)
                    # Use last row
                    X_row = X.iloc[-1:].fillna(0)
                    probs = m.predict_proba(X_row)[:, 1]
                    tf_probs.append(probs[0])

                if tf_probs:
                    group_probs.append(np.mean(tf_probs))

            if len(group_probs) < 2:
                return None

            proba = float(np.mean(group_probs))

            # Get previous bar proba for cross detection
            prev_proba = None
            if len(feat_df) >= 2:
                prev_feat = feat_df.iloc[-2:-1]
                prev_group_probs = []
                for tf, models in groups.items():
                    tf_probs = []
                    for seed, m, mf in models:
                        available = [c for c in mf if c in feat_df.columns]
                        if len(available) < 5:
                            continue
                        X = feat_df[available].fillna(0).clip(-10, 10)
                        X_prev = X.iloc[-2:-1].fillna(0)
                        if len(X_prev) > 0:
                            probs = m.predict_proba(X_prev)[:, 1]
                            tf_probs.append(probs[0])
                    if tf_probs:
                        prev_group_probs.append(np.mean(tf_probs))
                if len(prev_group_probs) >= 2:
                    prev_proba = float(np.mean(prev_group_probs))

            # Determine signal: level-based (align with backtest)
            signal = 0  # flat
            if proba >= THRESHOLD:
                signal = 1  # LONG
            elif proba <= (1 - THRESHOLD):
                signal = -1  # SHORT

            return {
                'proba': proba,
                'signal': signal,
                'prev_proba': prev_proba,
            }

        except Exception as e:
            print(f"    ⚠️ Signal error for {symbol}: {e}")
            return None


# ─── Paper Trading Engine ───────────────────────────────────────────────────
class PaperTrader:
    """Main paper trading engine."""

    def __init__(self, report: bool = False):
        init_db()
        self.conn = sqlite3.connect(str(DB_PATH))
        self.gen = SignalGenerator()
        self.report_mode = report
        self._client = None  # lazy-init

    def _get_client(self):
        """Lazy-init Binance testnet client."""
        if self._client is None:
            from src.utils.binance_client import BinanceRESTClient
            self._client = BinanceRESTClient(testnet=True)
        return self._client

    def _init_leverage(self, symbol: str) -> bool:
        """Set leverage to 10x on Binance testnet for a symbol. Returns True on success."""
        try:
            self._get_client().change_leverage(symbol, 10)
            return True
        except Exception:
            return False

    def _setup_testnet(self):
        """Reset stale positions in DB and init leverage for all symbols."""
        c = self.conn.cursor()

        # Check if we need initial setup (schema_version flag)
        c.execute("SELECT COUNT(*) FROM paper_state WHERE position != 0")
        stale = c.fetchone()[0]

        if stale > 0:
            print(f"\n  ⚠️ Found {stale} stale position(s) in DB (from old paper-only runs). Resetting...")
            c.execute("UPDATE paper_state SET position=0, entry_price=0, entry_time=0, entry_proba=0, hold_remaining=0, cooldown_remaining=0, qty=0 WHERE position != 0")
            self.conn.commit()
            print(f"  ✅ Stale positions reset. New positions will be placed as real orders on Binance testnet.")

        # Init leverage 10x for all symbols
        ok = 0
        for sym in PAPER_SYMBOLS:
            if self._init_leverage(sym):
                ok += 1
        print(f"  ✅ Leverage 10x set for {ok}/{len(PAPER_SYMBOLS)} symbols")

    def _sync_binance_positions(self):
        """Sync DB state with actual open positions on Binance testnet.
        Prevents duplicate position entries when script restarts."""
        try:
            client = self._get_client()
            positions = client.get_position_info()
            c = self.conn.cursor()
            synced = 0
            for p in positions:
                sym = p.get('symbol', '')
                if sym.upper() not in PAPER_SYMBOLS:
                    continue
                pos_amt = float(p.get('position_amt', 0))
                notional = float(p.get('notional', 0))
                if abs(pos_amt) < 0.001:
                    continue
                direction = 1 if pos_amt > 0 else -1
                qty = abs(pos_amt)
                entry_price = abs(notional / pos_amt) if pos_amt != 0 else 0
                # Update DB state to match Binance
                c.execute("""
                    UPDATE paper_state
                    SET position=?, qty=?, entry_price=?, entry_time=?,
                        hold_remaining=?, cooldown_remaining=0
                    WHERE symbol=?
                """, (direction, qty, entry_price, int(time.time() * 1000), 9, sym))
                synced += 1
            self.conn.commit()
            if synced > 0:
                print(f"  🔄 Synced {synced} existing position(s) from Binance testnet")
        except Exception as e:
            print(f"  ⚠️ Position sync skipped: {e}")

    def close(self):
        self.conn.close()

    def send_telegram(self, message: str):
        """Send a message via Telegram bot."""
        if not TELEGRAM_BOT_TOKEN:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown',
            }, timeout=10)
        except Exception as e:
            print(f"    ⚠️ Telegram send failed: {e}")

    def run(self):
        """Execute one full paper trading cycle."""
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  PAPER TRADE RUN — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        print(f"{'='*60}")

        capital, peak_capital = get_capital(self.conn)

        # Initial setup: reset stale positions (from old paper-only runs) and init leverage
        self._setup_testnet()

        # Sync DB state with actual positions on Binance testnet
        # Prevents duplicate entries if trader restarts mid-position
        self._sync_binance_positions()

        states = get_state(self.conn)
        signals_total = 0
        trades_opened = 0
        trades_closed = 0
        trade_report_lines = []

        for i, symbol in enumerate(PAPER_SYMBOLS):
            print(f"\n  [{i+1}/{len(PAPER_SYMBOLS)}] {symbol}...")
            state = states.get(symbol, {
                'position': 0, 'entry_price': 0, 'entry_time': 0,
                'entry_proba': 0, 'hold_remaining': 0,
                'cooldown_remaining': 0, 'qty': 0,
            })

            # Decrement cooldown
            if state['cooldown_remaining'] > 0:
                state['cooldown_remaining'] -= 1
                print(f"    Cooldown: {state['cooldown_remaining']} bars remaining")

            # Decrement hold and check exit for open positions
            if state['position'] != 0:
                state['hold_remaining'] -= 1
                if state['hold_remaining'] <= 0:
                    # Exit position
                    exit_result = self._exit_position(symbol, state, capital, peak_capital)
                    if exit_result:
                        capital, peak_capital, trade_line = exit_result
                        trade_report_lines.append(trade_line)
                        trades_closed += 1
                        print(f"    ✅ EXIT {symbol} — {trade_line}")

            # Generate signal if no position and not in cooldown
            if state['position'] == 0 and state['cooldown_remaining'] <= 0:
                sig = self.gen.generate_signal(symbol)
                if sig:
                    signals_total += 1
                    log_signal(self.conn, symbol, int(time.time() * 1000),
                               sig['proba'], sig['signal'], capital)

                    print(f"    Signal: proba={sig['proba']:.4f} → {'LONG' if sig['signal']==1 else 'SHORT' if sig['signal']==-1 else 'FLAT'}")

                    if sig['signal'] != 0:
                        # Enter position
                        entry_result = self._enter_position(
                            symbol, sig, state, capital, peak_capital
                        )
                        if entry_result:
                            capital, peak_capital = entry_result
                            trades_opened += 1
                else:
                    print(f"    ⚠️ No signal generated")

        # Update capital
        update_capital(self.conn, capital, peak_capital)

        # Save state
        save_state(self.conn, states)

        # Summary
        duration_ms = int((time.time() - start_time) * 1000)
        dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

        print(f"\n{'='*60}")
        print(f"  RUN SUMMARY")
        print(f"{'='*60}")
        print(f"  Signals: {signals_total} | Opened: {trades_opened} | Closed: {trades_closed}")
        print(f"  Capital: ${capital:.2f} | Peak: ${peak_capital:.2f} | DD: {dd:.2f}%")
        print(f"  Duration: {duration_ms}ms")

        log_run(self.conn, duration_ms, signals_total, trades_opened,
                trades_closed, capital, peak_capital, dd)

        # Send Telegram report
        if trade_report_lines or self.report_mode:
            msg = self._build_report(capital, peak_capital, dd, trades_opened,
                                     trades_closed, trade_report_lines)
            self.send_telegram(msg)

        return capital, peak_capital, dd

    def _enter_position(self, symbol: str, sig: Dict, state: Dict,
                        capital: float, peak_capital: float) -> Optional[Tuple[float, float]]:
        """Enter a paper trade position — simulates in DB + places real MARKET order on Binance testnet."""
        direction = sig['signal']  # 1=long, -1=short
        proba = sig['proba']
        direction_label = 'LONG' if direction == 1 else 'SHORT'

        entry_price = None
        client = self._get_client()

        # Step 1: Get current price and place real order on Binance testnet
        try:
            ob = client.get_orderbook_depth(symbol)
            if direction == 1:
                price_ref = float(ob['asks'][0][0])
            else:
                price_ref = float(ob['bids'][0][0])
        except Exception as e:
            print(f"    ⚠️ Orderbook fetch failed: {e}")
            price_ref = 50000.0  # fallback

        # Calculate position size: 1% margin × 10x leverage = 10% exposure
        position_value = capital * MARGIN_PCT * LEVERAGE_X
        raw_qty = position_value / price_ref

        # Round qty to LOT_SIZE step_size for the symbol
        step_size = 0.01  # default
        min_qty = 0.01
        try:
            exch_info = client.get_exchange_info()
            if isinstance(exch_info, dict):
                for sym_info in exch_info.get('symbols', []):
                    if sym_info.get('symbol') == symbol.upper():
                        for f in sym_info.get('filters', []):
                            if f.get('filter_type') == 'LOT_SIZE':
                                step_size = float(f.get('step_size', 0.01))
                                min_qty = float(f.get('min_qty', 0.01))
                                break
                        break
        except Exception:
            pass
        qty = (int(raw_qty / step_size)) * step_size
        if qty < min_qty:
            qty = min_qty
            print(f"    ⚠️ Qty below min, using {min_qty}")
        # Keep reasonable decimal precision (handle integer or float step_size)
        step_str = f"{step_size}".rstrip('0').rstrip('.')
        if '.' in step_str:
            step_prec = max(6, len(step_str.split('.')[1]))
        else:
            step_prec = 6
        qty = round(qty, step_prec)

        side = "BUY" if direction == 1 else "SELL"
        order_placed = False
        try:
            # Place real order on Binance testnet
            order_resp = client.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=qty,
            )
            order_id = order_resp.get('order_id', '?')
            status = order_resp.get('status', 'N/A')
            executed_qty = float(order_resp.get('executed_qty', 0))
            avg_price_raw = float(order_resp.get('avg_price', 0))

            if executed_qty > 0 and avg_price_raw > 0:
                # Order filled on testnet — use avg_price
                entry_price = avg_price_raw
                order_placed = True
                print(f"    📡 ORDER FILLED: {side} {symbol} qty={executed_qty} @ ${avg_price_raw:.4f} (order #{order_id})")
            else:
                # Order response shows NEW/unfilled — testnet has delayed fill updates
                # Wait briefly, then check if position was actually created
                time.sleep(2)
                actual_pos = None
                try:
                    positions = client.get_position_info()
                    for p in positions:
                        if p.get('symbol') == symbol.upper():
                            pos_amt = float(p.get('position_amt', 0))
                            if abs(pos_amt) > 0.0001:
                                actual_pos = p
                                break
                except Exception:
                    pass

                if actual_pos is not None:
                    pos_amt = float(actual_pos.get('position_amt', 0))
                    notional = float(actual_pos.get('notional', 0))
                    entry_price = abs(notional / pos_amt) if pos_amt != 0 else price_ref
                    order_placed = True
                    print(f"    📡 ORDER FILLED (delayed): {side} {symbol} {pos_amt:.4f} @ notional=${notional:.2f} avg=${entry_price:.4f}")
                else:
                    # Genuinely not filled — use orderbook price
                    entry_price = price_ref * (1 + SLIPPAGE) if direction == 1 else price_ref * (1 - SLIPPAGE)
                    print(f"    📡 ORDER SUBMITTED (pending): {side} {symbol} order #{order_id} status={status} — using orderbook price ${entry_price:.4f}")
        except Exception as e:
            print(f"    ⚠️ Binance order failed (paper-only fallback): {e}")
            # Paper-only fallback: use orderbook price with slippage
            entry_price = price_ref * (1 + SLIPPAGE) if direction == 1 else price_ref * (1 - SLIPPAGE)

        # Apply fee
        entry_cost = entry_price * qty * TAKER_FEE

        state['position'] = direction
        state['entry_price'] = entry_price
        state['entry_time'] = int(time.time() * 1000)
        state['entry_proba'] = proba
        state['hold_remaining'] = HOLD_BARS
        state['cooldown_remaining'] = 0
        state['qty'] = qty

        # Deduct entry fee from capital
        capital -= entry_cost
        peak_capital = max(peak_capital, capital)

        print(f"    ✅ ENTER {direction_label} @ ${entry_price:.4f} | qty={qty:.6f} | proba={proba:.4f}")

        return capital, peak_capital

    def _exit_position(self, symbol: str, state: Dict,
                       capital: float, peak_capital: float) -> Optional[Tuple[float, float, str]]:
        """Exit a paper trade position — simulates in DB + places real MARKET order on Binance testnet."""
        direction = state['position']
        direction_label = 'LONG' if direction == 1 else 'SHORT'

        qty = state['qty']
        entry_price = state['entry_price']

        # Place real exit order on Binance testnet
        client = self._get_client()
        exit_price = None

        try:
            # For exit: LONG close = SELL, SHORT close = BUY
            side = "SELL" if direction == 1 else "BUY"
            order_resp = client.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=qty,
            )
            order_id = order_resp.get('order_id', '?')
            executed_qty = float(order_resp.get('executed_qty', 0))
            avg_price_raw = float(order_resp.get('avg_price', 0))

            if executed_qty > 0 and avg_price_raw > 0:
                exit_price = avg_price_raw
                print(f"    📡 EXIT ORDER FILLED: {side} {symbol} qty={executed_qty} @ ${avg_price_raw:.4f} (order #{order_id})")
            else:
                # Order response shows NEW/unfilled — check actual position
                status = order_resp.get('status', 'N/A')
                time.sleep(2)
                try:
                    positions = client.get_position_info()
                    for p in positions:
                        if p.get('symbol') == symbol.upper():
                            pos_amt = float(p.get('position_amt', 0))
                            # If LONG was closed: pos_amt should be 0 or reduced
                            if abs(pos_amt) < abs(state.get('position', 0)) * 0.1:
                                print(f"    📡 EXIT POSITION CLOSED: {side} {symbol} remaining_pos={pos_amt:.4f} (order #{order_id})")
                                break
                except Exception:
                    pass
                # Fallback: get price from orderbook with slippage
                ob = client.get_orderbook_depth(symbol)
                if direction == 1:
                    exit_price = float(ob['bids'][0][0]) * (1 - SLIPPAGE)
                else:
                    exit_price = float(ob['asks'][0][0]) * (1 + SLIPPAGE)
                print(f"    📡 EXIT ORDER SUBMITTED (pending): {side} {symbol} order #{order_id} status={status} — using orderbook price ${exit_price:.4f}")
        except Exception as e:
            print(f"    ⚠️ Binance exit order failed (paper-only fallback): {e}")
            # Paper-only fallback: get price from orderbook
            try:
                ob = client.get_orderbook_depth(symbol)
                if direction == 1:
                    exit_price = float(ob['bids'][0][0]) * (1 - SLIPPAGE)
                else:
                    exit_price = float(ob['asks'][0][0]) * (1 + SLIPPAGE)
            except Exception:
                exit_price = entry_price * 1.001  # placeholder

        # Calculate PnL
        if direction == 1:
            raw_pnl = qty * (exit_price - entry_price)
        else:
            raw_pnl = qty * (entry_price - exit_price)

        # Fees on entry + exit
        entry_cost = entry_price * qty * TAKER_FEE
        exit_cost = exit_price * qty * TAKER_FEE
        pnl_net = raw_pnl - entry_cost - exit_cost

        # PnL percent
        pnl_pct = pnl_net / capital * 100

        capital += pnl_net
        peak_capital = max(peak_capital, capital)

        hold_bars = HOLD_BARS - state['hold_remaining']

        trade = {
            'symbol': symbol,
            'direction': direction_label.lower(),
            'entry_time': state['entry_time'],
            'exit_time': int(time.time() * 1000),
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_pct,
            'entry_proba': state['entry_proba'],
            'hold_bars': hold_bars,
            'exit_reason': 'hold_expiry',
        }

        log_trade(self.conn, trade)

        # Reset position state
        state['position'] = 0
        state['entry_price'] = 0
        state['entry_time'] = 0
        state['entry_proba'] = 0
        state['qty'] = 0
        state['cooldown_remaining'] = COOLDOWN_BARS

        pnl_icon = "🟢" if pnl_net > 0 else "🔴"
        trade_line = (
            f"{pnl_icon} {symbol}: {direction_label} @ ${entry_price:.2f} → ${exit_price:.2f} "
            f"| PnL: ${pnl_net:.2f} ({pnl_pct:+.2f}%) | hold={hold_bars}bars | proba={state['entry_proba']:.3f}"
        )

        print(f"    ✅ EXIT {direction_label} @ ${exit_price:.2f} | PnL=${pnl_net:.2f} ({pnl_pct:+.2f}%)")

        return capital, peak_capital, trade_line

    def _build_report(self, capital: float, peak_capital: float, dd: float,
                      opened: int, closed: int, trade_lines: List[str]) -> str:
        """Build Telegram report message."""
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
        pnl = capital - INITIAL_CAPITAL
        pnl_pct = pnl / INITIAL_CAPITAL * 100

        # Get recent trade stats
        c = self.conn.cursor()
        c.execute("""
            SELECT COUNT(*), COALESCE(SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(pnl_net), 0), COALESCE(SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN pnl_net < 0 THEN ABS(pnl_net) ELSE 0 END), 0)
            FROM paper_trades
        """)
        row = c.fetchone()
        total_trades, wins, total_pnl, gross_profit, gross_loss = row
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Get open positions
        running_positions = []
        c.execute("SELECT symbol, position FROM paper_state WHERE position != 0")
        for sym, pos in c.fetchall():
            running_positions.append(f"  {'🟢' if pos==1 else '🔴'} {sym}: {'LONG' if pos==1 else 'SHORT'}")

        msg = (
            f"📊 *Mimia Paper Trade Report*\n"
            f"`{now}`\n\n"
            f"*Portfolio*\n"
            f"  Capital: `${capital:.2f}` | Peak: `${peak_capital:.2f}`\n"
            f"  PnL: `${pnl:+.2f}` ({pnl_pct:+.2f}%) | DD: `{dd:.2f}%`\n"
            f"  Win Rate: `{win_rate:.1f}%` | PF: `{pf:.2f}` ({total_trades} trades)\n\n"
        )

        if running_positions:
            msg += f"*Open Positions ({len(running_positions)})*\n"
            msg += "\n".join(running_positions) + "\n\n"

        if trade_lines:
            msg += f"*Recent Trades ({len(trade_lines)})*\n"
            msg += "\n".join(trade_lines[-5:]) + "\n\n"

        msg += (
            f"*This Run*\n"
            f"  Trades opened: `{opened}` | closed: `{closed}`\n"
            f"---\n"
            f"`mimia quant • paper trading`"
        )

        if len(msg) > 4096:
            msg = msg[:4000] + "\n\n... (truncated)"

        return msg


# ─── CLI ────────────────────────────────────────────────────────────────────
def show_status():
    """Show current paper trading status."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))

    capital, peak_capital = get_capital(conn)
    pnl = capital - INITIAL_CAPITAL
    pnl_pct = pnl / INITIAL_CAPITAL * 100
    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

    print(f"\n📊 Paper Trading Status")
    print(f"{'='*50}")
    print(f"  Capital: ${capital:.2f}")
    print(f"  Peak:    ${peak_capital:.2f}")
    print(f"  PnL:     ${pnl:+.2f} ({pnl_pct:+.2f}%)")
    print(f"  DD:      {dd:.2f}%")

    # Trades summary
    c = conn.cursor()
    c.execute("SELECT COUNT(*), SUM(pnl_net) FROM paper_trades")
    cnt, total_pnl = c.fetchone()
    total_pnl = total_pnl or 0
    print(f"\n  Total Trades: {cnt}")
    print(f"  Cumulative PnL: ${total_pnl:.2f}")

    if cnt > 0:
        c.execute("SELECT COUNT(*), SUM(pnl_net) FROM paper_trades WHERE pnl_net > 0")
        wins, win_pnl = c.fetchone()
        win_pnl = win_pnl or 0
        win_rate = wins / cnt * 100
        print(f"  Wins: {wins}/{cnt} ({win_rate:.1f}%)")

        c.execute("SELECT SUM(pnl_net) FROM paper_trades WHERE pnl_net < 0")
        loss_pnl = c.fetchone()[0] or 0
        pf = win_pnl / abs(loss_pnl) if loss_pnl != 0 else float('inf')
        print(f"  Profit Factor: {pf:.2f}")

    # Open positions
    print(f"\n  Open Positions:")
    c.execute("SELECT symbol, position, entry_price, hold_remaining FROM paper_state WHERE position != 0")
    positions = c.fetchall()
    if positions:
        for sym, pos, price, hold in positions:
            dir_str = "🟢 LONG" if pos == 1 else "🔴 SHORT"
            print(f"    {dir_str} {sym} @ ${price:.2f} | hold_remaining={hold}")
    else:
        print(f"    None (all flat)")

    # Last runs
    print(f"\n  Last 5 Runs:")
    c.execute("SELECT timestamp, signals_generated, trades_opened, trades_closed FROM paper_runs ORDER BY id DESC LIMIT 5")
    for ts, sigs, opened, closed in c.fetchall():
        t = datetime.fromtimestamp(ts / 1000).strftime('%m-%d %H:%M')
        print(f"    {t}: {sigs} signals, {opened} opened, {closed} closed")

    conn.close()


def show_report():
    """Generate and send a full daily report."""
    init_db()
    pt = PaperTrader(report=True)

    capital, peak_capital = get_capital(pt.conn)
    dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0

    # Get today's trades
    today_start = int((datetime.utcnow().replace(hour=0, minute=0, second=0) - timedelta(days=1)).timestamp() * 1000)
    c = pt.conn.cursor()
    c.execute("""
        SELECT symbol, direction, entry_time, exit_time, entry_price, exit_price,
               pnl_net, pnl_pct, entry_proba
        FROM paper_trades
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
    c.execute("SELECT COUNT(*), SUM(pnl_net) FROM paper_trades WHERE entry_time >= ?", (week_start,))
    week_cnt, week_pnl = c.fetchone()
    week_pnl = week_pnl or 0
    c.execute("SELECT COUNT(*) FROM paper_trades WHERE entry_time >= ? AND pnl_net > 0", (week_start,))
    week_wins = c.fetchone()[0] or 0
    week_wr = week_wins / week_cnt * 100 if week_cnt > 0 else 0

    # Open positions
    c.execute("SELECT symbol, position, entry_price FROM paper_state WHERE position != 0")
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
        FROM paper_trades GROUP BY symbol ORDER BY pnl DESC
    """)
    symbol_pnl = c.fetchall()
    if symbol_pnl:
        msg += "*Performance by Symbol*\n"
        for sym, cnt, pnl in symbol_pnl:
            icon = "🟢" if (pnl or 0) > 0 else "🔴"
            msg += f"  {icon} {sym}: ${pnl:.2f} ({cnt} trades)\n"

    pt.send_telegram(msg)
    pt.close()


def reset_state():
    """Reset all paper trading state (positions, but keep trade history)."""
    init_db()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("UPDATE paper_state SET position=0, entry_price=0, entry_time=0, "
              "entry_proba=0, hold_remaining=0, cooldown_remaining=0, qty=0")
    c.execute("DELETE FROM paper_signals")
    c.execute("DELETE FROM paper_capital")
    c.execute("INSERT INTO paper_capital (timestamp, capital, peak_capital) VALUES (?, ?, ?)",
              (int(time.time() * 1000), INITIAL_CAPITAL, INITIAL_CAPITAL))
    conn.commit()
    conn.close()
    print("✅ State reset. All positions set to flat, capital reset to $5000.")


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mimia Paper Trading Engine')
    parser.add_argument('--init', action='store_true', help='Initialize database')
    parser.add_argument('--status', action='store_true', help='Show current status')
    parser.add_argument('--report', action='store_true', help='Send daily report')
    parser.add_argument('--reset', action='store_true', help='Reset trading state')
    args = parser.parse_args()

    if args.init:
        init_db()
        print(f"✅ Database initialized at {DB_PATH}")
    elif args.status:
        show_status()
    elif args.report:
        show_report()
    elif args.reset:
        reset_state()
    else:
        # Run one paper trading cycle
        pt = PaperTrader()
        try:
            pt.run()
        except KeyboardInterrupt:
            print("\n⏹ Stopped by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            traceback.print_exc()
        finally:
            pt.close()
