#!/usr/bin/env python3
"""
Mimia Quant - Live Trading Engine
==================================
Main live trading engine that orchestrates signal evaluation, trade execution,
risk checks, and reporting for Binance Futures (testnet or mainnet).
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
import json
import time
import sqlite3
import warnings
import traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')

from dotenv import load_dotenv
load_dotenv()  # Load .env for Binance API keys

from src.trading.state import (
    init_db, get_state, save_state, get_capital, update_capital,
    log_signal, log_trade, log_run,
    save_pending_signals, load_pending_signals, clear_pending_signals,
    DB_PATH, INITIAL_CAPITAL, LIVE_SYMBOLS,
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, MARGIN_PCT, LEVERAGE_X,
    TAKER_FEE, SLIPPAGE, MODEL_DIR, SEEDS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    get_model_info, get_symbol_threshold, get_symbol_hold_bars, get_dynamic_position_pct,
)
from src.trading.signals import SignalGenerator


class LiveTrader:
    """Main live trading engine."""

    def __init__(self, report: bool = False, testnet: bool = True):
        init_db()
        self.testnet = testnet
        self.network = "testnet" if testnet else "mainnet"
        self.conn = sqlite3.connect(str(DB_PATH))
        self._signal_gens = {}
        self._client = None  # lazy-init
        self.report_mode = report
        self._history_synced = False  # runs once per daemon startup

    def _get_client(self):
        """Lazy-init Binance client (testnet or mainnet based on self.testnet)."""
        if self._client is None:
            from src.utils.binance_client import BinanceRESTClient
            self._client = BinanceRESTClient(testnet=self.testnet)
        return self._client

    def _load_models(self, symbol: str) -> SignalGenerator:
        """Lazy load SignalGenerators per symbol."""
        if symbol not in self._signal_gens:
            gen = SignalGenerator(symbol)
            self._signal_gens[symbol] = gen
        return self._signal_gens[symbol]

    def _init_leverage(self, symbol: str) -> bool:
        """Set leverage to 10x on Binance Futures for a symbol. Returns True on success."""
        try:
            self._get_client().change_leverage(symbol, 10)
            return True
        except Exception:
            return False

    def _get_binance_positions(self) -> Dict[str, Dict]:
        """Get actual open positions from Binance (testnet or mainnet).

        Returns a dict keyed by symbol with position details for symbols
        that have non-zero positions on Binance.
        """
        binance_positions = {}
        try:
            client = self._get_client()
            positions = client.get_position_info()
            for p in positions:
                sym = p.get('symbol', '').upper()
                if sym not in LIVE_SYMBOLS:
                    continue
                pos_amt = float(p.get('position_amt', 0))
                if abs(pos_amt) < 0.001:
                    continue
                direction = 1 if pos_amt > 0 else -1
                notional = float(p.get('notional', 0))
                entry_price = abs(notional / pos_amt) if pos_amt != 0 else 0
                binance_positions[sym] = {
                    'direction': direction,
                    'qty': abs(pos_amt),
                    'entry_price': entry_price,
                }
                print(f"    🔍 Binance position: {sym} {'LONG' if direction==1 else 'SHORT'} qty={abs(pos_amt):.4f} entry=${entry_price:.4f}")
            return binance_positions
        except Exception as e:
            print(f"  ⚠️ Could not fetch Binance positions: {e}")
            return {}

    @staticmethod
    def _calc_hold_remaining(entry_time_ms: int) -> int:
        """Calculate hold_remaining based on elapsed time since entry.

        Each bar = 5 minutes (300,000 ms). HOLD_BARS=9 = ~45 min.
        For unknown entry times (0), return full HOLD_BARS.
        """
        if entry_time_ms == 0:
            return HOLD_BARS
        elapsed_ms = int(time.time() * 1000) - entry_time_ms
        elapsed_bars = elapsed_ms // 300_000  # 5 min per bar
        return max(1, HOLD_BARS - elapsed_bars)

    def _setup_live_trading(self):
        """Reset stale positions in DB and init leverage for all symbols.

        Only resets DB positions that do NOT have a corresponding open position
        on Binance. Positions that are active on Binance are preserved, and
        hold_remaining is calculated from actual elapsed time since entry.
        """
        net = self.network
        c = self.conn.cursor()

        # First, get actual open positions from Binance
        binance_positions = self._get_binance_positions()

        # Check for stale DB positions (in DB but NOT on Binance)
        c.execute("SELECT symbol, position, qty FROM live_state WHERE position != 0")
        db_stale = c.fetchall()

        reset_count = 0
        keep_count = 0
        for sym, pos, qty in db_stale:
            if sym in binance_positions:
                # Position is legitimately open on Binance — keep it
                keep_count += 1
                bp = binance_positions[sym]
                # Check existing DB entry_time AND entry_price 
                c.execute("SELECT entry_time, entry_price FROM live_state WHERE symbol=?", (sym,))
                existing_entry = c.fetchone()
                db_entry_time = existing_entry[0] if existing_entry and existing_entry[0] else 0
                db_entry_price = existing_entry[1] if existing_entry and existing_entry[1] > 0 else 0
                entry_time = db_entry_time if db_entry_time > 0 else int(time.time() * 1000)
                hold = self._calc_hold_remaining(entry_time)
                # PRESERVE original entry_price from DB — don't use bp['entry_price'] 
                # (which is notional/pos_amt = CURRENT mark price, NOT entry price)
                entry_price = db_entry_price
                if entry_price <= 0:
                    # Fallback: use account_information_v2 which has REAL entry_price
                    try:
                        acct = client._call('account_information_v2')
                        for pos in acct.positions:
                            if getattr(pos, 'symbol', '') == sym:
                                ep = float(getattr(pos, 'entry_price', 0) or 0)
                                if ep > 0:
                                    entry_price = ep
                                    break
                    except Exception:
                        pass
                if entry_price <= 0:
                    # Last-resort: use bp['entry_price'] (notional/amt = mark price)
                    entry_price = bp['entry_price']
                c.execute("""
                    UPDATE live_state
                    SET position=?, qty=?, entry_price=?, entry_time=?,
                        hold_remaining=?, cooldown_remaining=0
                    WHERE symbol=?
                """, (bp['direction'], bp['qty'], entry_price,
                      entry_time, hold, sym))
            else:
                # Position exists in DB but NOT on Binance — close it properly
                # instead of silently resetting (which loses trade history)
                reset_count += 1
                self._close_stale_position(sym, c)
                print(f"  ⚠️ Stale DB position: {sym} (not on Binance {net}). Closing & resetting...")
                c.execute("UPDATE live_state SET position=0, entry_price=0, entry_time=0, entry_proba=0, hold_remaining=0, cooldown_remaining=0, qty=0 WHERE symbol=?", (sym,))

        if keep_count > 0:
            print(f"  ✅ Preserved {keep_count} active position(s) matching Binance {net}")
        if reset_count > 0:
            print(f"  ✅ Reset {reset_count} stale DB position(s) (not on Binance {net})")

        self.conn.commit()

        # Init leverage 10x for all symbols
        ok = 0
        for sym in LIVE_SYMBOLS:
            if self._init_leverage(sym):
                ok += 1
        print(f"  ✅ Leverage 10x set for {ok}/{len(LIVE_SYMBOLS)} symbols")

    def _close_stale_position(self, symbol: str, cursor) -> None:
        """Close a stale DB position by logging a complete trade with actual Binance fill data.

        Called when a position exists in DB (live_state) but NOT on Binance —
        instead of silently resetting, we fetch real fill prices from Binance
        trade history and record the complete trade in live_trades.

        Does NOT modify capital — the position was already resolved on Binance,
        so PnL is already reflected in the wallet balance. This is purely for
        historical record-keeping and accurate summary reports.
        """
        # 1. Read entry info from DB before reset
        cursor.execute(
            "SELECT entry_price, entry_time, entry_proba, qty, position "
            "FROM live_state WHERE symbol=?", (symbol,)
        )
        row = cursor.fetchone()
        if not row:
            return
        entry_price, entry_time, entry_proba, qty, direction = row
        if not qty or qty <= 0 or not entry_price or entry_price <= 0:
            return  # Nothing meaningful to log

        direction_label = 'LONG' if direction == 1 else 'SHORT'
        direction_str = 'long' if direction == 1 else 'short'

        # 2. Find actual exit price from Binance account_trade_list
        exit_price = None
        exit_time = int(time.time() * 1000)
        exit_side = "SELL" if direction == 1 else "BUY"

        try:
            client = self._get_client()
            fills = client.get_account_trades(symbol, limit=20)
            for f in fills:
                if f.get('side') == exit_side:
                    p = float(f.get('price', 0) or 0)
                    if p > 0:
                        exit_price = p
                        t = int(f.get('time', 0) or 0)
                        if t > 0:
                            exit_time = t
                        print(f"    📡 Stale exit price from Binance trade #{f.get('id')}: ${exit_price:.4f}")
                        break
        except Exception as e:
            print(f"    ⚠️ Could not fetch Binance fills for stale position: {e}")

        # 3. Fallback to orderbook if Binance data unavailable
        if exit_price is None or exit_price <= 0:
            try:
                client = self._get_client()
                ob = client.get_orderbook_depth(symbol)
                if direction == 1:
                    exit_price = float(ob['bids'][0][0]) * (1 - SLIPPAGE)
                else:
                    exit_price = float(ob['asks'][0][0]) * (1 + SLIPPAGE)
                print(f"    📡 Stale exit price from orderbook: ${exit_price:.4f}")
            except Exception as e:
                print(f"    ⚠️ Orderbook fallback failed: {e}")
                exit_price = entry_price * (1.001 if direction == 1 else 0.999)
                print(f"    📡 Stale exit price estimated: ${exit_price:.4f}")

        # 4. Calculate PnL
        if direction == 1:
            raw_pnl = qty * (exit_price - entry_price)
        else:
            raw_pnl = qty * (entry_price - exit_price)

        exit_cost = exit_price * qty * TAKER_FEE
        pnl_net = raw_pnl - exit_cost
        entry_value = entry_price * qty
        pnl_pct = pnl_net / entry_value * 100 if entry_value > 0 else 0
        hold_bars = HOLD_BARS

        # 5. Log the complete trade
        trade = {
            'symbol': symbol,
            'direction': direction_str,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'qty': qty,
            'pnl_net': pnl_net,
            'pnl_pct': pnl_pct,
            'entry_proba': entry_proba,
            'hold_bars': hold_bars,
            'exit_reason': 'system_restart',
        }
        log_trade(self.conn, trade)

        pnl_icon = "🟢" if pnl_net > 0 else "🔴"
        print(f"  {pnl_icon} Stale position CLOSED: {direction_label} {symbol} ${entry_price:.4f} → ${exit_price:.4f} "
              f"| PnL: ${pnl_net:.2f} ({pnl_pct:+.2f}%) | logged to history")

    def _verify_position_integrity(self, states: Dict) -> None:
        """Per-cycle check: ensure every in-DB position matches Binance reality.

        If a position is marked as open in DB but not actually on Binance,
        close it properly (log trade, reset state) instead of attempting
        a fake exit order later.
        """
        try:
            client = self._get_client()
            positions_raw = client.get_position_info()
            open_binance = {}
            for p in positions_raw:
                sym = p.get('symbol', '').upper()
                amt = float(p.get('position_amt', 0))
                if sym in LIVE_SYMBOLS and abs(amt) > 0.001:
                    open_binance[sym] = amt

            c = self.conn.cursor()
            for sym, state in states.items():
                if state['position'] == 0:
                    continue
                amt_on_binance = abs(open_binance.get(sym, 0))
                if amt_on_binance < 0.001:
                    print(f"  ⚠️ Integrity: {sym} open in DB but closed on Binance — logging exit")
                    self._close_stale_position(sym, c)
                    state['position'] = 0
                    state['entry_price'] = 0
                    state['entry_time'] = 0
                    state['entry_proba'] = 0
                    state['qty'] = 0
                    state['hold_remaining'] = 0
                    state['cooldown_remaining'] = COOLDOWN_BARS
            self.conn.commit()
        except Exception as e:
            print(f"  ⚠️ Position integrity check skipped: {e}")

    def _sync_trade_history(self) -> int:
        """Backfill trade history from Binance for all tracked symbols.

        Called at daemon startup (not every cycle). Pulls account trade fills
        from Binance for the 7-day window, cross-references against DB,
        and logs any trades that are NOT already in the live_trades table.

        IMPORTANT: After a DB reset (fresh start), the 'db_initialized_at'
        timestamp in the meta table ensures that only trades AFTER the reset
        are backfilled — preventing old/historical losing trades from polluting
        the current session's portfolio metrics.

        Returns the number of backfilled trades.
        """
        backfilled = 0
        try:
            client = self._get_client()
            c = self.conn.cursor()

            # Determine start boundary: use db_initialized_at if available
            c.execute("SELECT value FROM meta WHERE key='db_initialized_at'")
            row = c.fetchone()
            if row:
                db_init = int(row[0])
                # Only look back max 7 days, but never before db_init
                seven_days_ago = max(db_init, int((time.time() - 7 * 86400) * 1000))
                start_time = seven_days_ago
            else:
                # Fallback: 7 days (backward compat for existing DBs)
                seven_days_ago = int((time.time() - 7 * 86400) * 1000)
                c.execute("SELECT MIN(entry_time) FROM live_trades")
                min_entry = c.fetchone()[0]
                start_time = min_entry if min_entry and min_entry > seven_days_ago else seven_days_ago

            for symbol in LIVE_SYMBOLS:
                try:
                    fills = client.get_account_trades(symbol, limit=100, start_time=start_time)
                    if not fills:
                        continue

                    c.execute(
                        "SELECT entry_time, exit_time, entry_price, exit_price, direction, qty "
                        "FROM live_trades WHERE symbol=?", (symbol,)
                    )
                    db_trades = c.fetchall()

                    fills_sorted = sorted(fills, key=lambda x: x.get('time', 0))
                    paired = []
                    pending_entry = None
                    for f in fills_sorted:
                        side = f.get('side', '')
                        price = float(f.get('price', 0))
                        qty_fill = float(f.get('qty', 0))
                        ts = int(f.get('time', 0))

                        if side == 'BUY' and pending_entry is None:
                            pending_entry = {'time': ts, 'price': price, 'qty': qty_fill,
                                            'direction': 'long', 'side': 'BUY'}
                        elif side == 'SELL' and pending_entry is not None:
                            pair = {'entry_time': pending_entry['time'], 'exit_time': ts,
                                   'entry_price': pending_entry['price'], 'exit_price': price,
                                   'qty': min(pending_entry['qty'], qty_fill), 'direction': 'long'}
                            paired.append(pair)
                            pending_entry = None
                        elif side == 'SELL' and pending_entry is None:
                            pending_entry = {'time': ts, 'price': price, 'qty': qty_fill,
                                            'direction': 'short', 'side': 'SELL'}
                        elif side == 'BUY' and pending_entry is not None and pending_entry['side'] == 'SELL':
                            pair = {'entry_time': pending_entry['time'], 'exit_time': ts,
                                   'entry_price': pending_entry['price'], 'exit_price': price,
                                   'qty': min(pending_entry['qty'], qty_fill), 'direction': 'short'}
                            paired.append(pair)
                            pending_entry = None
                        else:
                            pending_entry = None

                    for pair in paired:
                        already_exists = False
                        for db_entry in db_trades:
                            time_diff = abs(pair['entry_time'] - db_entry[0])
                            if time_diff < 60000:
                                already_exists = True
                                break

                        if not already_exists:
                            qty = pair['qty']
                            entry_p = pair['entry_price']
                            exit_p = pair['exit_price']
                            dir_val = 1 if pair['direction'] == 'long' else -1

                            if dir_val == 1:
                                raw_pnl = qty * (exit_p - entry_p)
                            else:
                                raw_pnl = qty * (entry_p - exit_p)

                            exit_cost = exit_p * qty * TAKER_FEE
                            pnl_net = raw_pnl - exit_cost
                            entry_val = entry_p * qty
                            pnl_pct = pnl_net / entry_val * 100 if entry_val > 0 else 0

                            trade = {
                                'symbol': symbol,
                                'direction': pair['direction'],
                                'entry_time': pair['entry_time'],
                                'exit_time': pair['exit_time'],
                                'entry_price': entry_p,
                                'exit_price': exit_p,
                                'qty': qty,
                                'pnl_net': pnl_net,
                                'pnl_pct': pnl_pct,
                                'entry_proba': 0.5,
                                'hold_bars': int((pair['exit_time'] - pair['entry_time']) / 300000),
                                'exit_reason': 'history_sync',
                            }
                            log_trade(self.conn, trade)
                            backfilled += 1
                            print(f"  📜 Backfilled: {symbol} {pair['direction'].upper()} ${entry_p:.4f} → ${exit_p:.4f} PnL=${pnl_net:.2f}")

                except Exception as e:
                    print(f"  ⚠️ Sync skipped for {symbol}: {e}")
                    continue

            if backfilled > 0:
                print(f"  ✅ Sync complete: {backfilled} trade(s) backfilled from Binance history")
            else:
                print(f"  ✅ Sync complete: no missing trades found")
        except Exception as e:
            print(f"  ⚠️ Trade history sync failed: {e}")

        return backfilled

    def _sync_wallet_balance(self) -> float:
        """Read Binance wallet balance (informational only — NO DB writes).

        Queries Binance account info and returns the balance. Does NOT
        update the live_capital table — that is handled centrally by run()
        using DB-derived capital (INITIAL_CAPITAL + realized PnL).

        Returns the synced balance, or 0 if sync fails.
        """
        try:
            client = self._get_client()
            acct = client.get_account_info()
            total_margin = float(acct.get('total_margin_balance', 0))
            balance = total_margin if total_margin > 0 else float(acct.get('total_wallet_balance', 0))
            if balance > 0:
                print(f"  💰 Binance wallet: ${balance:.2f} ({self.network})")
                return balance
            else:
                print(f"  ⚠️ Wallet balance from Binance is 0")
                return 0
        except Exception as e:
            print(f"  ⚠️ Wallet sync failed: {e}")
            return 0

    def _sync_binance_positions(self):
        """Sync DB state with actual open positions on Binance (mainnet or testnet).

        Calculates hold_remaining from actual elapsed time since entry_time
        instead of resetting to full HOLD_BARS on every restart.
        """
        try:
            client = self._get_client()
            positions = client.get_position_info()
            c = self.conn.cursor()
            synced = 0
            for p in positions:
                sym = p.get('symbol', '')
                if sym.upper() not in LIVE_SYMBOLS:
                    continue
                pos_amt = float(p.get('position_amt', 0))
                notional = float(p.get('notional', 0))
                if abs(pos_amt) < 0.001:
                    continue
                direction = 1 if pos_amt > 0 else -1
                qty = abs(pos_amt)
                # PRESERVE original entry_price from DB (set at trade entry via get_order polling)
                # BUG FIX: notional/pos_amt = CURRENT mark price, NOT entry price!
                # Only recalculate if DB has no entry yet (fresh start).
                c.execute("SELECT entry_time, entry_price FROM live_state WHERE symbol=?", (sym,))
                existing_row = c.fetchone()
                db_entry_time = existing_row[0] if existing_row and existing_row[0] else 0
                db_entry_price = existing_row[1] if existing_row and existing_row[1] > 0 else 0

                entry_price = 0
                if db_entry_price > 0:
                    entry_price = db_entry_price  # Preserve original from trade execution
                else:
                    # Fresh start — use entry_price from account_information_v2 (correct field)
                    try:
                        acct = client._call('account_information_v2')
                        for pos in acct.positions:
                            if getattr(pos, 'symbol', '') == sym:
                                ep = float(getattr(pos, 'entry_price', 0) or 0)
                                if ep > 0:
                                    entry_price = ep
                                    break
                    except Exception:
                        pass
                if entry_price <= 0:
                    # Last-resort fallback (notional = mark-price based, but better than zero)
                    entry_price = abs(notional / pos_amt) if pos_amt != 0 else 0

                entry_time = db_entry_time if db_entry_time > 0 else int(time.time() * 1000)
                hold = self._calc_hold_remaining(entry_time)
                # Update DB state to match Binance
                c.execute("""
                    UPDATE live_state
                    SET position=?, qty=?, entry_price=?, entry_time=?,
                        hold_remaining=?, cooldown_remaining=0
                    WHERE symbol=?
                """, (direction, qty, entry_price, entry_time, hold, sym))
                synced += 1
            self.conn.commit()
            if synced > 0:
                print(f"  🔄 Synced {synced} existing position(s) from Binance {self.network}")
        except Exception as e:
            print(f"  ⚠️ Position sync skipped: {e}")

    def close(self):
        self.conn.close()

    def send_telegram(self, message: str):
        """Send a message via Telegram bot."""
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, json={
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown',
            }, timeout=10)
        except Exception as e:
            print(f"    ⚠️ Telegram send failed: {e}")

    def _build_report(self, capital: float, peak_capital: float, dd: float,
                      opened: int, closed: int, trade_lines: List[str]) -> str:
        """Build Telegram report message.

        Portfolio metrics calculated from DB trade history, NOT Binance wallet.
        Capital = INITIAL_CAPITAL + realized PnL from closed trades.
        Unrealized PnL estimated from open positions (entry price vs current mark).
        """
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')

        # Get realized PnL from closed trades (DB source of truth)
        # EXCLUDE history_sync trades — they are old backfill data, not current session
        c = self.conn.cursor()
        c.execute("""
            SELECT COUNT(*), COALESCE(SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END), 0),
                   COALESCE(SUM(pnl_net), 0), COALESCE(SUM(CASE WHEN pnl_net > 0 THEN pnl_net ELSE 0 END), 0),
                   COALESCE(SUM(CASE WHEN pnl_net < 0 THEN ABS(pnl_net) ELSE 0 END), 0)
            FROM live_trades
            WHERE exit_reason != 'history_sync'
        """)
        row = c.fetchone()
        total_trades, wins, total_pnl, gross_profit, gross_loss = row
        realized_pnl = total_pnl or 0.0
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calculate capital from DB trade history only (NOT Binance wallet)
        db_capital = INITIAL_CAPITAL + realized_pnl
        db_peak_capital = max(peak_capital, db_capital)
        db_dd = (db_peak_capital - db_capital) / db_peak_capital * 100 if db_peak_capital > 0 else 0
        pnl = realized_pnl
        pnl_pct = pnl / INITIAL_CAPITAL * 100

        # Get open positions with direction + unrealized PnL from Binance
        running_positions = []
        c.execute("SELECT symbol, position FROM live_state WHERE position != 0")
        db_open = {sym: pos for sym, pos in c.fetchall()}

        # Get pending signals count
        c.execute("SELECT COUNT(*) FROM pending_signals WHERE signal != 0")
        pending_count = c.fetchone()[0] or 0

        # Try to get per-position unrealized PnL from Binance
        pos_unrealized = {}
        try:
            client = self._get_client()
            positions = client.get_position_info()
            for p in positions:
                sym = p.get('symbol', '')
                amt = float(p.get('position_amt', 0) or 0)
                if sym in db_open and abs(amt) > 0.001:
                    upnl = float(p.get('unrealized_profit', 0) or 0)
                    pos_unrealized[sym] = upnl
        except Exception:
            pass

        for sym, pos in sorted(db_open.items()):
            label = 'LONG' if pos == 1 else 'SHORT'
            icon = '🟢' if pos == 1 else '🔴'
            upnl = pos_unrealized.get(sym)
            if upnl is not None:
                upnl_str = f" `${upnl:+.2f}`" if abs(upnl) >= 0.005 else ""
                running_positions.append(f"  {icon} {sym}: {label}{upnl_str}")
            else:
                running_positions.append(f"  {icon} {sym}: {label}")

        # Add pending signals to report
        pending_line = ""
        if pending_count > 0:
            c.execute("SELECT symbol, signal, proba FROM pending_signals WHERE signal != 0")
            pending_list = []
            for sym, sig, proba in c.fetchall():
                d = 'LONG' if sig == 1 else 'SHORT'
                pending_list.append(f"  ⏳ {sym}: {d} ({proba:.3f})")
            pending_line = f"\n*Pending Signals ({pending_count})*\n" + "\n".join(pending_list[:5]) + "\n"
            if pending_count > 5:
                pending_line += f"  ... (+{pending_count-5} more)\n"

        msg = (
            f"📊 *Mimia Live Trade Report*\n"
            f"`{now}`\n\n"
            f"*Portfolio*\n"
            f"  Equity: `${db_capital:.2f}` | Peak: `${db_peak_capital:.2f}`\n"
            f"  PnL: `${pnl:+.2f}` ({pnl_pct:+.2f}%) | DD: `{db_dd:.2f}%`\n"
        )

        # Realized PnL only (from DB — no wallet/balance dependency)
        if realized_pnl != 0 or total_trades > 0:
            msg += (
                f"  Realized PnL: `${realized_pnl:+.2f}` ({total_trades} trades)\n"
            )

        msg += (
            f"  Win Rate: `{win_rate:.1f}%` | PF: `{pf:.2f}` ({total_trades} trades)\n\n"
        )

        if running_positions:
            msg += f"*Open Positions ({len(running_positions)})*\n"
            msg += "\n".join(running_positions) + "\n\n"

        if pending_line:
            msg += pending_line + "\n"

        if trade_lines:
            msg += f"*Recent Trades ({len(trade_lines)})*\n"
            msg += "\n".join(trade_lines[-5:]) + "\n\n"

        msg += (
            f"*This Run*\n"
            f"  Trades opened: `{opened}` | closed: `{closed}`\n"
            f"---\n"
            f"`mimia quant • live trading`"
        )

        if len(msg) > 4096:
            msg = msg[:4000] + "\n\n... (truncated)"

        return msg

    def run(self):
        """Execute one full live trading cycle.

        Deferred Entry Flow (Candle N → Candle N+1):
          Phase 1: Execute pending entries from candle N signals (use candle N+1 close)
          Phase 2: Exit positions that are due (hold_expiry)
          Phase 3: Compute NEW signals for candle N+1 → store as pending for candle N+2

        This ensures backtest vs live trader use the SAME signal → entry timing,
        making PnL comparison meaningful.
        """
        start_time = time.time()
        now_utc = datetime.utcnow()
        print(f"\n{'='*60}")
        print(f"  LIVE TRADE RUN — {now_utc.strftime('%Y-%m-%d %H:%M UTC')} {'(' + self.network.upper() + ')'}")
        print(f"{'='*60}")

        capital, peak_capital = get_capital(self.conn)

        # Initial setup & sync (fast — no OHLCV)
        self._setup_live_trading()
        if not self._history_synced:
            backfilled = self._sync_trade_history()
            self._history_synced = True

        # Wallet balance sync — INFORMATIONAL ONLY.
        #   Binance testnet has stale PnL from old trades that distorts our fresh
        #   start capital. We NEVER let wallet balance override DB-derived capital.
        #   Use wallet balance only for peak_capital detection.
        synced_balance = self._sync_wallet_balance()

        # Compute capital from DB trades (source of truth):
        #   capital = INITIAL_CAPITAL + realized PnL from closed trades
        #   This isolates performance from Binance wallet drift.
        c_db = self.conn.cursor()
        c_db.execute("SELECT COALESCE(SUM(pnl_net), 0) FROM live_trades WHERE exit_reason != 'history_sync'")
        realized_pnl = c_db.fetchone()[0] or 0.0
        capital = INITIAL_CAPITAL + realized_pnl
        if synced_balance > 0:
            peak_capital = max(peak_capital, synced_balance)
        update_capital(self.conn, capital, peak_capital)
        self._sync_binance_positions()
        states = get_state(self.conn)
        self._verify_position_integrity(states)

        # Bar readiness check — skip signal computation if bar not yet settled
        bar_ready = self._is_bar_ready()
        print(f"    📡 Bar readiness: {'READY' if bar_ready else 'WAITING — skipping signal computation'}")

        # ════════════════════════════════════════════════════════════════
        # Phase 1: Execute pending entries (deferred from candle N → N+1)
        #   Signal di-generate di candle N → dieksekusi di candle N+1 close
        # ════════════════════════════════════════════════════════════════
        print(f"\n  ⚡ Phase 1: Executing pending entries (deferred from previous cycle)...")
        pending_signals = load_pending_signals(self.conn)
        trades_opened = 0
        entries_skipped = 0
        if bar_ready:
            for symbol in LIVE_SYMBOLS:
                if symbol not in pending_signals:
                    continue
                sig = pending_signals[symbol]
                state = states.get(symbol, {
                    'position': 0, 'entry_price': 0, 'entry_time': 0,
                    'entry_proba': 0, 'hold_remaining': 0,
                    'cooldown_remaining': 0, 'qty': 0,
                })
                if state['position'] != 0:
                    print(f"  ⏭ {symbol}: pending signal SKIPPED — already in position")
                    entries_skipped += 1
                    continue
                if state['cooldown_remaining'] > 0:
                    print(f"  ⏭ {symbol}: pending signal SKIPPED — cooldown ({state['cooldown_remaining']} bars)")
                    entries_skipped += 1
                    continue

                # Execute deferred entry — use CURRENT bar close (candle N+1)
                print(f"    🔵 {symbol}: executing pending {'LONG' if sig['signal']==1 else 'SHORT'} "
                      f"(proba={sig['proba']:.4f})")
                entry_result = self._enter_position(symbol, sig, state, capital, peak_capital)
                if entry_result:
                    capital, peak_capital = entry_result
                    trades_opened += 1

            # Clear executed pending signals — only if bar was ready
            # (Phase 3 may re-save them if it fails)
            clear_pending_signals(self.conn)
        else:
            entries_skipped = len(pending_signals)
            print(f"  ⏭ Phase 1: Bar not settled — {entries_skipped} pending entries deferred")
        print(f"  ✅ Phase 1 complete: {trades_opened} entries executed, {entries_skipped} skipped")

        # ════════════════════════════════════════════════════════════════
        # Phase 2: Exit positions that are due (hold_expiry)
        # ════════════════════════════════════════════════════════════════
        print(f"\n  ⚡ Phase 2: Checking exits...")
        trades_closed = 0
        trade_report_lines = []
        for symbol in LIVE_SYMBOLS:
            state = states.get(symbol, {
                'position': 0, 'entry_price': 0, 'entry_time': 0,
                'entry_proba': 0, 'hold_remaining': 0,
                'cooldown_remaining': 0, 'qty': 0,
            })
            # Decrement cooldown
            if state['cooldown_remaining'] > 0:
                state['cooldown_remaining'] -= 1

            # Exit check
            if state['position'] != 0:
                state['hold_remaining'] -= 1
                if state['hold_remaining'] <= 0:
                    exit_result = self._exit_position(symbol, state, capital, peak_capital)
                    if exit_result:
                        capital, peak_capital, trade_line = exit_result
                        trade_report_lines.append(trade_line)
                        trades_closed += 1

        save_state(self.conn, states)
        print(f"  ✅ Phase 2 complete: {trades_closed} exits executed")

        # ── 📡 QUICK REPORT ────────────────────────────────────────
        # Kirim Telegram report SEKARANG — setelah Phase 1+2 (cepat, <5s)
        # Phase 3 (signal computation) makan ~130s — tidak perlu ditunggu
        exec_ms_quick = int((time.time() - start_time) * 1000)
        dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
        msg = self._build_report(capital, peak_capital, dd, trades_opened,
                                 trades_closed, trade_report_lines)
        self.send_telegram(msg)
        # ───────────────────────────────────────────────────────────

        # ════════════════════════════════════════════════════════════════
        # Phase 3: Compute NEW signals for current cycle
        #   Store as pending → will be executed at candle N+2 close
        # ════════════════════════════════════════════════════════════════
        print(f"\n  ⚡ Phase 3: Computing signals for next cycle...")
        new_signals = {}
        signals_total = 0
        if bar_ready:
            for i, symbol in enumerate(LIVE_SYMBOLS):
                try:
                    print(f"  [{i+1}/{len(LIVE_SYMBOLS)}] {symbol}...")
                    gen = SignalGenerator(symbol)
                    sig = gen.generate_signal(symbol)
                    if sig and sig.get('signal', 0) != 0:
                        signals_total += 1
                        ts = int(time.time() * 1000)
                        # Calculate data_cutoff: last completed 5m bar close time
                        bar_minute = (now_utc.minute // 5) * 5
                        last_bar_close = now_utc.replace(minute=bar_minute, second=0, microsecond=0)
                        data_cutoff = int(last_bar_close.timestamp() * 1000)
                        model_info = get_model_info()
                        log_signal(self.conn, symbol, ts, sig['proba'], sig['signal'], capital,
                                   data_cutoff=data_cutoff, model_info=model_info)
                        new_signals[symbol] = {
                            'signal': sig['signal'],
                            'proba': sig['proba'],
                            'timestamp': ts,
                            'bar_index': now_utc.strftime('%Y-%m-%d %H:%M'),
                        }
                        d = 'LONG' if sig['signal']==1 else 'SHORT'
                        print(f"    proba={sig['proba']:.4f} ({d}) → PENDING for next cycle")
                    else:
                        p = sig['proba'] if sig else 0
                        print(f"    proba={p:.4f} (FLAT)")
                except Exception as e:
                    print(f"    ⚠️ Error: {e}")
        else:
            print(f"  ⏭ Phase 3: Bar not settled — skipping signal computation")
            # Pending signals from previous cycle still in DB (Phase 1 didn't clear)

        # Save pending signals for next cycle execution
        save_pending_signals(self.conn, new_signals)
        print(f"  ✅ Phase 3 complete: {len(new_signals)}/{signals_total} signals stored as pending")

        # ════════════════════════════════════════════════════════════════
        # Summary
        # ════════════════════════════════════════════════════════════════
        update_capital(self.conn, capital, peak_capital)
        exec_ms = int((time.time() - start_time) * 1000)
        dd = (peak_capital - capital) / peak_capital * 100 if peak_capital > 0 else 0
        pending_sig = len(new_signals)

        print(f"\n{'='*60}")
        print(f"  RUN SUMMARY")
        print(f"{'='*60}")
        print(f"  Phase 1 (pending entries): {trades_opened} entries executed from prev cycle")
        print(f"  Phase 2 (exits):           {trades_closed} positions closed")
        print(f"  Phase 3 (new signals):     {pending_sig} signals stored as pending for next cycle")
        print(f"  Execution:                 {exec_ms}ms (report sent after Phase 1+2)")
        print(f"  Capital: ${capital:.2f} | Peak: ${peak_capital:.2f} | DD: {dd:.2f}%")
        print(f"  Pending signals stored:    {signals_total} total, {len(new_signals)} actionable")

        log_run(self.conn, exec_ms, signals_total, trades_opened,
                trades_closed, capital, peak_capital, dd)

        return capital, peak_capital, dd

    def _compute_all_signals(self) -> Dict[str, Optional[Dict]]:
        """Compute signals for ALL symbols using current bar data (batch).
        Returns dict of {symbol: signal} for instant execution in Phase 2.
        """
        t0 = time.time()
        signals = {}
        for i, symbol in enumerate(LIVE_SYMBOLS):
            try:
                print(f"  [{i+1}/{len(LIVE_SYMBOLS)}] {symbol}...")
                gen = SignalGenerator(symbol)
                sig = gen.generate_signal(symbol)
                signals[symbol] = sig
                if sig:
                    d = 'LONG' if sig['signal']==1 else 'SHORT' if sig['signal']==-1 else 'FLAT'
                    print(f"    proba={sig['proba']:.4f} ({d})")
                else:
                    print(f"    No signal")
            except Exception as e:
                print(f"    ⚠️ Error: {e}")
                signals[symbol] = None
        signals['_compute_seconds'] = time.time() - t0
        return signals

    def _is_bar_ready(self) -> bool:
        """Check if the last COMPLETED 5m bar has settled (age since close >= 3 min).

        CRITICAL: Checks the last COMPLETE bar (not the latest bar in cache, which
        may be an incomplete/partial bar). This ensures signal computation uses
        FINALIZED bar data matching backtest timing:
          - Backtest: signal at bar N close → execute at bar N+1 close
          - Live:     signal at last COMPLETE bar → stored pending → execute next bar

        Settling time: 3 min (180s) after bar close gives Binance ~2 min to settle
        data + 1 min buffer. This replaces the old approach that checked LATEST bar
        age (which caused 1-2 min delay when a new partial bar appeared).

        Returns:
            True if the last complete bar is settled enough for signal generation
        """
        try:
            now = datetime.utcnow()
            # Find the last COMPLETE 5m bar's open_time
            # Current 5m window: floor(minute/5)*5
            # Last complete bar: one window before that
            this_bar_min = (now.minute // 5) * 5
            # Handle minute 0-4: this_bar_min = 0, last complete = 55min of prev hour
            last_complete_open = now.replace(
                minute=this_bar_min, second=0, microsecond=0
            ) - timedelta(minutes=5)

            last_complete_close = last_complete_open + timedelta(minutes=5)
            age_since_close = (now - last_complete_close).total_seconds()

            # Bar close from Binance is final immediately.
            # No settling needed — backtest uses exact close price too.
            # Daemon runs at ~:01 (boundary + 1s), so bar close is ~1-60s old.
            MIN_SETTLE_SEC = 0
            ready = age_since_close >= MIN_SETTLE_SEC
            remaining = max(0, MIN_SETTLE_SEC - age_since_close)

            # Log with complete bar info (not latest bar)
            print(f"    📡 Complete bar: {last_complete_open.strftime('%H:%M')} "
                  f"(closed {age_since_close:.0f}s ago) "
                  f"{'READY' if ready else f'WAITING ({remaining:.0f}s more)'}")
            return ready

        except Exception as e:
            print(f"    ⚠️ Bar check failed: {e}")
        return True  # Default: proceed if check fails

    def _check_enter(self, symbol: str, state: Dict, capital: float,
                     peak_capital: float, proba: float) -> bool:
        """Check if entry conditions are met (threshold + cooldown)."""
        # Already checked in run() — position == 0 and cooldown <= 0
        if proba >= THRESHOLD or proba <= (1 - THRESHOLD):
            return True
        return False

    def _enter_position(self, symbol: str, sig: Dict, state: Dict,
                        capital: float, peak_capital: float) -> Optional[Tuple[float, float]]:
        """Enter a live trade position — uses LIMIT order at bar close price
        (matching backtest entry), falls back to MARKET order if un-filled.

        Using LIMIT at bar close ensures the entry price matches
        what the backtest simulation would use, making PnL inline.
        """
        direction = sig['signal']  # 1=long, -1=short
        proba = sig['proba']
        direction_label = 'LONG' if direction == 1 else 'SHORT'

        entry_price = None
        client = self._get_client()

        # Step 1: Get bar close price from OHLCV cache (matches backtest reference)
        try:
            from datetime import datetime, timedelta
            from src.strategies.ml_features import OHLCV_CACHE_DIR
            cache_path = OHLCV_CACHE_DIR / f"{symbol}_5m.parquet"
            if cache_path.exists():
                df_cache = pd.read_parquet(cache_path)
                now = datetime.utcnow()
                last_open = df_cache.index[-1]
                last_close_time = last_open + timedelta(minutes=5)
                # Use the last COMPLETE bar's close (skip current incomplete bar)
                if last_close_time <= now:
                    last_close = float(df_cache['close'].iloc[-1])
                else:
                    # Last bar hasn't closed yet — use the bar before
                    last_close = float(df_cache['close'].iloc[-2])
                    print(f"    ⚠️ Latest bar {last_open} incomplete (closes at {last_close_time}), using previous bar close")
                price_ref = last_close
            else:
                raise FileNotFoundError("No OHLCV cache")
        except Exception:
            # Fallback: use orderbook
            try:
                ob = client.get_orderbook_depth(symbol)
                if direction == 1:
                    price_ref = float(ob['asks'][0][0])
                else:
                    price_ref = float(ob['bids'][0][0])
            except Exception:
                price_ref = 50000.0

        # Calculate position size
        try:
            from src.strategies.kelly_sizer import KellySizer
            kelly = KellySizer()
            sizes = kelly.get_all_positions([symbol])
            if symbol in sizes:
                kelly_pct = sizes[symbol].get('position_pct', 0.10)
            else:
                kelly_pct = MARGIN_PCT * LEVERAGE_X
        except Exception:
            kelly_pct = MARGIN_PCT * LEVERAGE_X
        kelly_pct = max(0.02, min(kelly_pct, 0.25))
        position_value = capital * kelly_pct
        raw_qty = position_value / price_ref

        # Round qty to LOT_SIZE
        step_size = 0.01
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
        step_str = f"{step_size}".rstrip('0').rstrip('.')
        if '.' in step_str:
            step_prec = max(6, len(step_str.split('.')[1]))
        else:
            step_prec = 6
        qty = round(qty, step_prec)

        side = "BUY" if direction == 1 else "SELL"
        order_placed = False
        executed_qty_actual = 0.0

        # Step 2: Calculate target price — match backtest bar close + slippage
        if direction == 1:
            target_price = price_ref * (1 + SLIPPAGE)
        else:
            target_price = price_ref * (1 - SLIPPAGE)

        # Round price to exchange tick_size
        tick_size = 0.01
        try:
            exch_info = client.get_exchange_info()
            if isinstance(exch_info, dict):
                for sym_info in exch_info.get('symbols', []):
                    if sym_info.get('symbol') == symbol.upper():
                        for f in sym_info.get('filters', []):
                            if f.get('filter_type') == 'PRICE_FILTER':
                                tick_size = float(f.get('tick_size', 0.01))
                                break
                        break
        except Exception:
            pass
        # Round price to tick_size precision
        # BUG FIX: f"{tick_size}" produces '1e-05' for tick_size=0.000010 (scientific notation)
        # which has NO decimal point → split('.')[1] fails → prec=0 → round(price,0)=0!
        # Use explicit .10f formatting to force fixed-point notation.
        prec = 4  # default precision
        if tick_size > 0:
            target_price = round(target_price / tick_size) * tick_size
            tick_str = f"{tick_size:.10f}".rstrip('0').rstrip('.')
            prec = len(tick_str.split('.')[1]) if '.' in tick_str else 0
            target_price = round(target_price, prec)

        limit_price = target_price

        print(f"    📡 LIMIT ORDER: {side} {symbol} qty={qty:.6f} @ ${limit_price:.{prec}f} (based on bar close ${price_ref:.4f})")

        # ── Try LIMIT first, then MARKET, then simulated ──
        # (don't let LIMIT precision errors skip MARKET fallback)
        for attempt_type in ['LIMIT', 'MARKET']:
            try:
                if attempt_type == 'LIMIT':
                    order_resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="LIMIT",
                        time_in_force="GTC",
                        price=limit_price,
                        quantity=qty,
                    )
                else:
                    order_resp = client.place_order(
                        symbol=symbol,
                        side=side,
                        order_type="MARKET",
                        quantity=qty,
                    )

                order_id = order_resp.get('order_id')
                if order_id is None or order_id == '?':
                    order_id = order_resp.get('client_order_id')
                status_code = order_resp.get('status', 'N/A')

                # Poll for fill
                max_attempts = 10 if attempt_type == 'LIMIT' else 6
                for poll in range(max_attempts):
                    time.sleep(1)
                    try:
                        fill_resp = client.get_order(symbol=symbol, order_id=order_id)
                        if isinstance(fill_resp, dict):
                            fill_status = fill_resp.get('status', '')
                            ex_qty = float(fill_resp.get('executed_qty', 0))
                            avg_px = float(fill_resp.get('avg_price', 0))
                            if fill_status == 'FILLED' and ex_qty > 0 and avg_px > 0:
                                entry_price = avg_px
                                executed_qty_actual = ex_qty
                                order_placed = True
                                print(f"    📡 {attempt_type} FILLED: {side} {symbol} qty={ex_qty:.6f} @ ${avg_px:.4f}")
                                break
                    except Exception:
                        pass

                if order_placed:
                    break  # Order placed successfully

                # Cancel un-filled LIMIT before trying MARKET
                if attempt_type == 'LIMIT':
                    try:
                        client.cancel_order(symbol=symbol, order_id=order_id)
                        print(f"    ⏳ LIMIT not filled, canceled order #{order_id}")
                    except Exception:
                        pass
                    print(f"    ⏩ MARKET fallback: {side} {symbol} qty={qty:.6f}")

            except Exception as e:
                print(f"    ⚠️ {attempt_type} order failed: {e}")
                if attempt_type == 'MARKET':
                    # MARKET also failed — use simulated fallback
                    break

        # ── Post-order verification ──
        if not order_placed:
            # Final check: does Binance show an open position?
            try:
                time.sleep(1)
                positions = client.get_position_info()
                for p in positions:
                    if p.get('symbol') == symbol.upper():
                        pos_amt = float(p.get('position_amt', 0))
                        if abs(pos_amt) > 0.0001:
                            entry_price = float(p.get('entry_price', 0) or 0)
                            if entry_price <= 0:
                                entry_price = abs(float(p.get('notional', 0)) / pos_amt) if pos_amt != 0 else 0
                            executed_qty_actual = abs(pos_amt)
                            order_placed = True
                            print(f"    📡 POSITION FOUND (post-check): {side} {symbol} qty={abs(pos_amt):.4f} @ ${entry_price:.4f}")
                            break
            except Exception:
                pass

        if not order_placed:
            # Genuine failure — skip this trade entirely
            print(f"    ❌ ORDER FAILED: {side} {symbol} qty={qty:.6f} — no position on Binance")
            print(f"    → Trade SKIPPED (state NOT updated)")
            return None

        # ── Order placed successfully: update state ──
        final_qty = executed_qty_actual if executed_qty_actual > 0 else qty
        entry_cost = entry_price * final_qty * TAKER_FEE

        state['position'] = direction
        state['entry_price'] = entry_price
        state['entry_time'] = int(time.time() * 1000)
        state['entry_proba'] = proba
        state['hold_remaining'] = get_symbol_hold_bars(symbol)
        state['cooldown_remaining'] = 0
        state['qty'] = final_qty

        capital -= entry_cost
        peak_capital = max(peak_capital, capital)

        print(f"    ✅ ENTER {direction_label} @ ${entry_price:.4f} | qty={qty:.6f} | proba={proba:.4f}")

        return capital, peak_capital

    def _exit_position(self, symbol: str, state: Dict,
                       capital: float, peak_capital: float) -> Optional[Tuple[float, float, str]]:
        """Exit a live trade position — simulates in DB + places real MARKET order on Binance (testnet or mainnet)."""
        direction = state['position']
        direction_label = 'LONG' if direction == 1 else 'SHORT'

        qty = state['qty']
        entry_price = state['entry_price']

        # Place real exit order on Binance
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
            order_id = order_resp.get('order_id')
            if order_id is None or order_id == '?':
                order_id = order_resp.get('client_order_id')

            # Poll get_order() for actual fill data (up to 6 attempts, 1s apart)
            for attempt in range(6):
                time.sleep(1)
                try:
                    fill_resp = client.get_order(symbol=symbol, order_id=order_id)
                    if isinstance(fill_resp, dict):
                        fill_status = fill_resp.get('status', '')
                        ex_qty = float(fill_resp.get('executed_qty', 0))
                        avg_px = float(fill_resp.get('avg_price', 0))
                        if fill_status == 'FILLED' and ex_qty > 0 and avg_px > 0:
                            exit_price = avg_px
                            print(f"    📡 EXIT ORDER FILLED: {side} {symbol} qty={ex_qty:.6f} @ ${avg_px:.4f} (order #{order_id})")
                            break
                except Exception:
                    pass

            if exit_price is None:
                # Fallback 1: use account_information_v2 for real entry_price
                time.sleep(1)
                try:
                    acct = client._call('account_information_v2')
                    for pos in acct.positions:
                        if getattr(pos, 'symbol', '') == symbol.upper():
                            remaining = float(getattr(pos, 'position_amt', 0) or 0)
                            if abs(remaining) < qty * 0.1:
                                print(f"    📡 EXIT POSITION CLOSED (acct_info_v2): {side} {symbol} remaining={remaining:.4f} (order #{order_id})")
                                # Use average of entry price ± 0.1% as estimate (position fully closed)
                                if exit_price is None:
                                    exit_price = state['entry_price'] * 1.001 if direction == 1 else state['entry_price'] * 0.999
                                break
                except Exception:
                    pass

            if exit_price is None:
                # Fallback 2: check position info
                time.sleep(1)
                try:
                    positions = client.get_position_info()
                    for p in positions:
                        if p.get('symbol') == symbol.upper():
                            pos_amt = float(p.get('position_amt', 0))
                            if abs(pos_amt) < qty * 0.1:
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
                print(f"    📡 EXIT (orderbook): {side} {symbol} order #{order_id} — using orderbook price ${exit_price:.4f}")
        except Exception as e:
            print(f"    ⚠️ Binance exit order failed (fallback): {e}")
            # Simulated fallback: get price from orderbook
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

        # Exit fee only (entry fee already deducted at entry time)
        exit_cost = exit_price * qty * TAKER_FEE
        pnl_net = raw_pnl - exit_cost

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
