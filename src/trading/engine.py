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
    DB_PATH, INITIAL_CAPITAL, LIVE_SYMBOLS,
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, MARGIN_PCT, LEVERAGE_X,
    TAKER_FEE, SLIPPAGE, MODEL_DIR, SEEDS,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
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
        self.report_mode = report
        self._client = None  # lazy-init

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

    def _setup_live_trading(self):
        """Reset stale positions in DB and init leverage for all symbols."""
        net = self.network
        c = self.conn.cursor()

        # Check if we need initial setup (schema_version flag)
        c.execute("SELECT COUNT(*) FROM live_state WHERE position != 0")
        stale = c.fetchone()[0]

        if stale > 0:
            print(f"\n  ⚠️ Found {stale} stale position(s) in DB (from old runs). Resetting...")
            c.execute("UPDATE live_state SET position=0, entry_price=0, entry_time=0, entry_proba=0, hold_remaining=0, cooldown_remaining=0, qty=0 WHERE position != 0")
            self.conn.commit()
            print(f"  ✅ Stale positions reset. New positions will be placed as real orders on Binance {net}.")

        # Init leverage 10x for all symbols
        ok = 0
        for sym in LIVE_SYMBOLS:
            if self._init_leverage(sym):
                ok += 1
        print(f"  ✅ Leverage 10x set for {ok}/{len(LIVE_SYMBOLS)} symbols")

    def _sync_binance_positions(self):
        """Sync DB state with actual open positions on Binance (mainnet or testnet).
        Prevents duplicate position entries when script restarts."""
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
                entry_price = abs(notional / pos_amt) if pos_amt != 0 else 0
                # Update DB state to match Binance
                c.execute("""
                    UPDATE live_state
                    SET position=?, qty=?, entry_price=?, entry_time=?,
                        hold_remaining=?, cooldown_remaining=0
                    WHERE symbol=?
                """, (direction, qty, entry_price, int(time.time() * 1000), 9, sym))
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
            FROM live_trades
        """)
        row = c.fetchone()
        total_trades, wins, total_pnl, gross_profit, gross_loss = row
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Get open positions
        running_positions = []
        c.execute("SELECT symbol, position FROM live_state WHERE position != 0")
        for sym, pos in c.fetchall():
            running_positions.append(f"  {'🟢' if pos==1 else '🔴'} {sym}: {'LONG' if pos==1 else 'SHORT'}")

        msg = (
            f"📊 *Mimia Live Trade Report*\n"
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
            f"`mimia quant • live trading`"
        )

        if len(msg) > 4096:
            msg = msg[:4000] + "\n\n... (truncated)"

        return msg

    def run(self):
        """Execute one full live trading cycle."""
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"  LIVE TRADE RUN — {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} {'(' + self.network.upper() + ')'}")
        print(f"{'='*60}")

        capital, peak_capital = get_capital(self.conn)

        # Initial setup: reset stale positions (from old runs) and init leverage
        self._setup_live_trading()

        # Sync DB state with actual positions on Binance ({self.network})
        # Prevents duplicate entries if trader restarts mid-position
        self._sync_binance_positions()

        states = get_state(self.conn)
        signals_total = 0
        trades_opened = 0
        trades_closed = 0
        trade_report_lines = []

        for i, symbol in enumerate(LIVE_SYMBOLS):
            print(f"\n  [{i+1}/{len(LIVE_SYMBOLS)}] {symbol}...")
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
                gen = self._load_models(symbol)
                sig = gen.generate_signal(symbol)
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

    def _check_enter(self, symbol: str, state: Dict, capital: float,
                     peak_capital: float, proba: float) -> bool:
        """Check if entry conditions are met (threshold + cooldown)."""
        # Already checked in run() — position == 0 and cooldown <= 0
        if proba >= THRESHOLD or proba <= (1 - THRESHOLD):
            return True
        return False

    def _enter_position(self, symbol: str, sig: Dict, state: Dict,
                        capital: float, peak_capital: float) -> Optional[Tuple[float, float]]:
        """Enter a live trade position — simulates in DB + places real MARKET order on Binance (testnet or mainnet)."""
        direction = sig['signal']  # 1=long, -1=short
        proba = sig['proba']
        direction_label = 'LONG' if direction == 1 else 'SHORT'

        entry_price = None
        client = self._get_client()

        # Step 1: Get current price and place real order on Binance
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
            # Place real order on Binance
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
                # Order filled — use avg_price
                entry_price = avg_price_raw
                order_placed = True
                print(f"    📡 ORDER FILLED: {side} {symbol} qty={executed_qty} @ ${avg_price_raw:.4f} (order #{order_id})")
            else:
                # Order response shows NEW/unfilled — Binance has delayed fill updates
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
            print(f"    ⚠️ Binance order failed (fallback): {e}")
            # Simulated fallback: use orderbook price with slippage
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
