#!/usr/bin/env python3
"""
Mimia — Comprehensive PnL & Trade Metrics Comparison
=====================================================
Compares backtest simulation vs live trader on PnL, Win Rate, 
Profit Factor, entry/exit prices for the past N hours.

Usage:
    python scripts/compare_pnl_metrics.py --symbol ETHUSDT --hours 24
    python scripts/compare_pnl_metrics.py --all --hours 24
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import argparse
import sqlite3
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.trading.signals import SignalGenerator
from src.trading.state import (
    LIVE_SYMBOLS, THRESHOLD, HOLD_BARS, COOLDOWN_BARS,
    DB_PATH, TAKER_FEE, SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL,
)
from src.strategies.ml_features import OHLCV_CACHE_DIR


def compute_probas(symbol: str, hours: int = 36) -> pd.Series:
    """Compute model proba for every bar in the time window."""
    gen = SignalGenerator(symbol)
    spot = symbol[4:] if symbol.startswith("1000") else symbol
    
    cached = gen._load_models(symbol)
    if cached is None:
        raise ValueError(f"Cannot load models for {symbol}")
    
    feat_df = cached['features']
    groups = cached['groups']
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    feat_df = feat_df[feat_df.index >= cutoff].copy()
    
    group_probs = {}
    for tf, models in groups.items():
        tf_p = []
        for _, m, mf in models:
            avail = [c for c in mf if c in feat_df.columns]
            if len(avail) < 5: continue
            X = feat_df[avail].fillna(0).clip(-10, 10).values
            tf_p.append(m.predict_proba(X)[:, 1])
        if tf_p:
            group_probs[tf] = np.nanmean(tf_p, axis=0)
    
    prob_stack = np.column_stack([group_probs[tf] for tf in group_probs])
    return pd.Series(np.mean(prob_stack, axis=1), index=feat_df.index)


def simulate_bt(symbol: str, probas: pd.Series, prices: pd.DataFrame,
                hours: int = 24) -> dict:
    """
    Bar-by-bar backtest simulation with full PnL calculation.
    Same logic as live trader: level-based entry, hold for N bars, cooldown.
    """
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    capital = INITIAL_CAPITAL
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    entry_time = None
    entry_proba = 0.0
    hold_rem = 0
    cooldown = 0
    
    trades = []
    long_pnl = 0.0
    short_pnl = 0.0
    
    # Align prices to proba timestamps
    aligned = prices.join(probas.to_frame('proba'), how='inner')
    aligned = aligned[aligned.index >= cutoff].copy()
    
    for idx, row in aligned.iterrows():
        price = float(row['close'])
        proba = float(row['proba'])
        
        hold_rem = max(0, hold_rem - 1)
        cooldown = max(0, cooldown - 1)
        
        # Exit
        if position != 0 and hold_rem <= 0:
            if position == 1:
                ex_price = price * (1 - SLIPPAGE)
                raw = entry_qty * (ex_price - entry_price)
            else:
                ex_price = price * (1 + SLIPPAGE)
                raw = entry_qty * (entry_price - ex_price)
            
            ex_cost = ex_price * entry_qty * TAKER_FEE
            pnl = raw - ex_cost
            pnl_pct = pnl / (entry_price * entry_qty) * 100
            
            capital += raw
            capital -= ex_cost
            
            if position == 1: long_pnl += pnl
            else: short_pnl += pnl
            
            trades.append({
                'direction': 'long' if position == 1 else 'short',
                'entry_time': entry_time.timestamp() * 1000 if entry_time else 0,
                'exit_time': idx.timestamp() * 1000,
                'entry_price': entry_price,
                'exit_price': ex_price,
                'qty': entry_qty,
                'pnl_net': pnl,
                'pnl_pct': pnl_pct,
                'entry_proba': entry_proba,
                'hold_bars': HOLD_BARS,
            })
            
            position = 0
            cooldown = COOLDOWN_BARS
        
        # Entry
        if position == 0 and cooldown <= 0:
            sig = 0
            if proba >= THRESHOLD: sig = 1
            elif proba <= (1 - THRESHOLD): sig = -1
            
            if sig != 0:
                if sig == 1:
                    entry_price = price * (1 + SLIPPAGE)
                else:
                    entry_price = price * (1 - SLIPPAGE)
                
                entry_qty = (capital * POSITION_PCT) / entry_price
                capital -= entry_price * entry_qty * TAKER_FEE
                
                position = sig
                entry_time = idx
                entry_proba = proba
                hold_rem = HOLD_BARS
    
    # Calculate metrics
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    losses = n_trades - wins
    wr = wins / n_trades * 100 if n_trades else 0
    gp = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gl = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    total_pnl = capital - INITIAL_CAPITAL
    
    return {
        'total_trades': n_trades,
        'wins': wins, 'losses': losses,
        'win_rate': wr, 'profit_factor': pf,
        'total_pnl': total_pnl, 'return_pct': total_pnl / INITIAL_CAPITAL * 100,
        'long_pnl': long_pnl, 'short_pnl': short_pnl,
        'avg_entry': np.mean([t['entry_price'] for t in trades]) if trades else 0,
        'avg_exit': np.mean([t['exit_price'] for t in trades]) if trades else 0,
        'trades': trades,
    }


def get_live(symbol: str, hours: int = 24) -> dict:
    """Extract model-driven live trader metrics."""
    cutoff_ms = int((time.time() - hours * 3600) * 1000)
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.execute('''
        SELECT direction, entry_time, entry_price, exit_price, pnl_net, pnl_pct, entry_proba
        FROM live_trades
        WHERE symbol=? AND entry_time>=? AND exit_reason='hold_expiry' AND entry_proba!=0.5
    ''', (symbol, cutoff_ms))
    
    trades = []
    total = 0.0; long_pnl = 0.0; short_pnl = 0.0
    wins = 0; eprices = []; xprices = []
    
    for r in c.fetchall():
        d, _, ep, xp, pnl, pct, proba = r
        pnl = float(pnl or 0)
        total += pnl
        eprices.append(float(ep))
        xprices.append(float(xp or 0))
        if d == 'long': long_pnl += pnl
        else: short_pnl += pnl
        if pnl > 0: wins += 1
        trades.append({
            'direction': d, 'entry_price': float(ep), 'exit_price': float(xp or 0),
            'pnl_net': pnl, 'pnl_pct': float(pct or 0), 'entry_proba': float(proba or 0),
        })
    
    conn.close()
    n = len(trades)
    wr = wins / n * 100 if n else 0
    gp = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gl = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'total_trades': n, 'wins': wins, 'losses': n - wins,
        'win_rate': wr, 'profit_factor': pf,
        'total_pnl': total, 'return_pct': total / INITIAL_CAPITAL * 100,
        'long_pnl': long_pnl, 'short_pnl': short_pnl,
        'avg_entry': np.mean(eprices) if eprices else 0,
        'avg_exit': np.mean(xprices) if xprices else 0,
        'trades': trades,
    }


def print_comparison(symbol: str, bt: dict, live: dict):
    """Pretty-print side-by-side comparison."""
    def pnl_icon(pnl):
        return '🟢' if pnl > 0 else '🔴'
    
    print(f"\n  {'─'*65}")
    print(f"  📊 {symbol}")
    print(f"  {'─'*65}")
    
    rows = [
        ('Total Trades',  str(bt['total_trades']), str(live['total_trades']), False),
        ('Win Rate',      f"{bt['win_rate']:.1f}%", f"{live['win_rate']:.1f}%", False),
        ('Profit Factor', f"{bt['profit_factor']:.2f}", f"{live['profit_factor']:.2f}", False),
        ('', '', '', False),
        ('Total PnL',     f"{pnl_icon(bt['total_pnl'])} ${bt['total_pnl']:+.2f}", 
                         f"{pnl_icon(live['total_pnl'])} ${live['total_pnl']:+.2f}", True),
        ('Return %',      f"{bt['return_pct']:+.2f}%", f"{live['return_pct']:+.2f}%", False),
        ('Long PnL',      f"${bt['long_pnl']:+.2f}", f"${live['long_pnl']:+.2f}", True),
        ('Short PnL',     f"${bt['short_pnl']:+.2f}", f"${live['short_pnl']:+.2f}", True),
        ('', '', '', False),
        ('Avg Entry',     f"${bt['avg_entry']:.4f}", f"${live['avg_entry']:.4f}", True),
        ('Avg Exit',      f"${bt['avg_exit']:.4f}", f"${live['avg_exit']:.4f}", True),
    ]
    
    print(f"  {'Metric':20s} {'Backtest':>18s} {'Live':>18s} {'Match':>8s}")
    print(f"  {'─'*66}")
    
    for name, bv, lv, should_check in rows:
        if not name:
            print()
            continue
        
        if should_check:
            if 'PnL' in name or 'Total' in name:
                diff = abs(bt['total_pnl'] - live['total_pnl'])
                match = diff < 5.0
            elif 'Long' in name:
                diff = abs(bt['long_pnl'] - live['long_pnl'])
                match = diff < 3.0
            elif 'Short' in name:
                diff = abs(bt['short_pnl'] - live['short_pnl'])
                match = diff < 3.0
            elif 'Entry' in name or 'Exit' in name:
                # Price difference as % of price
                bp = bt['avg_entry'] if 'Entry' in name else bt['avg_exit']
                lp = live['avg_entry'] if 'Entry' in name else live['avg_exit']
                if bp > 0:
                    diff_pct = abs(bp - lp) / bp * 100
                    match = diff_pct < 0.5  # within 0.5%
                else:
                    match = True
            else:
                match = True
        else:
            match = True
        
        icon = '✅' if match else '⚠️'
        print(f"  {name:20s} {bv:>18s} {lv:>18s} {icon:>8s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', '-s', default=None)
    parser.add_argument('--hours', type=int, default=24)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    hours = args.hours
    
    if args.all:
        symbols = LIVE_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol]
    else:
        conn = sqlite3.connect(str(DB_PATH))
        cutoff_ms = int((time.time() - hours * 3600) * 1000)
        c = conn.execute('''
            SELECT symbol, COUNT(*) FROM live_trades
            WHERE entry_time>=? AND exit_reason='hold_expiry' AND entry_proba!=0.5
            GROUP BY symbol HAVING COUNT(*)>=3 ORDER BY COUNT(*) DESC
        ''', (cutoff_ms,))
        symbols = [r[0] for r in c.fetchall()]
        conn.close()
    
    print(f"{'='*70}")
    print(f"  PnL & METRICS COMPARISON — Last {hours}h")
    print(f"  Symbols: {len(symbols)} — {', '.join(symbols[:6])}{'...' if len(symbols)>6 else ''}")
    print(f"{'='*70}")
    
    results = []
    for sym in symbols:
        try:
            probas = compute_probas(sym, hours=hours + 12)
            if probas is None: 
                print(f"\n  ⚠️ {sym}: No proba data"); continue
            
            # Fetch OHLCV prices
            gen = SignalGenerator(sym)
            spot = sym[4:] if sym.startswith("1000") else sym
            df_p = gen._ensure_ohlcv_data(spot)
            if df_p is None:
                print(f"\n  ⚠️ {sym}: No price data"); continue
            
            bt = simulate_bt(sym, probas, df_p, hours=hours)
            live = get_live(sym, hours=hours)
            
            print_comparison(sym, bt, live)
            results.append((sym, bt, live))
        except Exception as e:
            print(f"\n  ❌ {sym}: {e}")
    
    # Summary
    if results:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Symbol':12s} {'BT T':6s} {'Live T':7s} {'BT PnL':10s} {'Live PnL':10s} {'BT WR':7s} {'Live WR':7s}")
        print(f"  {'─'*62}")
        
        tot_bt_pnl = 0; tot_live_pnl = 0
        tot_bt_t = 0; tot_live_t = 0
        
        for sym, bt, live in results:
            tot_bt_pnl += bt['total_pnl']
            tot_live_pnl += live['total_pnl']
            tot_bt_t += bt['total_trades']
            tot_live_t += live['total_trades']
            print(f"  {sym:12s} {bt['total_trades']:>6d} {live['total_trades']:>7d} "
                  f"${bt['total_pnl']:>+7.2f}  ${live['total_pnl']:>+7.2f}  "
                  f"{bt['win_rate']:>5.1f}% {live['win_rate']:>5.1f}%")
        
        print(f"  {'─'*62}")
        pnl_diff = abs(tot_bt_pnl - tot_live_pnl)
        print(f"  {'TOTAL':12s} {tot_bt_t:>6d} {tot_live_t:>7d} "
              f"${tot_bt_pnl:>+7.2f}  ${tot_live_pnl:>+7.2f}  "
              f"| PnL diff: ${pnl_diff:.2f}")
        print(f"  {'':12s} {'─'*50}")
        if pnl_diff < 20:
            print(f"  ✅ PnL inline — diff only ${pnl_diff:.2f}")
        else:
            print(f"  ⚠️ PnL differs by ${pnl_diff:.2f}")
    
    print()


if __name__ == '__main__':
    main()
