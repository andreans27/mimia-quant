#!/usr/bin/env python3
"""
Mimia — Backtest vs Live Trader: Exact Time Range Comparison
=============================================================
Compare backtest simulation vs live trader for an EXACT time range.
Range: 2026-04-27 06:00 UTC → 2026-04-28 06:00 UTC (13:00 WIB → 13:00 WIB)
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.trading.signals import SignalGenerator
from src.trading.state import (
    THRESHOLD, HOLD_BARS, COOLDOWN_BARS, DB_PATH,
    TAKER_FEE, SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL,
)


# ─── TIME RANGE (tz-naive — OHLCV cache index is tz-naive) ────────
START_DT = datetime(2026, 4, 27, 6, 0, 0)   # 13:00 WIB = 06:00 UTC
END_DT   = datetime(2026, 4, 28, 6, 0, 0)   # 13:00 WIB = 06:00 UTC
START_MS = int(START_DT.timestamp() * 1000)
END_MS   = int(END_DT.timestamp() * 1000)


# ─── BACKTEST SIMULATION ────────────────────────────────────────────

def reconstruct_initial_state(symbol: str) -> dict:
    """Reconstruct the live trader's state at START_DT from trade history.
    
    A trade opened at entry_time with hold_bars remains open until
    entry_time + hold_bars * 5min. If START_DT falls within that window,
    the trade is still open.
    """
    conn = sqlite3.connect(str(DB_PATH))
    
    # Find trades that were OPEN at START_DT
    # (entry_time < START_MS AND exit_time >= START_MS OR exit_time IS NULL)
    # Or: trades with no exit_time (still open) or exit_time after start
    c = conn.execute('''
        SELECT direction, entry_time, exit_time, entry_price, qty, entry_proba, hold_bars
        FROM live_trades
        WHERE symbol=? AND entry_time < ? AND (exit_time IS NULL OR exit_time >= ?)
        ORDER BY entry_time DESC LIMIT 1
    ''', (symbol, START_MS, START_MS))
    
    row = c.fetchone()
    conn.close()
    
    if row is None:
        return {'position': 0, 'hold_rem': 0, 'cooldown': 0, 'entry_price': 0, 'qty': 0}
    
    direction, entry_ms, exit_ms, entry_p, qty, proba, hold_b = row
    entry_ms = entry_ms or 0
    
    # Calculate hold remaining at START_DT
    elapsed_ms = START_MS - entry_ms
    elapsed_bars = elapsed_ms // 300_000  # 5 min per bar
    hold_rem = max(0, int(hold_b or HOLD_BARS) - elapsed_bars)
    
    position = 1 if direction == 'long' else -1
    
    print(f"    🔄 Initial state: {'LONG' if position==1 else 'SHORT'} @ ${float(entry_p):.4f} "
          f"| hold_rem={hold_rem} bars | entered {elapsed_bars} bars ago")
    
    return {
        'position': position,
        'hold_rem': hold_rem,
        'cooldown': 0,
        'entry_price': float(entry_p or 0),
        'entry_proba': float(proba or 0),
        'qty': float(qty or 0),
    }


def simulate_bt(symbol: str, initial_state: dict = None) -> dict:
    """
    Run bar-by-bar backtest for the EXACT time range.
    Uses SignalGenerator._load_models (same code path as live trader).
    """
    gen = SignalGenerator(symbol)
    
    # Load models + features
    cached = gen._load_models(symbol)
    if cached is None:
        raise ValueError(f"Cannot load models for {symbol}")
    
    feat_df = cached['features']
    groups = cached['groups']
    
    # Filter features to match OHLCV range (need extra for warmup)
    feat_df = feat_df[feat_df.index >= START_DT - timedelta(hours=6)].copy()
    
    # Compute proba for all bars
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
    probas = pd.Series(np.mean(prob_stack, axis=1), index=feat_df.index)
    
    # Fetch OHLCV prices
    spot = symbol[4:] if symbol.startswith("1000") else symbol
    df_p = gen._ensure_ohlcv_data(spot)
    if df_p is None:
        return None
    
    # Align proba with prices
    df_aligned = df_p.join(probas.to_frame('proba'), how='inner')
    # Filter to exact range
    df_aligned = df_aligned[(df_aligned.index >= START_DT) & (df_aligned.index < END_DT)].copy()
    
    if len(df_aligned) < 10:
        return None
    
    # ── TRADING SIMULATION ──
    # Matches live trader's batch approach:
    #   Phase 1: ALL signals computed using bar close data (pre-computed probas)
    #   Phase 2: Execute trades ~2 min after bar close (simulated by using
    #            NEXT bar's price for entry, not current bar's close)
    capital = INITIAL_CAPITAL
    if initial_state:
        position = initial_state.get('position', 0)
        hold_rem = initial_state.get('hold_rem', 0)
        cooldown = initial_state.get('cooldown', 0)
        entry_price = initial_state.get('entry_price', 0.0)
        entry_proba = initial_state.get('entry_proba', 0.0)
        entry_qty = initial_state.get('qty', 0.0)
        entry_time = None
        if position != 0:
            print(f"    🔄 Simulation starting with existing {'LONG' if position==1 else 'SHORT'} position (hold_rem={hold_rem})")
    else:
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
    wins = 0
    losses = 0
    pending_signal = 0
    pending_proba = 0.0
    
    for idx, row in df_aligned.iterrows():
        price = float(row['close'])
        proba = float(row['proba'])
        
        hold_rem = max(0, hold_rem - 1)
        cooldown = max(0, cooldown - 1)
        
        # EXIT
        if position != 0 and hold_rem <= 0:
            if position == 1:
                ex_price = price * (1 - SLIPPAGE)
                raw = entry_qty * (ex_price - entry_price)
            else:
                ex_price = price * (1 + SLIPPAGE)
                raw = entry_qty * (entry_price - ex_price)
            
            ex_cost = ex_price * entry_qty * TAKER_FEE
            pnl = raw - ex_cost
            pnl_pct = pnl / (entry_price * entry_qty) * 100 if entry_price * entry_qty > 0 else 0
            
            capital += raw
            capital -= ex_cost
            
            if position == 1: long_pnl += pnl
            else: short_pnl += pnl
            if pnl > 0: wins += 1
            else: losses += 1
            
            trades.append({
                'direction': 'long' if position == 1 else 'short',
                'entry_time': entry_time,
                'exit_time': idx,
                'entry_price': entry_price,
                'exit_price': ex_price,
                'qty': entry_qty,
                'pnl_net': pnl,
                'pnl_pct': pnl_pct,
                'entry_proba': entry_proba,
            })
            
            position = 0
            cooldown = COOLDOWN_BARS

        # ENTRY (deferred: signal dari candle N, execute di candle N+1 close)
        # Ini mencerminkan realita: compute butuh 2 menit → order terisi di candle berikutnya
        if pending_signal != 0 and position == 0 and cooldown <= 0:
            sig = pending_signal
            pending_signal = 0
            if sig == 1:
                entry_price = price * (1 + SLIPPAGE)
            else:
                entry_price = price * (1 - SLIPPAGE)
            entry_qty = (capital * POSITION_PCT) / entry_price
            capital -= entry_price * entry_qty * TAKER_FEE
            position = sig
            entry_time = idx
            entry_proba = pending_proba
            hold_rem = HOLD_BARS

        # Evaluate signal untuk candle ini (akan dieksekusi di candle berikutnya)
        sig = 0
        if proba >= THRESHOLD: sig = 1
        elif proba <= (1 - THRESHOLD): sig = -1
        pending_signal = sig
        pending_proba = proba

    n = len(trades)
    wr = wins / n * 100 if n else 0
    gp = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gl = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'total_trades': n,
        'wins': wins, 'losses': losses,
        'win_rate': wr, 'profit_factor': pf,
        'total_pnl': capital - INITIAL_CAPITAL,
        'long_pnl': long_pnl, 'short_pnl': short_pnl,
        'avg_entry': np.mean([t['entry_price'] for t in trades]) if trades else 0,
        'avg_exit': np.mean([t['exit_price'] for t in trades]) if trades else 0,
        'trades': trades,
    }


# ─── LIVE TRADER DATA ───────────────────────────────────────────────

def get_live(symbol: str) -> dict:
    """Get live trader trades in exact time range."""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.execute('''
        SELECT direction, entry_time, exit_time, entry_price, exit_price,
               qty, pnl_net, pnl_pct, entry_proba
        FROM live_trades
        WHERE symbol=? AND entry_time>=? AND entry_time<?
          AND exit_reason='hold_expiry'
        ORDER BY entry_time
    ''', (symbol, START_MS, END_MS))
    
    trades = []
    total = 0.0; lp = 0.0; sp = 0.0
    wins = 0; eps = []; xps = []
    
    for r in c.fetchall():
        d, et_ms, xt_ms, ep, xp, _, pnl, pct, proba = r
        pnl = float(pnl or 0)
        total += pnl
        eps.append(float(ep))
        xps.append(float(xp or 0))
        if d == 'long': lp += pnl
        else: sp += pnl
        if pnl > 0: wins += 1
        trades.append({
            'direction': d,
            'entry_price': float(ep), 'exit_price': float(xp or 0),
            'entry_time': datetime.utcfromtimestamp(et_ms/1000) if et_ms else None,
            'exit_time': datetime.utcfromtimestamp(xt_ms/1000) if xt_ms else None,
            'pnl_net': pnl, 'pnl_pct': float(pct or 0),
            'entry_proba': float(proba or 0),
        })
    
    conn.close()
    n = len(trades)
    wr = wins / n * 100 if n else 0
    gp = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gl = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    
    return {
        'total_trades': n,
        'wins': wins, 'losses': n - wins,
        'win_rate': wr, 'profit_factor': pf,
        'total_pnl': total, 'long_pnl': lp, 'short_pnl': sp,
        'avg_entry': np.mean(eps) if eps else 0,
        'avg_exit': np.mean(xps) if xps else 0,
        'trades': trades,
    }


# ─── COMPARISON ─────────────────────────────────────────────────────

def compare_symbol(symbol: str):
    """Full comparison for one symbol."""
    initial_state = reconstruct_initial_state(symbol)
    bt = simulate_bt(symbol, initial_state=initial_state)
    live = get_live(symbol)
    
    if bt is None:
        return None
    
    pnl_icon = lambda p: '🟢' if p > 0 else '🔴'
    
    print(f"\n  {'─'*65}")
    print(f"  {symbol}")
    print(f"  {'─'*65}")
    
    rows = [
        ('Trades',   f"{bt['total_trades']:3d}", f"{live['total_trades']:3d}"),
        ('Win Rate', f"{bt['win_rate']:5.1f}%",  f"{live['win_rate']:5.1f}%"),
        ('PF',       f"{bt['profit_factor']:6.2f}", f"{live['profit_factor']:6.2f}"),
        ('', '', ''),
        ('Total PnL', f"{pnl_icon(bt['total_pnl'])} ${bt['total_pnl']:+7.2f}", f"{pnl_icon(live['total_pnl'])} ${live['total_pnl']:+7.2f}"),
        ('Long PnL',  f"${bt['long_pnl']:+8.2f}", f"${live['long_pnl']:+8.2f}"),
        ('Short PnL', f"${bt['short_pnl']:+8.2f}", f"${live['short_pnl']:+8.2f}"),
        ('', '', ''),
        ('Avg Entry', f"${bt['avg_entry']:8.4f}", f"${live['avg_entry']:8.4f}"),
        ('Avg Exit',  f"${bt['avg_exit']:8.4f}", f"${live['avg_exit']:8.4f}"),
    ]
    
    print(f"  {'Metric':12s} {'Backtest':>18s} {'Live':>18s} {'Match?':>8s}")
    print(f"  {'─'*58}")
    
    for name, bv, lv in rows:
        if not name:
            print()
            continue
        
        # Auto-detect match
        match = True
        if 'PnL' in name or 'Total' in name:
            diff = abs(bt['total_pnl'] - live['total_pnl'])
            match = diff < 5.0
        elif 'Long' in name:
            diff = abs(bt['long_pnl'] - live['long_pnl'])
            match = diff < 3.0
        elif 'Short' in name:
            diff = abs(bt['short_pnl'] - live['short_pnl'])
            match = diff < 3.0
        elif 'Trades' == name.strip():
            match = abs(bt['total_trades'] - live['total_trades']) <= 2
        elif 'Entry' in name or 'Exit' in name:
            bp = bt['avg_entry'] if 'Entry' in name else bt['avg_exit']
            lp = live['avg_entry'] if 'Entry' in name else live['avg_exit']
            if bp > 0:
                match = abs(bp - lp) / bp * 100 < 0.5
        
        icon = '✅' if match else '⚠️'
        print(f"  {name:12s} {bv:>18s} {lv:>18s} {icon:>8s}")
    
    return {
        'symbol': symbol,
        'bt': bt,
        'live': live,
        'pnl_diff': abs(bt['total_pnl'] - live['total_pnl']),
        'trade_diff': bt['total_trades'] - live['total_trades'],
    }


# ─── MAIN ───────────────────────────────────────────────────────────

import pandas as pd

def main():
    # Auto-detect symbols with live trades in range
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.execute('''
        SELECT symbol, COUNT(*) FROM live_trades
        WHERE entry_time>=? AND entry_time<? AND exit_reason='hold_expiry'
        GROUP BY symbol HAVING COUNT(*)>=3 ORDER BY COUNT(*) DESC
    ''', (START_MS, END_MS))
    symbols = [r[0] for r in c.fetchall()]
    conn.close()
    
    period_str = f"{START_DT.strftime('%Y-%m-%d %H:%M')} → {END_DT.strftime('%Y-%m-%d %H:%M')} UTC"
    
    print(f"{'='*70}")
    print(f"  BACKTEST vs LIVE TRADER COMPARISON")
    print(f"  Period: {period_str}")
    print(f"  Symbols: {len(symbols)} — {', '.join(symbols[:5])}...")
    print(f"{'='*70}")
    
    results = []
    for sym in symbols:
        try:
            r = compare_symbol(sym)
            if r: results.append(r)
        except Exception as e:
            print(f"\n  ❌ {sym}: {e}")
    
    # ── TOTAL ──
    if results:
        print(f"\n{'='*70}")
        print(f"  AGGREGATE TOTALS")
        print(f"{'='*70}")
        print(f"  {'Symbol':12s} {'BT T':5s} {'Live T':6s} {'ΔT':4s} {'BT PnL':>11s} {'Live PnL':>11s} {'ΔPnL':>8s} {'Match?':6s}")
        print(f"  {'─'*66}")
        
        tot_bt = 0; tot_live = 0; tot_bt_pnl = 0.0; tot_live_pnl = 0.0
        total_pnl_diff = 0.0
        good_symbols = 0
        
        for r in results:
            s = r['symbol']; b = r['bt']; l = r['live']
            tot_bt += b['total_trades']
            tot_live += l['total_trades']
            tot_bt_pnl += b['total_pnl']
            tot_live_pnl += l['total_pnl']
            pnl_diff = abs(b['total_pnl'] - l['total_pnl'])
            total_pnl_diff += pnl_diff
            match = '✅' if pnl_diff < 5 else '⚠️'
            if match == '✅': good_symbols += 1
            t_diff = b['total_trades'] - l['total_trades']
            print(f"  {s:12s} {b['total_trades']:>5d} {l['total_trades']:>6d} {t_diff:>+3d} "
                  f"{'🟢' if b['total_pnl']>0 else '🔴'} ${b['total_pnl']:>+7.2f} "
                  f"{'🟢' if l['total_pnl']>0 else '🔴'} ${l['total_pnl']:>+7.2f} "
                  f"${pnl_diff:>6.2f} {match:6s}")
        
        print(f"  {'─'*66}")
        print(f"  {'TOTAL':12s} {tot_bt:>5d} {tot_live:>6d} {tot_bt-tot_live:>+3d} "
              f"{'🟢' if tot_bt_pnl>0 else '🔴'} ${tot_bt_pnl:>+7.2f} "
              f"{'🟢' if tot_live_pnl>0 else '🔴'} ${tot_live_pnl:>+7.2f} "
              f"${total_pnl_diff:>6.2f}")
        
        print(f"\n  Symbols dengan PnL inline (diff < $5): {good_symbols}/{len(results)}")
        
        # Trade-by-trade analysis untuk symbols dengan match tertinggi
        print(f"\n{'='*70}")
        print(f"  TRADE-LEVEL ANALYSIS")
        print(f"{'='*70}")
        
        # Cari symbol dengan trade count paling dekat
        best = min(results, key=lambda r: abs(r['trade_diff']))
        print(f"\n  Best trade count match: {best['symbol']} (BT={best['bt']['total_trades']}, Live={best['live']['total_trades']})")
        print(f"  Tapi PnL diff = ${best['pnl_diff']:.2f}")
        
        # Cari symbol dengan PnL paling dekat
        best_pnl = min(results, key=lambda r: r['pnl_diff'])
        print(f"  Best PnL match: {best_pnl['symbol']} (BT=${best_pnl['bt']['total_pnl']:.2f}, Live=${best_pnl['live']['total_pnl']:.2f})")
        print(f"  Diff = ${best_pnl['pnl_diff']:.2f}")
        
        # Trade-level perbandingan untuk best PnL symbol
        sym = best_pnl['symbol']
        bt_trades = best_pnl['bt']['trades']
        live_trades = best_pnl['live']['trades']
        
        if bt_trades and live_trades:
            print(f"\n  ── {sym} Trade Detail ──")
            print(f"  {'BT Entry':10s} {'BT Dir':6s} {'BT PnL':8s} {'Live Entry':10s} {'Live Dir':6s} {'Live PnL':8s}")
            print(f"  {'─'*50}")
            
            # Match trades by proximity
            import bisect
            bt_sorted = sorted([(t['entry_time'], i, t) for i, t in enumerate(bt_trades)])
            live_used = [False] * len(live_trades)
            
            for bt_ts, bi, bt_t in bt_sorted:
                bt_ts_ms = bt_ts.timestamp() * 1000 if hasattr(bt_ts, 'timestamp') else 0
                best_li = None
                best_diff = 15  # 15 min window
                for li, lt in enumerate(live_trades):
                    if live_used[li]: continue
                    lt_ts = lt['entry_time']
                    if lt_ts is None: continue
                    lt_ts_ms = lt_ts.timestamp() * 1000 if hasattr(lt_ts, 'timestamp') else 0
                    diff = abs(bt_ts_ms - lt_ts_ms) / 60000  # convert to minutes
                    if diff < best_diff:
                        best_diff = diff
                        best_li = li
                
                bt_dir = 'LON' if bt_t['direction'] == 'long' else 'SHT'
                
                if best_li is not None:
                    live_used[best_li] = True
                    lt = live_trades[best_li]
                    live_dir = 'LON' if lt['direction'] == 'long' else 'SHT'
                    dir_match = '✅' if bt_t['direction'] == lt['direction'] else '❌'
                    bt_time_str = bt_ts.strftime('%H:%M') if hasattr(bt_ts, 'strftime') else str(bt_ts)
                    lt_time_str = lt_ts.strftime('%H:%M') if hasattr(lt_ts, 'strftime') else str(lt_ts)
                    print(f"  {bt_time_str:10s} {bt_dir:6s} ${bt_t['pnl_net']:>+6.2f} "
                          f"{lt_time_str:10s} {live_dir:6s} ${lt['pnl_net']:>+6.2f}  {dir_match}")
                else:
                    bt_time_str = bt_ts.strftime('%H:%M') if hasattr(bt_ts, 'strftime') else str(bt_ts)
                    print(f"  {bt_time_str:10s} {bt_dir:6s} ${bt_t['pnl_net']:>+6.2f} "
                          f"{'─':>10s} {'─':>6s} {'─':>8s}  ❌ (no match)")
    
    print()


if __name__ == '__main__':
    main()
