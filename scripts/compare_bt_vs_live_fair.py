#!/usr/bin/env python3
"""Compare live vs backtest for trades before 00:00 UTC (all 1h bars complete)."""
import sys; sys.path.insert(0, '.')
import sqlite3, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.signals import SignalGenerator as SG
from src.trading.state import LIVE_SYMBOLS

THR = 0.50; COOLDOWN = 2; HOLD = 10
def sizing_fn(p):
    if p>=0.80: return 0.35
    if p>=0.75: return 0.28
    if p>=0.70: return 0.22
    if p>=0.65: return 0.18
    return 0.15

# LIVE TRADES window 18:00-00:00
db = sqlite3.connect('data/live_trading.db')
start_ms = int(datetime(2026, 4, 29, 18, 0).timestamp() * 1000)
end_ms = int(datetime(2026, 4, 30, 0, 0).timestamp() * 1000)
live_trades = db.execute("""
    SELECT id, symbol, direction, entry_time, entry_price, pnl_net, entry_proba
    FROM live_trades WHERE entry_time >= ? AND entry_time < ? AND exit_reason != 'history_sync'
    ORDER BY entry_time
""", (start_ms, end_ms)).fetchall()
db.close()

print(f"=== LIVE TRADES (18:00-00:00) ===")
print(f"Total: {len(live_trades)}")
live_wins = sum(1 for t in live_trades if t[5] > 0)
live_pnl = sum(t[5] for t in live_trades)
print(f"WR: {live_wins}/{len(live_trades)} ({live_wins/len(live_trades)*100:.1f}%) PnL: ${live_pnl:.2f}")
for t in live_trades:
    d = 'LONG' if t[2]==1 else 'SHORT'
    et = datetime.fromtimestamp(t[3]/1000).strftime('%H:%M')
    print(f"  #{t[0]} {t[1]:<12} {d} ${t[4]:<.4f} PnL=${t[5]:+.2f} proba={t[6]:.4f} [{et}]")

# BACKTEST with truncated data (only up to 00:00)
print(f"\n=== BACKTEST (truncated to 00:00) ===")
cutoff = datetime(2026, 4, 30, 0, 0)

def bt_symbol(sym, df_5m):
    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
    gen = SG(sym)
    cached = gen._load_models(sym)
    if cached is None: return []
    groups = cached['groups']
    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values
    feat_index = feat_df.index
    all_feats = sorted(set(f for _,ms in groups.items() for _,m,mf in ms for f in mf))
    mf_idx = {f:i for i,f in enumerate(all_feats)}
    fc_map = {f:i for i,f in enumerate(feat_df.columns)}
    long_r, short_r = [], []
    for tg, ms in groups.items():
        for _,m,mf in ms:
            avail = [mf_idx[f] for f in mf if f in mf_idx]
            if len(avail)>=5:
                ref = (m, np.array(avail, np.int32))
                if tg=='long': long_r.append(ref)
                elif tg=='short': short_r.append(ref)
    start_dt = datetime(2026, 4, 29, 18, 0)
    ts = max(200, int(np.searchsorted(feat_index, start_dt)))
    tr = list(range(ts, len(feat_index)))
    n = len(tr)
    if n == 0: return []
    feat_np = feat_df.values
    mat = np.zeros((n, len(all_feats)), np.float64)
    for name, pos in mf_idx.items():
        if name in fc_map:
            mat[:,pos] = np.clip(np.nan_to_num(feat_np[ts:ts+n, fc_map[name]], nan=0.0), -10, 10)
    def batch(refs):
        p = np.zeros(n); nv = np.zeros(n, np.int32)
        for m, fi in refs:
            try:
                preds = m.predict_proba(mat[:,fi])[:,1]
                p += preds; nv += 1
            except: pass
        return np.where(nv>0, p/nv, 0.5)
    def cal(probas, side):
        cp = Path('data/ml_models') / f'{sym}_{side}_calibrator.json'
        if cp.exists():
            try:
                with open(cp) as f: c = json.load(f)
                z = c['coef'] * probas + c['intercept']
                return np.clip(1.0/(1.0+np.exp(-z)), 0.0, 1.0)
            except: pass
        return probas
    long_p = cal(batch(long_r), 'long')
    short_p = cal(batch(short_r), 'short')
    sigs = np.zeros(n, np.int32)
    lm = long_p >= THR; sm = short_p >= THR
    both = lm & sm; pick = long_p >= short_p
    sigs[both & pick]=1;sigs[both & ~pick]=-1
    sigs[lm & ~sm]=1;sigs[sm & ~lm]=-1
    probas = np.where(sigs==1, long_p, np.where(sigs==-1, short_p, np.maximum(long_p, short_p)))
    cap=5000.0; peak=5000.0
    pos=0;ep=0.0;eq=0.0;hr=0;cool=0;eproba=0.0;ps=0;pproba=0.0;entry_ts=None;epb=0.0
    trades=[]
    for ii,bi in enumerate(tr):
        try: price=float(close_arr[bi])
        except: continue
        bar_ts=feat_index[bi]
        hr=max(0,hr-1);cool=max(0,cool-1)
        if pos!=0 and hr<=0:
            xp=price*0.9995 if pos==1 else price*1.0005
            raw=eq*(xp-ep) if pos==1 else eq*(ep-xp)
            net=raw-xp*eq*0.0004
            cap+=net;peak=max(peak,cap)
            trades.append({'symbol':sym,'dir':'LONG' if pos==1 else 'SHORT','ep':eproba,'pnl':net,'et':entry_ts.isoformat() if entry_ts else None,'xt':bar_ts.isoformat()})
            pos=0;cool=COOLDOWN
        if ps!=0 and pos==0 and cool<=0:
            slip=1.0005 if ps==1 else 0.9995
            ep=price*slip
            pct=sizing_fn(pproba)
            eq=(cap*pct)/ep
            cap-=ep*eq*0.0004
            peak=max(peak,cap)
            pos=ps;eproba=pproba;hr=HOLD;entry_ts=bar_ts;epb=ep
        ps=sigs[ii];pproba=probas[ii]
    return trades

# Run backtest only for symbols with live trades
live_symbols = list(set(t[1] for t in live_trades))
all_bt = []
for sym in live_symbols:
    gen = SG(sym)
    df = gen._ensure_ohlcv_data(sym)
    df_trunc = df[df.index <= cutoff].copy()
    bt_trades = bt_symbol(sym, df_trunc)
    all_bt.extend(bt_trades)
    print(f"  {sym}: {len(bt_trades)} backtest trades")

bt_wins = sum(1 for t in all_bt if t['pnl'] > 0)
bt_pnl = sum(t['pnl'] for t in all_bt)
print(f"\nBacktest total: {len(all_bt)} trades, WR={bt_wins/len(all_bt)*100:.1f}%, PnL=${bt_pnl:.2f}")

# COMPARE
print(f"\n{'='*70}")
print(f"{'COMPARISON (18:00-00:00 UTC)':^70}")
print(f"{'='*70}")
print(f"{'Symbol':<12} {'Time':>6} {'LIVE':>16} {'BT':>22} {'Match':>6}")
print(f"{'-'*12} {'-'*6} {'-'*16} {'-'*22} {'-'*6}")

for t in live_trades:
    _, sym, direction, entry_ms, entry_px, pnl, lp = t
    d = 'LONG' if direction == 1 else 'SHORT'
    et = datetime.fromtimestamp(entry_ms/1000).strftime('%H:%M')
    
    # Find closest backtest trade
    best_bt = None
    best_delta = 99999
    for bt in all_bt:
        if bt['symbol'] != sym: continue
        if bt['et'] is None: continue
        bt_ts = datetime.fromisoformat(bt['et'])
        delta = abs((bt_ts - datetime.fromtimestamp(entry_ms/1000)).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_bt = bt
    
    if best_bt and best_delta < 900:  # within 15 min
        match = '✅' if (d == best_bt['dir']) else '❌'
        bt_str = f"{best_bt['dir']:>5} p={best_bt['ep']:.4f}"
    else:
        match = '❌'
        bt_str = f"{'N/A':>22}"
    
    print(f"  {sym:<12} {et:>6} {'LIVE:':>6}{d:>5} p={lp:.4f} | BT:{bt_str} {match}")

# Summary
matches = 0
for t in live_trades:
    _, sym, direction, entry_ms = t[:4]
    d = 'LONG' if direction == 1 else 'SHORT'
    best_bt = None
    best_delta = 99999
    for bt in all_bt:
        if bt['symbol'] != sym: continue
        if bt['et'] is None: continue
        bt_ts = datetime.fromisoformat(bt['et'])
        delta = abs((bt_ts - datetime.fromtimestamp(entry_ms/1000)).total_seconds())
        if delta < best_delta:
            best_delta = delta
            best_bt = bt
    if best_bt and best_delta < 900 and d == best_bt['dir']:
        matches += 1

print(f"\n{'='*70}")
print(f"Direction match: {matches}/{len(live_trades)} ({matches/len(live_trades)*100:.0f}%)")
print(f"Live WR: {live_wins}/{len(live_trades)} ({live_wins/len(live_trades)*100:.1f}%)")
print(f"BT WR:   {bt_wins}/{len(all_bt)} ({bt_wins/len(all_bt)*100:.1f}%)")
print(f"Live PnL: ${live_pnl:.2f}")
print(f"BT PnL:   ${bt_pnl:.2f}")
