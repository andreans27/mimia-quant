#!/usr/bin/env python3
"""Sweep global HOLD_BARS from 1 to 10 with optimal config."""
import sys; sys.path.insert(0, '.')
import json, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import THRESHOLD, HOLD_BARS, COOLDOWN_BARS, TAKER_FEE, SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL, LIVE_SYMBOLS
warnings.filterwarnings('ignore')
WARMUP_BARS = 200

COOLDOWN = 2
THR = 0.50

def sizing_fn(proba):
    if proba >= 0.80: return 0.35
    if proba >= 0.75: return 0.28
    if proba >= 0.70: return 0.22
    if proba >= 0.65: return 0.18
    return 0.15

class Precomputed:
    def __init__(self, symbol):
        self.symbol = symbol
    def load(self):
        gen = SignalGenerator(self.symbol)
        cached = gen._load_models(self.symbol)
        if not cached: return False
        mg = cached['groups']
        df = gen._ensure_ohlcv_data(self.symbol)
        if df is None or len(df) < 1000: return False
        feat_df = compute_5m_features_5tf(df, for_inference=True)
        if feat_df is None: return False
        self.close_arr = df.loc[feat_df.index, 'close'].astype(float).values
        self.feat_index = feat_df.index
        all_feats = sorted(set(f for _,ms in mg.items() for _,m,mf in ms for f in mf))
        mf_idx = {f:i for i,f in enumerate(all_feats)}
        fc_map = {f:i for i,f in enumerate(feat_df.columns)}
        long_r, short_r = [], []
        for tg, ms in mg.items():
            for _,m,mf in ms:
                avail = [mf_idx[f] for f in mf if f in mf_idx]
                if len(avail)>=5:
                    ref = (m, np.array(avail, np.int32))
                    if tg=='long': long_r.append(ref)
                    elif tg=='short': short_r.append(ref)
        now = datetime.utcnow().replace(second=0,microsecond=0)
        start = now - timedelta(hours=168)
        ts = max(WARMUP_BARS, int(np.searchsorted(self.feat_index, start)))
        self.test_range = list(range(ts, len(self.feat_index)))
        self.n_test = len(self.test_range)
        if self.n_test == 0: return False
        mat = np.zeros((self.n_test, len(all_feats)), np.float64)
        feat_np = feat_df.values
        for name, pos in mf_idx.items():
            if name in fc_map:
                mat[:,pos] = np.clip(np.nan_to_num(feat_np[ts:ts+self.n_test, fc_map[name]], nan=0.0), -10, 10)
        def batch(refs):
            p = np.zeros(self.n_test); nv = np.zeros(self.n_test, np.int32)
            for m, fi in refs:
                try:
                    preds = m.predict_proba(mat[:,fi])[:,1]
                    p += preds; nv += 1
                except: pass
            return np.where(nv>0, p/nv, 0.5)
        def cal(probas, side):
            cal_path = Path('data/ml_models') / f'{self.symbol}_{side}_calibrator.json'
            if cal_path.exists():
                try:
                    with open(cal_path) as f: c = json.load(f)
                    z = c['coef'] * probas + c['intercept']
                    return np.clip(1.0/(1.0+np.exp(-z)), 0.0, 1.0)
                except: pass
            return probas
        self.long_p = cal(batch(long_r), 'long')
        self.short_p = cal(batch(short_r), 'short')
        return True
    def simulate(self, hold):
        long_p=self.long_p; short_p=self.short_p
        n=self.n_test; tr=self.test_range
        sigs=np.zeros(n,np.int32)
        lm=long_p>=THR; sm=short_p>=THR
        both=lm&sm; pick=long_p>=short_p
        sigs[both&pick]=1;sigs[both&~pick]=-1
        sigs[lm&~sm]=1;sigs[sm&~lm]=-1
        probas=np.where(sigs==1,long_p,np.where(sigs==-1,short_p,np.maximum(long_p,short_p)))
        cap=5000.0;peak=5000.0
        pos=0;ep=0.0;eq=0.0;hr=0;cool=0;eproba=0.0;ps=0;pproba=0.0
        trades=[]
        for ii,bi in enumerate(tr):
            price=float(self.close_arr[bi])
            hr=max(0,hr-1);cool=max(0,cool-1)
            if pos!=0 and hr<=0:
                xp=price*(1-SLIPPAGE) if pos==1 else price*(1+SLIPPAGE)
                raw=eq*(xp-ep) if pos==1 else eq*(ep-xp)
                net=raw-xp*eq*TAKER_FEE
                cap+=net;peak=max(peak,cap)
                trades.append({'pnl':net})
                pos=0;cool=COOLDOWN
            if ps!=0 and pos==0 and cool<=0:
                slip=(1+SLIPPAGE) if ps==1 else (1-SLIPPAGE)
                ep=price*slip
                pct=sizing_fn(pproba)
                eq=(cap*pct)/ep
                cap-=ep*eq*TAKER_FEE
                peak=max(peak,cap)
                pos=ps;eproba=pproba;hr=hold
            ps=sigs[ii];pproba=probas[ii]
        nt=len(trades)
        wins=sum(1 for t in trades if t['pnl']>0)
        wr=wins/nt*100 if nt else 0
        tp=sum(t['pnl'] for t in trades)
        gp=sum(t['pnl'] for t in trades if t['pnl']>0)
        gl=abs(sum(t['pnl'] for t in trades if t['pnl']<=0))
        pf=gp/gl if gl else float('inf')
        dd=0.0;run=5000.0;rp=5000.0
        for t in trades:
            run+=t['pnl'];rp=max(rp,run)
            dd=max(dd,(rp-run)/rp*100)
        return {'trades':nt,'wr':round(wr,1),'pnl':round(tp,2),'pf':round(pf,1),'dd':round(dd,2)}

# Load
symbols = LIVE_SYMBOLS
data = {}
print("Loading...")
for i, sym in enumerate(symbols):
    print(f"  [{i+1}/{len(symbols)}] {sym}... ", end='', flush=True)
    pc = Precomputed(sym)
    if pc.load():
        data[sym] = pc
        print(f"OK ({pc.n_test} bars)")
    else:
        print("SKIP")
print(f"Loaded {len(data)}/{len(symbols)}\n")

# Sweep 1-10
HOLD_VALUES = list(range(1, 11))
print(f"{'Hold':>6} | {'Trades':>7} | {'WR%':>6} | {'PnL':>12} | {'PF':>7} | {'DD%':>7} | {'Δ PnL':>10}")
print(f"{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")

best = None
best_pnl = -999999
best_h = None
results = []
prev_pnl = 0

for h in HOLD_VALUES:
    total_pnl=0.0;total_trades=0;total_wins=0;max_dd=0.0
    t0=time.time()
    for sym in symbols:
        if sym not in data: continue
        r=data[sym].simulate(h)
        total_pnl+=r['pnl']
        total_trades+=r['trades']
        total_wins+=int(r['trades']*r['wr']/100)
        max_dd=max(max_dd,r['dd'])
    wr=total_wins/total_trades*100 if total_trades else 0
    dpnl=total_pnl-prev_pnl if prev_pnl!=0 else 0
    prev_pnl=total_pnl
    marker=" ★" if total_pnl>best_pnl else "  "
    if total_pnl>best_pnl:
        best_pnl=total_pnl;best_h=h
    print(f"  h={h:>2d}   {marker} | {total_trades:>7,d} | {wr:>5.1f}% | ${total_pnl:>+9.2f} |        | {max_dd:>6.2f}% | ${dpnl:>+9.2f}")
    results.append({'hold':h,'trades':total_trades,'wr':round(wr,1),'pnl':round(total_pnl,2),'dd':max_dd})

print(f"\n🏆 WINNER: HOLD_BARS = {best_h} (PnL: ${best_pnl:.2f})")
print("\nSorted by PnL:")
for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
    marker=" ⭐" if r['hold']==best_h else ""
    print(f"  h={r['hold']:>2d}: {r['trades']:>4d}t WR={r['wr']:>5.1f}% PnL=${r['pnl']:>+9.2f} DD={r['dd']:.2f}%{marker}")
