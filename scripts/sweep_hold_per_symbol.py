#!/usr/bin/env python3
"""Per-symbol HOLD_BARS sweep: find optimal hold for EACH symbol independently (1-10)."""
import sys; sys.path.insert(0, '.')
import json, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import THRESHOLD, COOLDOWN_BARS, TAKER_FEE, SLIPPAGE, INITIAL_CAPITAL, LIVE_SYMBOLS
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
    def simulate(self, hold, td_only=False):
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

# Per-symbol sweep: for each symbol, find best hold 1-10
HOLD_VALUES = list(range(1, 11))
best_per_symbol = {}

print("="*110)
print(f"{'Symbol':<12} | {'Best Hold':>9} | {'Trades':>7} | {'WR%':>6} | {'PnL':>12} | {'DD%':>7}")
print(f"{'-'*12}-+-{'-'*9}-+-{'-'*7}-+-{'-'*6}-+-{'-'*12}-+-{'-'*7}")

all_best_pnl = 0

for sym in symbols:
    if sym not in data: continue
    pc = data[sym]
    best_h = None
    best_pnl_sym = -999999
    best_result = None
    results_sym = []
    
    for h in HOLD_VALUES:
        r = pc.simulate(h)
        results_sym.append((h, r))
        if r['pnl'] > best_pnl_sym:
            best_pnl_sym = r['pnl']
            best_h = h
            best_result = r
    
    best_per_symbol[sym] = {
        'best_hold': best_h,
        'trades': best_result['trades'],
        'wr': best_result['wr'],
        'pnl': best_result['pnl'],
        'dd': best_result['dd'],
        'all': {str(h): r for h, r in results_sym},
    }
    all_best_pnl += best_result['pnl']
    
    # Print per-symbol table
    line = f"  {sym:<12} | h={best_h:>2d}       | {best_result['trades']:>7,d} | {best_result['wr']:>5.1f}% | ${best_result['pnl']:>+9.2f} | {best_result['dd']:>6.2f}%"
    print(line)

# Baseline: global HOLD_BARS=10
print(f"\n{'='*110}")
print("Baseline: global HOLD_BARS=10 for all symbols")
base_pnl = 0
for sym in symbols:
    if sym not in data: continue
    r = data[sym].simulate(10)
    base_pnl += r['pnl']
print(f"  Total PnL with global h=10: ${base_pnl:.2f}")

print(f"\nBest per-symbol total PnL: ${all_best_pnl:.2f}")
delta = all_best_pnl - base_pnl
print(f"Improvement: ${delta:+.2f} ({delta/abs(base_pnl)*100:+.1f}%)")

# Print per-symbol optimal vs global h=10 comparison
print(f"\n{'='*110}")
print(f"{'Symbol':<12} | {'Opt Hold':>9} | {'Trades':>7} | {'PnL':>10} | {'vs h=10':>10} | {'vs h=9':>10}")
print(f"{'-'*12}-+-{'-'*9}-+-{'-'*7}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

total_vs10 = 0
for sym in symbols:
    if sym not in data: continue
    opt_h = best_per_symbol[sym]['best_hold']
    opt_pnl = best_per_symbol[sym]['pnl']
    pnl10 = data[sym].simulate(10)['pnl']
    pnl9 = data[sym].simulate(9)['pnl']
    dv10 = opt_pnl - pnl10
    dv9 = opt_pnl - pnl9
    total_vs10 += dv10
    marker = " ←" if opt_h != 10 else ""
    print(f"  {sym:<12} | h={opt_h:>2d}       | {best_per_symbol[sym]['trades']:>7,d} | ${opt_pnl:>+7.2f} | ${dv10:>+7.2f} | ${dv9:>+7.2f}{marker}")

print(f"\n{'Total improvement vs h=10:':<45} ${total_vs10:+.2f}")
print(f"  Per-symbol optimal different from global h=10: {sum(1 for v in best_per_symbol.values() if v['best_hold'] != 10)}/{len(best_per_symbol)} symbols")

# Print symbols where opt != 10
diff_symbols = [(sym, v['best_hold']) for sym, v in best_per_symbol.items() if v['best_hold'] != 10]
if diff_symbols:
    print(f"\nSymbols with non-10 optimal:")
    for sym, h in sorted(diff_symbols, key=lambda x: x[0]):
        print(f"  {sym}: {h} bars")

# Save
out = {
    'config': {'threshold': THR, 'cooldown': COOLDOWN, 'sizing': 'aggressive'},
    'best_per_symbol': {sym: {'hold': v['best_hold'], 'pnl': v['pnl'], 'trades': v['trades'], 'wr': v['wr']}
                        for sym, v in best_per_symbol.items()},
    'global_h10_pnl': round(base_pnl, 2),
    'best_combined_pnl': round(all_best_pnl, 2),
    'improvement_vs_global': round(delta, 2),
}
with open('data/per_symbol_hold_sweep.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\n💾 Saved to data/per_symbol_hold_sweep.json")
