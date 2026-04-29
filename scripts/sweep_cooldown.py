#!/usr/bin/env python3
"""Sweep COOLDOWN_BARS from 0 to 5 with optimal config (thr=0.50, hold=10, aggressive sizing)."""
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

def aggressive_sizing(proba):
    if proba >= 0.80: return 0.35
    if proba >= 0.75: return 0.28
    if proba >= 0.70: return 0.22
    if proba >= 0.65: return 0.18
    return 0.15

COOLDOWN_TESTS = [0, 1, 2, 3, 4, 5]
THR = 0.50
HLD = 10

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
        test_range = list(range(ts, len(self.feat_index)))
        self.n_test = len(test_range)
        if self.n_test == 0: return False
        self.test_range = test_range
        
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
    
    def simulate(self, cd):
        long_p = self.long_p; short_p = self.short_p
        n = self.n_test; tr = self.test_range
        sigs = np.zeros(n, np.int32)
        lm = long_p >= THR; sm = short_p >= THR
        both = lm & sm; pick = long_p >= short_p
        sigs[both & pick]=1; sigs[both & ~pick]=-1
        sigs[lm & ~sm]=1; sigs[sm & ~lm]=-1
        probas = np.where(sigs==1, long_p, np.where(sigs==-1, short_p, np.maximum(long_p, short_p)))
        
        cap = 5000.0; peak = 5000.0
        pos=0; ep=0.0; eq=0.0; hr=0; cool=0; eproba=0.0; ps=0; pproba=0.0
        trades = []
        for ii, bi in enumerate(tr):
            price = float(self.close_arr[bi])
            hr = max(0, hr-1); cool = max(0, cool-1)
            if pos != 0 and hr <= 0:
                xp = price*(1-SLIPPAGE) if pos==1 else price*(1+SLIPPAGE)
                raw = eq*(xp-ep) if pos==1 else eq*(ep-xp)
                net = raw - xp*eq*TAKER_FEE
                cap += net; peak = max(peak, cap)
                trades.append({'pnl': net})
                pos=0; cool=cd
            if ps != 0 and pos == 0 and cool <= 0:
                slip = (1+SLIPPAGE) if ps==1 else (1-SLIPPAGE)
                ep = price*slip
                pct = aggressive_sizing(pproba)
                eq = (cap * pct) / ep
                cap -= ep*eq*TAKER_FEE
                peak = max(peak, cap)
                pos=ps; eproba=pproba; hr=HLD
            ps = sigs[ii]; pproba = probas[ii]
        
        nt = len(trades)
        wins = sum(1 for t in trades if t['pnl']>0)
        wr = wins/nt*100 if nt else 0
        tp = sum(t['pnl'] for t in trades)
        gp = sum(t['pnl'] for t in trades if t['pnl']>0)
        gl = abs(sum(t['pnl'] for t in trades if t['pnl']<=0))
        pf = gp/gl if gl else float('inf')
        return {'n_trades': nt, 'wr': round(wr,1), 'pnl': round(tp,2), 'pf': round(pf,1)}

print("🏆 COOLDOWN_BARS SWEEP (0-5)")
print(f"   Fixed: THRESHOLD={THR}, HOLD_BARS={HLD}, aggressive sizing")
print(f"   Symbols: {len(LIVE_SYMBOLS)} pairs, 168h (1 week)")
print()

# Load all symbols
symbols = LIVE_SYMBOLS[:]
data = {}
for i, sym in enumerate(symbols):
    print(f"  Loading [{i+1}/{len(symbols)}] {sym}... ", end='', flush=True)
    pc = Precomputed(sym)
    if pc.load():
        data[sym] = pc
        print(f"✅ ({pc.n_test} bars)")
    else:
        print("❌")

print(f"\n✅ {len(data)}/{len(symbols)} loaded\n")

# Sweep cooldown
print(f"{'Cooldown':>10} | {'Trades':>7} | {'WR%':>6} | {'PnL':>11} | {'PF':>8} | {'Δ PnL':>10} | {'Δ Tr':>6}")
print(f"{'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*11}-+-{'-'*8}-+-{'-'*10}-+-{'-'*6}")

best_pnl = -999999
best_cd = None
results = []

for cd in COOLDOWN_TESTS:
    total_pnl = 0.0; total_trades = 0; total_wins = 0; total_pf_sum = 0.0; n_sym = 0
    for sym in symbols:
        if sym not in data: continue
        r = data[sym].simulate(cd)
        total_pnl += r['pnl']
        total_trades += r['n_trades']
        total_wins += int(r['n_trades'] * r['wr'] / 100)
        n_sym += 1
    wr = total_wins/total_trades*100 if total_trades else 0
    delta_pnl = total_pnl - best_pnl if best_pnl > -999998 else 0
    delta_tr = total_trades - results[-1]['trades'] if results else 0
    marker = " ★" if total_pnl > best_pnl else "  "
    print(f"  cd={cd:>2d}      {marker} | {total_trades:>7,d} | {wr:>5.1f}% | ${total_pnl:>+9.2f} |        | ${delta_pnl:>+9.2f} | {delta_tr:>+5d}")
    
    results.append({'cd': cd, 'trades': total_trades, 'wr': round(wr,1), 'pnl': round(total_pnl,2)})
    if total_pnl > best_pnl:
        best_pnl = total_pnl
        best_cd = cd

print(f"\n🏆 WINNER: COOLDOWN_BARS = {best_cd}")
print(f"   PnL: ${best_pnl:.2f}")
print()

# Print all in one line
print("Summary:")
for r in results:
    print(f"  cd={r['cd']}: {r['trades']}t, WR={r['wr']}%, PnL=${r['pnl']:.2f}")
