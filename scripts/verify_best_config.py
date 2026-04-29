#!/usr/bin/env python3
"""Verify best config: thr=0.50, hold=10, aggressive sizing vs fixed 15%"""
import sys; sys.path.insert(0, '.')
import json, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from src.trading.signals import SignalGenerator
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.state import COOLDOWN_BARS, TAKER_FEE, SLIPPAGE, INITIAL_CAPITAL, LIVE_SYMBOLS
warnings.filterwarnings('ignore')
WARMUP_BARS = 200

def aggressive_sizing(proba):
    if proba >= 0.80: return 0.35
    if proba >= 0.75: return 0.28
    if proba >= 0.70: return 0.22
    if proba >= 0.65: return 0.18
    return 0.15

def fixed_sizing(p):
    return 0.15

def simulate(symbol, thr, hold, sizing_fn, test_hours=168):
    gen = SignalGenerator(symbol)
    cached = gen._load_models(symbol)
    if cached is None: return None
    df_5m = gen._ensure_ohlcv_data(symbol)
    if df_5m is None or len(df_5m) < 1000: return None
    feat_df = compute_5m_features_5tf(df_5m, for_inference=True)
    if feat_df is None: return None
    close_arr = df_5m.loc[feat_df.index, 'close'].astype(float).values
    model_groups = cached['groups']
    all_feats = sorted(set(f for _,ms in model_groups.items() for _,m,mf in ms for f in mf))
    mf_idx = {f:i for i,f in enumerate(all_feats)}
    feat_cols = list(feat_df.columns)
    fc_map = {f:i for i,f in enumerate(feat_cols)}
    long_r, short_r = [], []
    for tg, ms in model_groups.items():
        for _,m,mf in ms:
            avail = [mf_idx[f] for f in mf if f in mf_idx]
            if len(avail)>=5:
                ref = (m, np.array(avail, np.int32))
                if tg=='long': long_r.append(ref)
                elif tg=='short': short_r.append(ref)
    now = datetime.utcnow().replace(second=0,microsecond=0)
    start_dt = now - timedelta(hours=test_hours)
    test_start = max(WARMUP_BARS, int(np.searchsorted(feat_df.index, start_dt)))
    test_range = list(range(test_start, len(feat_df)))
    n_test = len(test_range)
    if n_test == 0: return None
    feat_np = feat_df.values
    mat = np.zeros((n_test, len(all_feats)), np.float64)
    for name, pos in mf_idx.items():
        if name in fc_map:
            mat[:,pos] = np.clip(np.nan_to_num(feat_np[test_start:test_start+n_test, fc_map[name]], nan=0.0), -10, 10)
    def batch_infer(refs):
        p = np.zeros(n_test); nv = np.zeros(n_test, np.int32)
        for m, fi in refs:
            try:
                preds = m.predict_proba(mat[:,fi])[:,1]
                p += preds; nv += 1
            except: pass
        return np.where(nv>0, p/nv, 0.5)
    def apply_cal(probas, side):
        cal_path = Path('data/ml_models') / f'{symbol}_{side}_calibrator.json'
        if cal_path.exists():
            try:
                with open(cal_path) as f: cal = json.load(f)
                z = cal['coef'] * probas + cal['intercept']
                return np.clip(1.0/(1.0+np.exp(-z)), 0.0, 1.0)
            except: pass
        return probas
    long_p = apply_cal(batch_infer(long_r), 'long')
    short_p = apply_cal(batch_infer(short_r), 'short')
    sigs = np.zeros(n_test, np.int32)
    lm = long_p >= thr; sm = short_p >= thr
    both = lm & sm; pick = long_p >= short_p
    sigs[both & pick]=1; sigs[both & ~pick]=-1
    sigs[lm & ~sm]=1; sigs[sm & ~lm]=-1
    probas = np.where(sigs==1, long_p, np.where(sigs==-1, short_p, np.maximum(long_p, short_p)))
    cap = 5000.0; peak = 5000.0
    pos=0; ep=0.0; eq=0.0; hr=0; cd=0; eproba=0.0; ps=0; pproba=0.0
    trades = []
    for ii, bi in enumerate(test_range):
        try:
            price = float(close_arr[bi])
        except:
            continue
        hr = max(0, hr-1); cd = max(0, cd-1)
        if pos != 0 and hr <= 0:
            xp = price*(1-SLIPPAGE) if pos==1 else price*(1+SLIPPAGE)
            raw = eq*(xp-ep) if pos==1 else eq*(ep-xp)
            net = raw - xp*eq*TAKER_FEE
            cap += net; peak = max(peak, cap)
            trades.append({'pnl': net})
            pos=0; cd=COOLDOWN_BARS
        if ps != 0 and pos == 0 and cd <= 0:
            slip = (1+SLIPPAGE) if ps==1 else (1-SLIPPAGE)
            ep = price*slip
            pct = sizing_fn(pproba)
            eq = (cap * pct) / ep
            cap -= ep*eq*TAKER_FEE
            peak = max(peak, cap)
            pos=ps; eproba=pproba; hr=hold
        ps = sigs[ii]; pproba = probas[ii]
    nt = len(trades)
    wins = sum(1 for t in trades if t['pnl']>0)
    wr = wins/nt*100 if nt else 0
    tp = sum(t['pnl'] for t in trades)
    return {'n_trades': nt, 'win_rate': wr, 'total_pnl': tp}

print("Verify: thr=0.50, hold=10 — aggressive sizing vs fixed 15%")
print()
agg_total, agg_trades = 0.0, 0
fix_total, fix_trades = 0.0, 0
for sym in LIVE_SYMBOLS:
    a = simulate(sym, 0.50, 10, aggressive_sizing)
    b = simulate(sym, 0.50, 10, fixed_sizing)
    if a and b:
        agg_total += a['total_pnl']
        agg_trades += a['n_trades']
        fix_total += b['total_pnl']
        fix_trades += b['n_trades']
        delta = a['total_pnl'] - b['total_pnl']
        print(f"  {sym:<12} fixed: ${b['total_pnl']:>+8.2f} ({b['n_trades']}t) -> agg: ${a['total_pnl']:>+8.2f} ({a['n_trades']}t) | delta ${delta:>+7.2f}")
print()
print(f"  {'TOTAL':<12} fixed: ${fix_total:>+8.2f} ({fix_trades}t) -> agg: ${agg_total:>+8.2f} ({agg_trades}t) | delta ${agg_total-fix_total:>+7.2f}")
pct_gain = (agg_total-fix_total)/fix_total*100
print(f"  ===> IMPROVEMENT: +{pct_gain:.1f}%")
