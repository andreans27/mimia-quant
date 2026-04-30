#!/usr/bin/env python3
"""CLEAN 7-day per-bar backtest — NO look-ahead, proper methodology."""
import sys; sys.path = ['/root/projects/mimia-quant'] + sys.path
import os; os.chdir('/root/projects/mimia-quant')
import warnings, numpy as np, pandas as pd, time
warnings.filterwarnings('ignore')

import xgboost as xgb
from src.strategies.ml_features import ensure_ohlcv_data, ensure_ohlcv_1h, compute_5m_features_5tf
from src.trading.state import MODEL_DIR

symbol = 'WIFUSDT'
TARGET_CANDLE = 9; TARGET_THR = 0.005; THR = 0.70
HOLD = TARGET_CANDLE; POS_PCT = 0.15; FEES = 0.0004

t0 = time.time()
print(f"=== CLEAN BACKTEST {symbol} ===")
print(f"THR={THR} | Hold={HOLD}b | TP={TARGET_THR*100:.1f}% | SL={TARGET_THR*100:.1f}%")

# Load data
df_5m = ensure_ohlcv_data(symbol, min_days=45)
df_1h = ensure_ohlcv_1h(symbol, min_days=45)

# Full features (drop last 12 to eliminate incomplete 1h)
feat_all = compute_5m_features_5tf(df_5m, target_candle=TARGET_CANDLE,
    target_threshold=TARGET_THR, intervals=['1h'], for_inference=False, df_1h=df_1h)
feat = feat_all.iloc[:-12].copy()
fn = [c for c in feat.columns if c not in ('target','target_long','target_short')]
X_all = feat[fn].fillna(0).clip(-10, 10).values.astype(np.float32)
n = len(feat)

# Load clean model
model = xgb.XGBClassifier(); model.load_model(str(MODEL_DIR / f'{symbol}_clean_v2_model.json'))
mask = np.load(MODEL_DIR / f'{symbol}_clean_v2_mask.npy')
print(f"✅ Clean model loaded ({mask.sum()}/{len(mask)} features)")

# Per-bar backtest on LAST 7 days
bt_start = n - 2016
trades = []; open_trades = []

t1 = time.time()
for i in range(bt_start, n):
    cp = float(df_5m['close'].iloc[i])
    bt = feat.index[i]
    row = X_all[i:i+1]
    
    # Ensemble with SINGLE model (clean_v2)
    proba = model.predict_proba(row[:, mask])[:, 1][0]
    signal = 1 if proba >= THR else 0
    # No short model yet for clean
    
    if signal != 0 and len(open_trades) == 0:
        entry = cp * (1 + FEES)
        open_trades.append({'side': signal, 'entry': entry, 'time': bt, 'bars': 0, 'entry_px': cp, 'proba': proba})
    
    for t in list(open_trades):
        t['bars'] += 1
        u = cp / t['entry'] - 1
        
        reason = None
        if u >= TARGET_THR * 0.8: reason = 'tp'
        elif u <= -TARGET_THR: reason = 'sl'
        elif t['bars'] >= HOLD: reason = 'timeout'
        
        if reason:
            ep = cp * (1 - FEES)
            t['pnl'] = ep / t['entry'] - 1
            t['exit_reason'] = reason
            trades.append(t); open_trades.remove(t)

print(f"⏱ Inference: {time.time()-t1:.0f}s for {n - bt_start} bars")

# Results
print(f"\n{'='*60}")
print(f"CLEAN BACKTEST — {symbol} (NO look-ahead)")
print(f"{'='*60}")
print(f"Period: {feat.index[bt_start]} → {feat.index[-1]}")
print(f"\n📊 TRADES: {len(trades)}")

if trades:
    w = [t for t in trades if t['pnl'] > 0]; l = [t for t in trades if t['pnl'] <= 0]
    wr = len(w)/len(trades)*100
    gp = sum(t['pnl'] for t in w)*POS_PCT*10000
    gl = sum(t['pnl'] for t in l)*POS_PCT*10000
    tp = sum(t['pnl'] for t in trades)*POS_PCT*10000 + 10000
    pf = abs(gp/gl) if gl != 0 else 99
    aw = np.mean([t['pnl'] for t in w])*100 if w else 0
    al = np.mean([t['pnl'] for t in l])*100 if l else 0
    
    print(f"   Win Rate:     {wr:.1f}% ({len(w)}/{len(trades)})")
    print(f"   PF:           {pf:.2f}x")
    print(f"   Avg W/L:      +{aw:.2f}% / {al:.2f}%")
    print(f"   Total PnL:    ${tp-10000:.2f} (start $10k)")
    print(f"   Return:       {(tp/10000-1)*100:.2f}%")
    
    print(f"\n   📅 Daily:")
    for day in sorted(set(t['time'].strftime('%a %m/%d') for t in trades)):
        dt = [t for t in trades if t['time'].strftime('%a %m/%d') == day]
        dw = len([t for t in dt if t['pnl'] > 0])/len(dt)*100
        dp = sum(t['pnl'] for t in dt)*POS_PCT*10000
        print(f"     {day}: {len(dt):3d} trades | WR={dw:.1f}% | PnL=${dp:+.2f}")
    
    for reason in ['timeout','tp','sl']:
        rt = [t for t in trades if t['exit_reason']==reason]
        if rt:
            rw = len([t for t in rt if t['pnl'] > 0])/len(rt)*100
            rp = sum(t['pnl'] for t in rt)*POS_PCT*10000
            print(f"   {reason:12s}: {len(rt):3d} trades | WR={rw:.1f}% | PnL=${rp:+.2f}")
    
    print(f"\n📋 ALL TRADES:")
    for t in trades:
        print(f"   {'L' if t['side']==1 else 'S'} {t['time'].strftime('%m/%d %H:%M')} | proba={t['proba']:.2f} | pnl={t['pnl']*100:+.2f}% | {t['exit_reason']}")
else:
    print("   ❌ No trades")
    
print(f"\n⏱ Total: {time.time()-t0:.0f}s")
