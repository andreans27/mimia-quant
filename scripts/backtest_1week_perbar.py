#!/usr/bin/env python3
"""Per-bar backtest for WIFUSDT using 45d training models."""
import sys, os, warnings, numpy as np, pandas as pd, time
sys.path = ['/root/projects/mimia-quant'] + sys.path
os.chdir('/root/projects/mimia-quant')
warnings.filterwarnings('ignore')

import xgboost as xgb
from src.strategies.ml_features import ensure_ohlcv_data, compute_5m_features_5tf
from src.trading.state import MODEL_DIR, HOLD_BARS, THRESHOLD

symbol = 'WIFUSDT'
thr = THRESHOLD  # 0.79
pos_pct = 0.15

t0 = time.time()
print(f"=== PER-BAR BACKTEST {symbol} ===", flush=True)

# 1. Load features
df_5m = ensure_ohlcv_data(symbol, min_days=45)
feat_df = compute_5m_features_5tf(df_5m, target_candle=9, target_threshold=0.005, intervals=['1h'], for_inference=False)
fn = [c for c in feat_df.columns if c not in ('target','target_long','target_short')]
X_df = feat_df[fn].fillna(0).clip(-10, 10)
print(f"📊 {len(feat_df)} rows, {len(fn)} feats ({time.time()-t0:.0f}s)", flush=True)

# 2. Train 3 seeds per side
hp = {42:{'max_depth':6,'subsample':0.80,'colsample_bytree':0.8,'learning_rate':0.05,'min_child_weight':5},
      101:{'max_depth':5,'subsample':0.85,'colsample_bytree':0.7,'learning_rate':0.06,'min_child_weight':3},
      202:{'max_depth':7,'subsample':0.75,'colsample_bytree':0.9,'learning_rate':0.04,'min_child_weight':7}}

models = {'long': [], 'short': []}
seeds = [42, 101, 202]
split = int(len(feat_df) * 0.65)

t1 = time.time()
for side in ['long', 'short']:
    y = feat_df[f'target_{side}'].values[:split]
    for s in seeds:
        h = hp[s]
        sw = (len(y)-y.sum())/max(y.sum(), 1)
        m = xgb.XGBClassifier(n_estimators=300, max_depth=h['max_depth'], learning_rate=h['learning_rate'],
            subsample=h['subsample'], colsample_bytree=h['colsample_bytree'], min_child_weight=h['min_child_weight'],
            scale_pos_weight=sw, objective='binary:logistic', eval_metric='auc', random_state=s, verbosity=0, use_label_encoder=False)
        np.random.seed(s)
        sel = [f for f, m_ in zip(fn, np.random.choice([True, False], len(fn), p=[0.85, 0.15])) if m_]
        if len(sel) < 10: sel = fn[:10]
        m.fit(X_df.iloc[:split][sel], y)
        models[side].append((m, sel))
    print(f"✅ {side}: {len(models[side])} seeds ({time.time()-t1:.0f}s)", flush=True)

# 3. Per-bar backtest
total = len(feat_df)
bt_start = total - 2016  # 7 days
trades = []
open_trades = []

t2 = time.time()
for i in range(bt_start, total - 10):
    bar_time = feat_df.index[i]
    cp = float(df_5m['close'].iloc[i])
    
    lp = np.mean([m.predict_proba(X_df.iloc[i:i+1][sel])[:, 1][0] for m, sel in models['long']])
    sp = np.mean([m.predict_proba(X_df.iloc[i:i+1][sel])[:, 1][0] for m, sel in models['short']])
    
    signal = 0
    if lp >= thr and lp >= sp: signal = 1
    elif sp >= thr and sp > lp: signal = -1
    
    if signal != 0 and len(open_trades) == 0:
        entry = cp * 1.0004 if signal == 1 else cp * 0.9996
        open_trades.append({'side': signal, 'entry': entry, 'time': bar_time, 'bars': 0, 'entry_price': cp})
    
    for t in list(open_trades):
        t['bars'] += 1
        u = (cp / t['entry'] - 1) if t['side'] == 1 else (t['entry'] / cp - 1)
        
        reason = None
        if t['bars'] >= HOLD_BARS: reason = 'timeout'
        elif u <= -0.05: reason = 'stoploss'
        elif u >= 0.02: reason = 'takeprofit'
        
        if reason:
            ep = cp * 0.9996 if t['side'] == 1 else cp * 1.0004
            if t['side'] == 1:
                t['pnl'] = ep / t['entry'] - 1
            else:
                t['pnl'] = t['entry'] / ep - 1
            t['exit_reason'] = reason
            t['exit_price'] = cp
            t['exit_time'] = bar_time
            trades.append(t)
            open_trades.remove(t)

print(f"⏱ Inference: {time.time()-t2:.0f}s", flush=True)

# 4. Results
print(f"\n{'='*60}", flush=True)
print(f"PER-BAR BACKTEST RESULTS — {symbol}", flush=True)
print(f"{'='*60}", flush=True)
print(f"Period: {feat_df.index[bt_start]} → {feat_df.index[-1]}", flush=True)
print(f"Threshold: {thr} | Hold: {HOLD_BARS} bars | Position: {pos_pct*100:.0f}%", flush=True)
print(f"Total bars: {total - bt_start}", flush=True)
print(f"\n📊 TRADES: {len(trades)}", flush=True)

if trades:
    w = [t for t in trades if t['pnl'] > 0]
    l = [t for t in trades if t['pnl'] <= 0]
    wr = len(w) / len(trades) * 100
    aw = np.mean([t['pnl'] for t in w]) * 100 if w else 0
    al = np.mean([t['pnl'] for t in l]) * 100 if l else 0
    gp = sum(t['pnl'] for t in w) * pos_pct * 10000
    gl = sum(t['pnl'] for t in l) * pos_pct * 10000
    tp = sum(t['pnl'] for t in trades) * pos_pct * 10000 + 10000
    pf = abs(gp / gl) if gl != 0 else 99
    
    print(f"   Win Rate:     {wr:.1f}% ({len(w)}/{len(trades)})", flush=True)
    print(f"   Avg Win:      +{aw:.2f}%", flush=True)
    print(f"   Avg Loss:     {al:.2f}%", flush=True)
    print(f"   Profit Factor: {pf:.2f}x", flush=True)
    print(f"   Gross P&L:    +${gp:.2f} / -${abs(gl):.2f}", flush=True)
    print(f"   Total PnL:    ${tp-10000:.2f} (start $10k)", flush=True)
    print(f"   Final Equity: ${tp:.2f}", flush=True)
    print(f"   Return:       {(tp/10000-1)*100:.2f}%", flush=True)
    
    # By day
    print(f"\n   📅 Daily Breakdown:", flush=True)
    for day in sorted(set(t['time'].strftime('%a %m/%d') for t in trades)):
        dt = [t for t in trades if t['time'].strftime('%a %m/%d') == day]
        dw = len([t for t in dt if t['pnl'] > 0]) / len(dt) * 100
        dp = sum(t['pnl'] for t in dt) * pos_pct * 10000
        print(f"     {day}: {len(dt):3d} trades | WR={dw:.1f}% | PnL=${dp:+.2f}", flush=True)
    
    # By side
    print(f"\n   🔄 By Side:", flush=True)
    for sn, sv in [('LONG', 1), ('SHORT', -1)]:
        st = [t for t in trades if t['side'] == sv]
        if st:
            sw = len([t for t in st if t['pnl'] > 0]) / len(st) * 100
            sp = sum(t['pnl'] for t in st) * pos_pct * 10000
            print(f"     {sn:6s}: {len(st):3d} trades | WR={sw:.1f}% | PnL=${sp:+.2f}", flush=True)
    
    # By exit reason
    print(f"\n   🚪 By Exit Reason:", flush=True)
    for reason in ['timeout', 'stoploss', 'takeprofit']:
        rt = [t for t in trades if t['exit_reason'] == reason]
        if rt:
            rw = len([t for t in rt if t['pnl'] > 0]) / len(rt) * 100
            rp = sum(t['pnl'] for t in rt) * pos_pct * 10000
            print(f"     {reason:12s}: {len(rt):3d} trades | WR={rw:.1f}% | PnL=${rp:+.2f}", flush=True)
    
    # Top/Bottom
    print(f"\n   🏆 Best 3:", flush=True)
    for t in sorted(trades, key=lambda x: -x['pnl'])[:3]:
        sn = 'LONG' if t['side'] == 1 else 'SHORT'
        print(f"     {sn:5s} {t['time'].strftime('%m/%d %H:%M')} | PnL={t['pnl']*100:+.2f}% | {t['exit_reason']}", flush=True)
    
    print(f"\n   💀 Worst 3:", flush=True)
    for t in sorted(trades, key=lambda x: x['pnl'])[:3]:
        sn = 'LONG' if t['side'] == 1 else 'SHORT'
        print(f"     {sn:5s} {t['time'].strftime('%m/%d %H:%M')} | PnL={t['pnl']*100:+.2f}% | {t['exit_reason']}", flush=True)
else:
    print("   ❌ No trades generated", flush=True)

print(f"\n⏱ Total time: {time.time()-t0:.0f}s", flush=True)
