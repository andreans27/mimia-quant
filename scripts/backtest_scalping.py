#!/usr/bin/env python3
"""
Scalping Experiment: 14d training, 3-bar target, micro-structure + OI features.
"""
import sys; sys.path = ['/root/projects/mimia-quant'] + sys.path
import os; os.chdir('/root/projects/mimia-quant')
import warnings, time, json, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from src.strategies.ml_features import ensure_ohlcv_data, compute_5m_features_5tf
from src.strategies.market_data_cache import ensure_all_market_data
from src.trading.state import MODEL_DIR, HOLD_BARS

symbol = 'WIFUSDT'
TRAIN_DAYS = 14
TARGET_CANDLE = 3      # 15 min scalping
TARGET_THR = 0.003     # 0.3%
THR = 0.55             # probability threshold (lower for scalping)
POS_PCT = 0.15
HOLD = 6               # 30 min hold for scalping
STOP_LOSS = 0.03       # 3% SL (tighter for scalping)
TAKE_PROFIT = 0.01     # 1% TP

t0 = time.time()
print(f"=== SCALPING EXPERIMENT {symbol} ===")
print(f"Train: {TRAIN_DAYS}d | Target: {TARGET_CANDLE}b ({TARGET_CANDLE*5}min) | THR={TARGET_THR}")

# 1. Clear old OHLCV cache to force re-fetch with new columns
cache_path = f"data/ohlcv_cache/{symbol}_5m.parquet"
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f"🧹 Cache cleared: {cache_path}")

# 2. Fetch OHLCV (now includes quote_volume, trades, taker_buy_quote)
df_5m = ensure_ohlcv_data(symbol, min_days=TRAIN_DAYS)
print(f"📊 OHLCV: {len(df_5m)} bars | Columns: {list(df_5m.columns)}")
has_all = all(c in df_5m.columns for c in ['taker_buy_quote', 'quote_volume', 'trades'])
print(f"   Micro columns complete: {'✅' if has_all else '❌'}")

# 3. Fetch market data (OI, funding rate, top trader)
market_data = ensure_all_market_data(symbol, force_refresh=True)
print(f"📊 Market data loaded: {len(market_data)} sources")

# 4. Compute features with scalping target + market data
feat_df = compute_5m_features_5tf(
    df_5m,
    target_candle=TARGET_CANDLE,
    target_threshold=TARGET_THR,
    intervals=['1h'],
    for_inference=False,
    market_data=market_data,
)
print(f"📊 Features: {len(feat_df)} rows, {len(feat_df.columns)} cols ({time.time()-t0:.0f}s)")

# 5. Train on LAST 14 days with 80/20 chronological split
fn = [c for c in feat_df.columns if c not in ('target','target_long','target_short')]
X_df = feat_df[fn].fillna(0).clip(-10, 10)

n = len(feat_df)
split = int(n * 0.80)
total_feats = len(fn)

print(f"📐 Split: Train {split} bars | OOS {n-split} bars | {total_feats} features")

# Count new micro features
micro = [c for c in fn if any(c.startswith(p) for p in ['taker','num_trades','trade_freq','trade_spike',
       'avg_trade','micro_body','micro_hl','quote_vol'])]
oi_feats = [c for c in fn if c.startswith('oi_')]
fr_feats = [c for c in fn if c.startswith('fr_')]
top_feats = [c for c in fn if c.startswith('top_')]
print(f"  micro_structure: {len(micro)} | OI: {len(oi_feats)} | FR: {len(fr_feats)} | Top: {len(top_feats)}")

# 6. Train models
HPARAMS = {42:{'max_depth':5,'subsample':0.80,'colsample_bytree':0.7,'learning_rate':0.05,'min_child_weight':3},
           101:{'max_depth':4,'subsample':0.85,'colsample_bytree':0.6,'learning_rate':0.06,'min_child_weight':3},
           202:{'max_depth':6,'subsample':0.75,'colsample_bytree':0.8,'learning_rate':0.04,'min_child_weight':5}}
seeds = [42, 101, 202]

models = {'long': [], 'short': []}

for side in ['long', 'short']:
    y = feat_df[f'target_{side}'].values
    X_t = X_df.iloc[:split].values.astype(np.float32)
    y_t = y[:split]
    X_v = X_df.iloc[split:].values.astype(np.float32)
    y_v = y[split:]
    
    t1 = time.time()
    for s in seeds:
        h = HPARAMS[s]
        sw = (len(y_t)-y_t.sum())/max(y_t.sum(),1)
        m = xgb.XGBClassifier(n_estimators=400, max_depth=h['max_depth'], learning_rate=h['learning_rate'],
            subsample=h['subsample'], colsample_bytree=h['colsample_bytree'], min_child_weight=h['min_child_weight'],
            scale_pos_weight=sw, objective='binary:logistic', eval_metric='auc', random_state=s, verbosity=0, use_label_encoder=False)
        np.random.seed(s)
        mask = np.random.choice([True, False], size=X_t.shape[1], p=[0.85, 0.15])
        if mask.sum() < 10: mask[:10] = True
        m.fit(X_t[:, mask], y_t)
        models[side].append((m, mask))
    
    # Validate
    p = np.zeros((len(X_v), len(seeds)))
    for i, (m, mask) in enumerate(models[side]):
        p[:, i] = m.predict_proba(X_v[:, mask])[:, 1]
    ep = p.mean(axis=1)
    
    auc = roc_auc_score(y_v, ep)
    test_pos = y_v.mean()
    
    # Threshold sweep
    for thr_dec in [40, 45, 50, 55, 60, 65, 70]:
        thr = thr_dec / 100.0
        pred = ep >= thr
        nt = pred.sum()
        if nt >= 10:
            wr = y_v[pred].mean()
    
    print(f"  {side}: AUC={auc:.3f} | test_pos={test_pos:.2%} | {time.time()-t1:.0f}s")

# 7. Per-bar backtest (LAST 7 days)
total = len(feat_df)
bt_start = total - 2016
trades = []
open_trades = []
FEES = 0.0004  # 0.04% per side

t2 = time.time()
for i in range(bt_start, total - TARGET_CANDLE):
    bar_time = feat_df.index[i]
    cp = float(df_5m['close'].iloc[i])
    
    row = X_df.iloc[i].values.astype(np.float32).reshape(1, -1)
    
    lp = np.mean([m.predict_proba(row[:, mask])[:, 1][0] for m, mask in models['long']])
    sp = np.mean([m.predict_proba(row[:, mask])[:, 1][0] for m, mask in models['short']])
    
    signal = 0
    if lp >= THR and lp >= sp: signal = 1
    elif sp >= THR and sp > lp: signal = -1
    
    if signal != 0 and len(open_trades) == 0:
        entry = cp * (1 + FEES) if signal == 1 else cp * (1 - FEES)
        open_trades.append({'side': signal, 'entry': entry, 'time': bar_time, 'bars': 0, 'entry_px': cp})
    
    for t in list(open_trades):
        t['bars'] += 1
        u = (cp / t['entry'] - 1) if t['side'] == 1 else (t['entry'] / cp - 1)
        
        reason = None
        if t['bars'] >= HOLD: reason = 'timeout'
        elif u <= -STOP_LOSS: reason = 'stoploss'
        elif u >= TAKE_PROFIT: reason = 'takeprofit'
        
        if reason:
            ep = cp * (1 - FEES) if t['side'] == 1 else cp * (1 + FEES)
            t['pnl'] = (ep/t['entry'] - 1) if t['side'] == 1 else (t['entry']/ep - 1)
            t['exit_reason'] = reason
            t['exit_px'] = cp
            trades.append(t)
            open_trades.remove(t)

# 8. Results
print(f"\n{'='*60}")
print(f"SCALPING BACKTEST — {symbol}")
print(f"{'='*60}")
print(f"Train: {TRAIN_DAYS}d | Target: {TARGET_CANDLE}b/{TARGET_THR*100:.1f}% | THR={THR} | Hold={HOLD}b")
print(f"SL={STOP_LOSS*100:.0f}% | TP={TAKE_PROFIT*100:.0f}% | Pos={POS_PCT*100:.0f}%")
print(f"Period: {feat_df.index[bt_start]} → {feat_df.index[-1]} | {time.time()-t2:.0f}s inference")
print(f"\n📊 TRADES: {len(trades)}")

if trades:
    w = [t for t in trades if t['pnl'] > 0]
    l = [t for t in trades if t['pnl'] <= 0]
    wr = len(w)/len(trades)*100
    aw = np.mean([t['pnl'] for t in w])*100 if w else 0
    al = np.mean([t['pnl'] for t in l])*100 if l else 0
    gp = sum(t['pnl'] for t in w)*POS_PCT*10000
    gl = sum(t['pnl'] for t in l)*POS_PCT*10000
    tp = sum(t['pnl'] for t in trades)*POS_PCT*10000 + 10000
    pf = abs(gp/gl) if gl != 0 else 99
    
    print(f"   Win Rate:     {wr:.1f}% ({len(w)}/{len(trades)})")
    print(f"   Avg Win:      +{aw:.2f}%")
    print(f"   Avg Loss:     {al:.2f}%")
    print(f"   Profit Factor: {pf:.2f}x")
    print(f"   Total PnL:    ${tp-10000:.2f} (start $10k)")
    print(f"   Final Equity: ${tp:.2f}")
    print(f"   Return:       {(tp/10000-1)*100:.2f}%")
    
    # Daily
    print(f"\n   📅 Daily:")
    for day in sorted(set(t['time'].strftime('%a %m/%d') for t in trades)):
        dt = [t for t in trades if t['time'].strftime('%a %m/%d') == day]
        dw = len([t for t in dt if t['pnl'] > 0])/len(dt)*100
        dp = sum(t['pnl'] for t in dt)*POS_PCT*10000
        dur = max(t['bars'] for t in dt) if dt else 0
        print(f"     {day}: {len(dt):3d} trades | WR={dw:.1f}% | PnL=${dp:+.2f} | max_hold={dur}b")
    
    # Side
    for sn, sv in [('LONG',1),('SHORT',-1)]:
        st = [t for t in trades if t['side']==sv]
        if st:
            sw = len([t for t in st if t['pnl']>0])/len(st)*100
            sp_sum = sum(t['pnl'] for t in st)*POS_PCT*10000
            print(f"\n   {sn}: {len(st):3d} trades | WR={sw:.1f}% | PnL=${sp_sum:+.2f}")
    
    # Exit
    print(f"\n   🚪 Exit:")
    for reason in ['timeout','stoploss','takeprofit']:
        rt = [t for t in trades if t['exit_reason']==reason]
        if rt:
            rw = len([t for t in rt if t['pnl']>0])/len(rt)*100
            rp = sum(t['pnl'] for t in rt)*POS_PCT*10000
            print(f"     {reason:12s}: {len(rt):3d} trades | WR={rw:.1f}% | PnL=${rp:+.2f}")
    
    # Best/Worst
    print(f"\n   🏆 Best: {sorted(trades, key=lambda x:-x['pnl'])[0]}")
    print(f"   💀 Worst: {sorted(trades, key=lambda x:x['pnl'])[0]}")
else:
    print("   ❌ No trades generated")

print(f"\n⏱ Total: {time.time()-t0:.0f}s")
