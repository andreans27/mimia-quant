#!/usr/bin/env python3
"""
Batch training: 45-day window, all 20 symbols, sequental.
Saves results to data/ml_models/_batch_45d_final.json
"""
import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/../..')
sys.path.insert(0, os.getcwd())

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from src.strategies.ml_features import ensure_ohlcv_data, ensure_ohlcv_1h, compute_5m_features_5tf
from src.trading.state import LIVE_SYMBOLS, MODEL_DIR, SEEDS

HPARAMS = {
    42:  {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5, 'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7, 'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4, 'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8, 'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}
TRAIN_DAYS, TRAIN_SPLIT = 45, 0.80

print(f"🔥 BATCH 45-DAY TRAINING")
print(f"Symbols: {len(LIVE_SYMBOLS)} | Workers: sequential\n")

results = []
for idx, symbol in enumerate(LIVE_SYMBOLS):
    t0 = time.time()
    try:
        df_5m = ensure_ohlcv_data(symbol, min_days=TRAIN_DAYS)
        if df_5m is None or len(df_5m) < 500:
            print(f"[{idx+1:2d}/20] {symbol:>15s}: SKIP (data)")
            continue
        
        # Fetch 1h OHLCV DIRECTLY from Binance (no look-ahead from resample)
        df_1h = ensure_ohlcv_1h(symbol, min_days=max(TRAIN_DAYS, 20))
        if df_1h is not None:
            print(f"    ✅ 1h: {len(df_1h)} bars (direct from Binance)")
        
        feat_df = compute_5m_features_5tf(df_5m, target_candle=9, target_threshold=0.005, intervals=['1h'], for_inference=False, df_1h=df_1h)
        if len(feat_df) < 300:
            print(f"[{idx+1:2d}/20] {symbol:>15s}: SKIP (feats={len(feat_df)})")
            continue
        
        n = len(feat_df); split = int(n * TRAIN_SPLIT)
        feature_names = [c for c in feat_df.columns if c not in ('target','target_long','target_short')]
        X = feat_df[feature_names].fillna(0).clip(-10, 10)
        
        sym_result = {'symbol': symbol, 'n_rows': n, 'n_feats': len(feature_names), 'time_s': 0}
        
        for side in ['long', 'short']:
            y = feat_df[f'target_{side}'].values
            X_t, y_t = X.iloc[:split].values.astype(np.float32), y[:split]
            X_v, y_v = X.iloc[split:].values.astype(np.float32), y[split:]
            
            models = []
            for seed in SEEDS:
                hp = HPARAMS[seed]
                sw = (len(y_t) - y_t.sum()) / max(y_t.sum(), 1)
                model = xgb.XGBClassifier(n_estimators=400, max_depth=hp['max_depth'],
                    learning_rate=hp['learning_rate'], subsample=hp['subsample'],
                    colsample_bytree=hp['colsample_bytree'], min_child_weight=hp['min_child_weight'],
                    scale_pos_weight=sw, objective='binary:logistic', eval_metric='auc',
                    random_state=seed, verbosity=0, use_label_encoder=False)
                mask = np.random.choice([True, False], size=X_t.shape[1], p=[0.85, 0.15])
                if mask.sum() < 10: mask[:10] = True
                model.fit(X_t[:, mask], y_t)
                model.save_model(str(MODEL_DIR / f'{symbol}_{side}_xgb_ens_{seed}.json'))
                models.append((model, mask))
            
            p = np.zeros((len(X_v), len(models)))
            for i, (m, mask) in enumerate(models):
                p[:, i] = m.predict_proba(X_v[:, mask])[:, 1]
            ep = p.mean(axis=1)
            
            test_auc = float(roc_auc_score(y_v, ep))
            
            # Threshold sweep — boolean mask fix applied
            thr_data = {}
            for thr_dec in range(40, 80):
                thr = thr_dec / 100.0
                pred_bool = ep >= thr
                nt = int(pred_bool.sum())
                if nt >= 10:
                    wr = float(y_v[pred_bool].mean())
                    thr_data[f'{thr:.2f}'] = {'n': nt, 'wr': round(wr, 4)}
            
            # Best WR
            best_wr = max([v['wr'] for v in thr_data.values()]) if thr_data else 0
            best_thr = max(thr_data.keys(), key=lambda k: thr_data[k]['wr']) if thr_data else '-'
            
            sym_result[side] = {
                'auc': test_auc,
                'test_pos_pct': float(y_v.mean()),
                'train_pos_pct': float(y_t.mean()),
                'best_wr': best_wr,
                'best_thr': best_thr,
                'thresholds': thr_data,
            }
        
        sym_result['time_s'] = round(time.time() - t0, 1)
        results.append(sym_result)
        
        # Incremental save
        with open(MODEL_DIR / '_batch_45d_final.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        s = sym_result['long']
        print(f"[{idx+1:2d}/20] {symbol:>15s}: AUC={s['auc']:.3f} | Best WR={s['best_wr']:.1%} @ THR={s['best_thr']} | {sym_result['time_s']:.0f}s")
        sys.stdout.flush()
        
    except Exception as e:
        import traceback
        print(f"[{idx+1:2d}/20] {symbol:>15s}: ERROR {e}")
        traceback.print_exc()
        sys.stdout.flush()

# Save results
out = MODEL_DIR / '_batch_45d_final.json'
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n{'='*60}")
print(f"RESULTS — {len(results)}/{len(LIVE_SYMBOLS)} symbols")
print(f"{'='*60}")
print(f"{'Symbol':>15s} | {'AUC':>5s} | {'Best WR':>8s} | {'THR':>5s} | {'n_test':>6s} | {'Short AUC':>10s}")
print(f"{'-'*15} | {'-'*5} | {'-'*8} | {'-'*5} | {'-'*6} | {'-'*10}")
for r in sorted(results, key=lambda x: x.get('long',{}).get('auc',0), reverse=True):
    s = r.get('long', {})
    ss = r.get('short', {})
    wr_s = f"{s['best_wr']:.0%}" if s.get('best_wr') else '-'
    thr_s = str(s.get('best_thr', '-'))
    auc_s = ss.get('auc', 0)
    n_test = max([v['n'] for v in s.get('thresholds',{}).values()]) if s.get('thresholds') else 0
    print(f"  {r['symbol']:>13s}: {s.get('auc',0):.3f} | {wr_s:>8s} | {thr_s:>5s} | {n_test:>5d} | {auc_s:.3f}")

# Count >=70% WR
wr70 = [r for r in results if r.get('long',{}).get('best_wr',0) >= 0.70]
print(f"\n✅ Symbols with WR >= 70%: {len(wr70)}/{len(results)}")
for r in wr70:
    print(f"  {r['symbol']:>15s}: {r['long']['best_wr']:.0%} @ {r['long']['best_thr']}")

print(f"\n📁 Results saved to {out}")
