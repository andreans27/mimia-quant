#!/usr/bin/env python3
"""
Sweep clean ML pipeline across variations: target windows, symbols, feature modes.
No look-ahead: direct 1h klines + 1h index shift.
"""

import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from pathlib import Path

from src.strategies.ml_features import (
    ensure_ohlcv_data, ensure_ohlcv_1h, compute_5m_features_5tf,
    compute_technical_features,
)
from src.trading.state import MODEL_DIR

# ─── Config ───
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
TARGET_BARS = [6, 9, 18]   # 30m, 45m, 90m
TARGET_THR = 0.005         # 0.5%
TRAIN_DAYS = 45
TRAIN_SPLIT = 0.80

HPARAMS = {
    42:  {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
}
SEEDS = [42]

results = []

for symbol in SYMBOLS:
    print(f"\n{'='*70}")
    print(f"📡 {symbol}")
    print('='*70)
    
    # Fetch data
    df_5m = ensure_ohlcv_data(symbol, min_days=TRAIN_DAYS)
    df_1h = ensure_ohlcv_1h(symbol, min_days=max(TRAIN_DAYS, 20))
    if df_5m is None or df_1h is None:
        print(f"  ❌ No data")
        continue
    
    for target_bars in TARGET_BARS:
        for use_subsample in [True, False]:
            mode_name = 'subsample' if use_subsample else 'full_feats'
            print(f"\n  ── target={target_bars} bars ({target_bars*5}m) | {mode_name} ──")
            
            t0 = time.time()
            
            # Compute features
            feat_df = compute_5m_features_5tf(
                df_5m, target_candle=target_bars,
                target_threshold=TARGET_THR,
                intervals=['1h'], for_inference=False, df_1h=df_1h
            )
            if len(feat_df) < 500:
                print(f"    ❌ Too few rows: {len(feat_df)}")
                continue
            
            feature_names = [c for c in feat_df.columns if c not in ('target','target_long','target_short')]
            X = feat_df[feature_names].fillna(0).clip(-10, 10)
            n = len(feat_df)
            split = int(n * TRAIN_SPLIT)
            
            result = {
                'symbol': symbol, 'target_bars': target_bars, 'mode': mode_name,
                'n_rows': n, 'n_feats': len(feature_names),
                'oos_rows': n - split,
            }
            
            for side in ['long', 'short']:
                y = feat_df[f'target_{side}'].values
                y_train, y_oos = y[:split], y[split:]
                result[f'{side}_pos_rate'] = round(y.mean() * 100, 2)
                result[f'{side}_oos_pos_rate'] = round(y_oos.mean() * 100, 2)
                
                sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
                
                models = []
                for seed in SEEDS:
                    hp = HPARAMS[seed]
                    model = xgb.XGBClassifier(n_estimators=400,
                        max_depth=hp['max_depth'],
                        learning_rate=hp['learning_rate'],
                        subsample=hp['subsample'],
                        colsample_bytree=hp['colsample_bytree'],
                        min_child_weight=hp['min_child_weight'],
                        scale_pos_weight=sw,
                        objective='binary:logistic', eval_metric='auc',
                        random_state=seed, verbosity=0)
                    
                    if use_subsample:
                        mask = np.random.choice([True, False], size=X.shape[1], p=[0.85, 0.15])
                        if mask.sum() < 10: mask[:10] = True
                        X_sub = X.iloc[:split].iloc[:, mask]
                    else:
                        mask = np.ones(X.shape[1], dtype=bool)
                        X_sub = X.iloc[:split]
                    
                    model.fit(X_sub, y_train)
                    models.append((model, mask))
                
                # OOS prediction
                p = np.zeros((n - split, len(models)))
                for i, (m, mask) in enumerate(models):
                    p[:, i] = m.predict_proba(X.iloc[split:, mask].values.astype(np.float32))[:, 1]
                ep = p.mean(axis=1)
                
                # AUC
                try:
                    result[f'{side}_auc'] = round(float(roc_auc_score(y_oos, ep)), 4)
                except:
                    result[f'{side}_auc'] = 0.0
                
                # Threshold sweep
                best_wr = 0
                best_thr = '-'
                best_n = 0
                for thr_dec in range(50, 90):
                    thr = thr_dec / 100.0
                    pred = ep >= thr
                    nt = int(pred.sum())
                    if nt >= 5:
                        wr = float(y_oos[pred].mean())
                        if wr > best_wr:
                            best_wr = wr
                            best_thr = f'{thr:.2f}'
                            best_n = nt
                
                result[f'{side}_best_wr'] = round(best_wr * 100, 1) if best_thr != '-' else 0
                result[f'{side}_best_thr'] = best_thr
                result[f'{side}_best_n'] = best_n
                
                # Also check: what's the WR at the THRESHOLD that gives at least 10+ trades?
                for thr_dec in range(50, 90):
                    thr = thr_dec / 100.0
                    pred = ep >= thr
                    nt = int(pred.sum())
                    if nt >= 10:
                        wr = float(y_oos[pred].mean())
                        result[f'{side}_wr_at_{thr:.2f}'] = round(wr * 100, 1)
                        result[f'{side}_n_at_{thr:.2f}'] = nt
                        break
                else:
                    result[f'{side}_wr_at_thr'] = 0
                    result[f'{side}_n_at_thr'] = 0
            
            result['time_s'] = round(time.time() - t0, 1)
            results.append(result)
            
            # Print summary
            for side in ['long', 'short']:
                wr = result[f'{side}_best_wr']
                thr = result[f'{side}_best_thr']
                n = result[f'{side}_best_n']
                auc = result[f'{side}_auc']
                pr = result[f'{side}_oos_pos_rate']
                print(f"    {side:>5s}: AUC={auc:.3f} OOS_pos={pr:.1f}% best_wr={wr:.1f}% @ thr={thr} n={n}")

# ─── Final Summary ───
print(f"\n\n{'='*70}")
print("FINAL SUMMARY")
print('='*70)
print(f"{'Symbol':>10s} {'Bars':>5s} {'Mode':>12s} {'L_AUC':>6s} {'L_WR':>5s} {'S_AUC':>6s} {'S_WR':>5s} {'L_pos':>5s}")
print('-' * 55)
for r in results:
    print(f"{r['symbol']:>10s} {r['target_bars']:>5d} {r['mode']:>12s} "
          f"{r['long_auc']:>6.3f} {r['long_best_wr']:>4.0f}% "
          f"{r['short_auc']:>6.3f} {r['short_best_wr']:>4.0f}% "
          f"{r['long_oos_pos_rate']:>4.1f}%")

# Save
Path('data/sweep_clean_results.json').write_text(json.dumps(results, indent=2))
print(f"\n✅ Saved to data/sweep_clean_results.json")
