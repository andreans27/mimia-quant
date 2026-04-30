#!/usr/bin/env python3
"""Compare in-memory model predictions vs disk-loaded model predictions."""

import sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
from pathlib import Path

from src.strategies.ml_features import (
    ensure_ohlcv_data, ensure_ohlcv_1h, compute_5m_features_5tf,
    compute_technical_features,
)
from src.trading.state import LIVE_SYMBOLS, MODEL_DIR, SEEDS

SYMBOL = "BTCUSDT"
TRAIN_DAYS = 45

# 1. Fetch & compute features (same as training)
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
df_1h = ensure_ohlcv_1h(SYMBOL, min_days=max(TRAIN_DAYS, 20))

feat_df = compute_5m_features_5tf(
    df_5m, target_candle=9, target_threshold=0.005,
    intervals=['1h'], for_inference=False, df_1h=df_1h
)

exclude = {'target', 'target_long', 'target_short'}
feature_names = [c for c in feat_df.columns if c not in exclude]
X = feat_df[feature_names].fillna(0).clip(-10, 10)

n = len(feat_df)
split = int(n * 0.80)

# 2. Train model (DataFrame-based, preserves feature names)
HPARAMS = {
    42:  {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5, 'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7, 'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4, 'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8, 'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}

models_inmem = {}
X_test = X.iloc[split:].values.astype(np.float32)

# Pick a single test row
test_row_idx = 10  # row 10 of test set
test_df_row = X.iloc[split:test_row_idx+1].iloc[-1:]  # one row as DataFrame

print("=== Model comparison: In-memory vs Disk-loaded ===")
for side in ['long', 'short']:
    y = feat_df[f'target_{side}'].values
    y_train = y[:split]
    
    models_inmem[side] = []
    for seed in SEEDS:
        hp = HPARAMS[seed]
        sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        model_inmem = xgb.XGBClassifier(n_estimators=400, max_depth=hp['max_depth'],
            learning_rate=hp['learning_rate'], subsample=hp['subsample'],
            colsample_bytree=hp['colsample_bytree'], min_child_weight=hp['min_child_weight'],
            scale_pos_weight=sw, objective='binary:logistic', eval_metric='auc',
            random_state=seed, verbosity=0, use_label_encoder=False)
        
        mask = np.random.choice([True, False], size=X.shape[1], p=[0.85, 0.15])
        if mask.sum() < 10: mask[:10] = True
        X_train_subset = X.iloc[:split].iloc[:, mask]
        model_inmem.fit(X_train_subset, y_train)
        
        # Debug shapes
        print(f"    mask.sum()={mask.sum()}, X_test_subset shape={test_df_row.iloc[:, mask].shape}")
        
        # Save to disk
        model_path = MODEL_DIR / f'{SYMBOL}_{side}_xgb_ens_{seed}.json'
        model_inmem.save_model(str(model_path))
        
        # Load from disk
        model_disk = xgb.XGBClassifier()
        model_disk.load_model(str(model_path))
        
        models_inmem[side].append((model_inmem, mask))
        
        # Predict on test row
        X_test_subset = test_df_row.iloc[:, mask].values.astype(np.float32)
        
        pred_inmem = model_inmem.predict_proba(X_test_subset)[:, 1][0]
        pred_disk = model_disk.predict_proba(X_test_subset)[:, 1][0]
        
        feat_names_inmem = model_inmem.get_booster().feature_names
        feat_names_disk = model_disk.get_booster().feature_names
        
        print(f"\n  {side} seed={seed}:")
        print(f"    Feature names type: inmem={type(feat_names_inmem)}, disk={type(feat_names_disk)}")
        if feat_names_inmem is not None:
            print(f"    Feature names: {feat_names_inmem[:3]}... ({len(feat_names_inmem)} total)")
        print(f"    In-memory: {pred_inmem:.6f}")
        print(f"    Disk:      {pred_disk:.6f}")
        print(f"    DIFF:      {abs(pred_inmem - pred_disk):.8f}")
        
        # Also check: does the in-memory model predict same on numpy vs DataFrame?
        X_test_np = X_test[test_row_idx:test_row_idx+1, mask].astype(np.float32)
        pred_np = model_inmem.predict_proba(X_test_np)[:, 1][0]
        print(f"    Numpy inp: {pred_np:.6f}")
        print(f"    DataFrame: {pred_inmem:.6f}")
        print(f"    Np-vs-Df:  {abs(pred_np - pred_inmem):.8f}")

print("\n\n=== Ensemble prediction on test data ===")
# Compare ensemble predictions
for side in ['long', 'short']:
    # In-memory ensemble
    p_im = np.zeros((len(X_test), len(SEEDS)))
    for i, (m, mask) in enumerate(models_inmem[side]):
        p_im[:, i] = m.predict_proba(X_test[:, mask])[:, 1]
    ensemble_im = p_im.mean(axis=1)
    
    # Disk ensemble (load fresh)
    p_disk = np.zeros((len(X_test), len(SEEDS)))
    for i, seed in enumerate(SEEDS):
        m = xgb.XGBClassifier()
        m.load_model(str(MODEL_DIR / f'{SYMBOL}_{side}_xgb_ens_{seed}.json'))
        mf = m.get_booster().feature_names
        print(f"\n  {side} seed={seed}: feature_names={mf is not None}")
        if mf is not None:
            print(f"    Features: {mf[:3]}... ({len(mf)} total)")
            # Build feature index
            feat_idx = [feature_names.index(f) for f in mf if f in feature_names]
            print(f"    Matched: {len(feat_idx)}/{len(mf)}")
            if len(feat_idx) < 3:
                print(f"    ❌ MODEL CANNOT PREDICT (too few features matched)")
            p_disk[:, i] = m.predict_proba(X_test[:, feat_idx])[:, 1]
    
    ensemble_disk = p_disk.mean(axis=1)
    diff = np.abs(ensemble_im - ensemble_disk)
    print(f"\n  {side}: mean diff={diff.mean()*100:.4f}%, max diff={diff.max()*100:.4f}%")
    print(f"  First 5 predictions:")
    for j in range(min(5, len(X_test))):
        print(f"    Row {j}: IM={ensemble_im[j]:.4f} Disk={ensemble_disk[j]:.4f} diff={abs(ensemble_im[j]-ensemble_disk[j])*100:.4f}%")
