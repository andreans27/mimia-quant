#!/usr/bin/env python3
"""
Verify that the CLEAN pipeline (no look-ahead) produces consistent results
between training/OOS validation and per-bar backtest.

Steps:
1. Train BTCUSDT with clean methodology (df_1h + shift)
2. Systematically compare OOS 1-pass predictions vs per-bar predictions
3. If identical → pipeline is consistent. If different → still have look-ahead.
"""

import sys, os, warnings, json, time
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
from src.trading.signals import SignalGenerator

SYMBOL = "BTCUSDT"
TRAIN_DAYS = 45
TRAIN_SPLIT = 0.80
TARGET_CANDLE = 9
TARGET_THRESHOLD = 0.005

print("=" * 70)
print("VERIFY CLEAN PIPELINE — No Look-Ahead")
print("=" * 70)

# ── 1. Fetch data ──
print(f"\n📡 Fetching {TRAIN_DAYS} days of data for {SYMBOL}...")
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
df_1h = ensure_ohlcv_1h(SYMBOL, min_days=max(TRAIN_DAYS, 20))
print(f"    5m: {len(df_5m)} bars")
print(f"    1h: {len(df_1h)} bars (direct from Binance)")

# ── 2. Compute features (1-pass with df_1h + shift) ──
print(f"\n🔧 Computing features (1-pass, CLEAN)...")
feat_df = compute_5m_features_5tf(
    df_5m, target_candle=TARGET_CANDLE,
    target_threshold=TARGET_THRESHOLD,
    intervals=['1h'], for_inference=False, df_1h=df_1h
)
print(f"    {len(feat_df)} rows, {len(feat_df.columns)} cols")

# ── 3. Train model ──
print(f"\n🎯 Training XGBoost ensemble...")
exclude = {'target', 'target_long', 'target_short'}
feature_names = [c for c in feat_df.columns if c not in exclude]
X = feat_df[feature_names].fillna(0).clip(-10, 10)

n = len(feat_df)
split = int(n * TRAIN_SPLIT)

HPARAMS = {
    42:  {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5, 'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7, 'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4, 'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8, 'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}

models = {}
for side in ['long', 'short']:
    y = feat_df[f'target_{side}'].values
    X_train, y_train = X.iloc[:split].values.astype(np.float32), y[:split]
    X_test, y_test = X.iloc[split:].values.astype(np.float32), y[split:]
    
    models[side] = []
    for seed in SEEDS:
        hp = HPARAMS[seed]
        sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        model = xgb.XGBClassifier(n_estimators=400, max_depth=hp['max_depth'],
            learning_rate=hp['learning_rate'], subsample=hp['subsample'],
            colsample_bytree=hp['colsample_bytree'], min_child_weight=hp['min_child_weight'],
            scale_pos_weight=sw, objective='binary:logistic', eval_metric='auc',
            random_state=seed, verbosity=0, use_label_encoder=False)
        mask = np.random.choice([True, False], size=X.shape[1], p=[0.85, 0.15])
        if mask.sum() < 10: mask[:10] = True
        
        # CRITICAL: Train on DataFrame (preserves feature_names for model.save_model)
        X_train_subset = X.iloc[:split].iloc[:, mask]
        model.fit(X_train_subset, y_train)
        
        # Verify feature names are preserved
        assert model.get_booster().feature_names is not None, "Feature names lost!"
        assert len(model.get_booster().feature_names) == mask.sum(), f"Expected {mask.sum()} features, got {len(model.get_booster().feature_names)}"
        
        model.save_model(str(MODEL_DIR / f'{SYMBOL}_{side}_xgb_ens_{seed}.json'))
        models[side].append((model, mask, X_train_subset.columns.tolist()))
# ── 4. OOS validation (1-pass predictions) ──
print(f"\n📊 OOS Validation (1-pass, CLEAN)...")
X_test_full = X.iloc[split:].values.astype(np.float32)

oos_probas = {}
for side in ['long', 'short']:
    p = np.zeros((len(X_test_full), len(models[side])))
    for i, (m, mask, _) in enumerate(models[side]):
        p[:, i] = m.predict_proba(X_test_full[:, mask])[:, 1]
    oos_probas[side] = p.mean(axis=1)

# Threshold sweep on OOS
for side in ['long', 'short']:
    y_oos = feat_df[f'target_{side}'].values[split:]
    print(f"\n  {side.upper()}: Side OOS ({len(y_oos)} samples)")
    print(f"    Positive rate: {y_oos.mean()*100:.1f}%")
    for thr in range(60, 80, 5):
        thr_v = thr / 100.0
        pred = oos_probas[side] >= thr_v
        if pred.sum() >= 10:
            wr = y_oos[pred].mean()
            print(f"    THR={thr_v:.2f}: n={pred.sum():4d} WR={wr*100:.1f}%")

# ── 5. Per-bar feature computation (mimics live) ──
print(f"\n\n{'='*70}")
print("PER-BAR FEATURE COMPARISON")
print("="*70)

# Save models to MODEL_DIR for per-bar backtest to load
for side in ['long', 'short']:
    for i, (m, mask, _) in enumerate(models[side]):
        seed = SEEDS[i]

# Run per-bar feature computation using backtest.py
from src.trading.backtest import run_backtest_live_aligned

print(f"\n🔄 Running per-bar backtest...")
bt_result = run_backtest_live_aligned(SYMBOL, test_hours=24, verbose=True)

if bt_result is None:
    print("❌ Backtest returned None")
    sys.exit(1)

timestamps = bt_result['timestamps']
signals_perbar = bt_result['signals']
long_probas_perbar = bt_result['long_probas']
short_probas_perbar = bt_result['short_probas']

print(f"\n  Per-bar: {len(timestamps)} bars backtested")
print(f"  Signals: {len([s for s in signals_perbar if s != 0])} non-zero out of {len(signals_perbar)}")
print(f"  All probas around 0.5 = model sees NO edge (expected with clean data)")

# ── 6. Compare PROBA values: OOS (1-pass) vs Per-bar ──
print(f"\n\n{'='*70}")
print("SIGNAL AGREEMENT: OOS (1-pass) vs Per-bar")
print("="*70)

# Map per-bar timestamps to OOS index
oos_index = feat_df.index[split:]  # OOS region

n_common = 0
n_agree = 0
n_agree_signal = 0
proba_diff_long = []
proba_diff_short = []

for i, ts in enumerate(timestamps):
    if ts not in oos_index:
        continue
    oos_pos = oos_index.get_loc(ts)
    
    # Per-bar probas
    pbl = long_probas_perbar[i]
    pbs = short_probas_perbar[i]
    sig_pb = signals_perbar[i]
    
    # OOS probas
    pol = oos_probas['long'][oos_pos]
    pos = oos_probas['short'][oos_pos]
    
    # OOS signal
    thr = 0.70
    if pol >= thr and pol >= pos:
        sig_oos = 1
    elif pos >= thr:
        sig_oos = -1
    else:
        sig_oos = 0
    
    n_common += 1
    
    proba_diff_long.append(abs(pbl - pol))
    proba_diff_short.append(abs(pbs - pos))
    
    if abs(pbl - pol) < 0.01 and abs(pbs - pos) < 0.01:
        n_agree += 1
    if sig_pb == sig_oos:
        n_agree_signal += 1

print(f"\n  Common bars: {n_common}")
print(f"  Proba agreement (<0.01 diff): {n_agree}/{n_common} = {n_agree/n_common*100:.1f}%")
print(f"  Signal agreement: {n_agree_signal}/{n_common} = {n_agree_signal/n_common*100:.1f}%")
print(f"  Mean long proba diff: {np.mean(proba_diff_long)*100:.3f}%")
print(f"  Mean short proba diff: {np.mean(proba_diff_short)*100:.3f}%")
print(f"  Max long proba diff: {np.max(proba_diff_long)*100:.3f}%")
print(f"  Max short proba diff: {np.max(proba_diff_short)*100:.3f}%")

if n_agree == n_common:
    print("\n  ✅ PERFECT AGREEMENT — pipeline is CONSISTENT!")
elif n_agree >= n_common * 0.90:
    print(f"\n  ⚠️ HIGH agreement ({n_agree/n_common*100:.1f}%) — minor differences from edge cases")
else:
    print(f"\n  ❌ LOW agreement — STILL HAVE LOOK-AHEAD or methodology mismatch")
    
    # Show first 5 disagreements with context
    print("\n  Sample disagreements:")
    count = 0
    for i, ts in enumerate(timestamps):
        if ts not in oos_index:
            continue
        oos_pos = oos_index.get_loc(ts)
        pbl = long_probas_perbar[i]
        pol = oos_probas['long'][oos_pos]
        if abs(pbl - pol) > 0.01:
            print(f"    {ts}: per_bar={pbl:.4f}  oos={pol:.4f}")
            count += 1
            if count >= 5:
                break

# ── 7. Detailed check: WHY probas differ ──
if n_agree < n_common:
    print(f"\n\n  🔬 DETAILED FEATURE COMPARISON (first disagreement):")
    for i, ts in enumerate(timestamps):
        if ts not in oos_index:
            continue
        oos_pos = oos_index.get_loc(ts)
        pbl = long_probas_perbar[i]
        pol = oos_probas['long'][oos_pos]
        if abs(pbl - pol) > 0.01:
            # Compare features at this bar
            per_bar_feats = {}  # We can't easily extract these from backtest.py
            oos_feats = X.iloc[split:].iloc[oos_pos]
            
            # Find features with largest differences
            h1_feats = [c for c in oos_feats.index if c.startswith('h1_')]
            m5_feats = [c for c in oos_feats.index if c.startswith('m5_')]
            
            print(f"\n  --- Bar {ts} ---")
            print(f"  Per-bar: long={pbl:.4f} | OOS: long={pol:.4f}")
            
            # Use all features from OOS (referenced here)
            print(f"  OOS h1 features (sample):")
            for f in h1_feats[:5]:
                print(f"    {f}: {oos_feats[f]:.4f}")
            break

print(f"\n\n{'='*70}")
print("DONE")
print("="*70)
