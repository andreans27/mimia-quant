#!/usr/bin/env python3
"""Trace EXACTLY what features the per-bar backtest passes to the model."""

import sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir('/root/projects/mimia-quant')
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
from src.trading.state import MODEL_DIR, SEEDS
from src.trading.signals import SignalGenerator

SYMBOL = "BTCUSDT"

# 1. Get data
df_5m = ensure_ohlcv_data(SYMBOL, min_days=45)
df_1h = ensure_ohlcv_1h(SYMBOL, min_days=45)

# 2. OOS features (clean)
feat_oos = compute_5m_features_5tf(
    df_5m, target_candle=9, target_threshold=0.005,
    intervals=['1h'], for_inference=False, df_1h=df_1h
)
feature_names = [c for c in feat_oos.columns if c not in ('target','target_long','target_short')]
X = feat_oos[feature_names].fillna(0).clip(-10, 10)
n = len(feat_oos); split = int(n * 0.80)

# 3. Pre-compute per-bar features (same as backtest)
feats_5m = compute_technical_features(df_5m, prefix='m5_')
full_1h_feats = compute_technical_features(df_1h, prefix='h1_')

# 4. Load models as backtest does
gen = SignalGenerator(SYMBOL)
cached = gen._load_models(SYMBOL)
if cached is None:
    print("FAIL: No models")
    sys.exit(1)
model_groups = cached['groups']

# 5. Build model feature index
all_model_feats = sorted(set(
    f for _, models in model_groups.items()
    for _, m, mf in models for f in mf
))
mf_to_idx = {f: i for i, f in enumerate(all_model_feats)}
n_mf = len(all_model_feats)

long_refs = []
short_refs = []
for tg, models in model_groups.items():
    for seed, m, mf in models:
        avail = [mf_to_idx[f] for f in mf if f in mf_to_idx]
        if len(avail) >= 5:
            arr = np.array(avail, dtype=np.int32)
            ref = (m, arr)
            if tg == 'long':
                long_refs.append(ref)
            elif tg == 'short':
                short_refs.append(ref)

print(f"Long models: {len(long_refs)}")
print(f"Short models: {len(short_refs)}")
print(f"All model features: {n_mf}")

# 6. Pick test bar
test_start = datetime.utcnow().replace(second=0, microsecond=0) - timedelta(hours=24)
test_idx = int(np.searchsorted(df_5m.index, test_start))
bar_ts = df_5m.index[test_idx]

# 7. Compute per-bar features EXACTLY as backtest.py does now
n_complete_1h = int(np.searchsorted(
    df_1h.index + timedelta(hours=1),
    bar_ts,
    side='right'
))
if n_complete_1h < 1:
    print("FAIL: No complete 1h bars")
    sys.exit(1)

feats_1h_aligned = full_1h_feats.iloc[[n_complete_1h - 1]].copy()
feats_1h_aligned.index = [bar_ts]

combined = pd.concat([feats_5m.loc[[bar_ts]], feats_1h_aligned], axis=1)

# 8. Build model input
feat_cols = list(combined.columns)
row_np = np.zeros(n_mf, dtype=np.float64)
for mf_name, mf_pos in mf_to_idx.items():
    if mf_name in feat_cols:
        val = combined[mf_name].values[0]
        row_np[mf_pos] = float(np.clip(0.0 if np.isnan(val) else val, -10, 10))

# 9. Compare with OOS row
oos_row = X.iloc[split:].iloc[0]  # first OOS row
# Find this bar's position in OOS
if bar_ts in X.index:
    oos_pos = X.index.get_loc(bar_ts)
    if oos_pos >= split:
        rel_pos = oos_pos - split
        oos_row = X.iloc[oos_pos]
        
        print(f"\nBar: {bar_ts} (test_idx={test_idx}, oos_pos={oos_pos}, rel_pos={rel_pos})")
        print(f"n_complete_1h={n_complete_1h}")
        
        # Compare the model input vectors
        # First, figure out what the OOS model uses
        # The OOS models use mask-based selection. We don't have masks because
        # we loaded from disk. But we can compare features by NAME.
        
        mismatches = []
        for mf_name in all_model_feats:
            mf_pos = mf_to_idx[mf_name]
            pb_val = row_np[mf_pos]
            
            # OOS value: from X (which is feat_oos clipped)
            if mf_name in feature_names:
                oos_raw = float(feat_oos.loc[bar_ts, mf_name]) if mf_name in feat_oos.columns else 0.0
                oos_val = np.clip(0.0 if np.isnan(oos_raw) else oos_raw, -10, 10)
            else:
                oos_val = 0.0
            
            if abs(pb_val - oos_val) > 0.001:
                mismatches.append((mf_name, oos_val, pb_val))
        
        if mismatches:
            print(f"  MISMATCHES: {len(mismatches)}/{n_mf}")
            mismatches.sort(key=lambda x: -abs(x[1]-x[2]))
            for c, o, p in mismatches[:10]:
                print(f"    {c}: OOS={o:.6f}  PerBar={p:.6f}  diff={abs(o-p):.6f}")
        else:
            print(f"  ✅ ALL {n_mf} model features MATCH!")
        
        # Now compare predictions
        def _infer(refs, feat_row_np):
            probas = []
            for model, feat_idx in refs:
                preds = model.predict_proba(feat_row_np[feat_idx].reshape(1, -1))[:, 1]
                probas.append(preds[0])
            return float(np.mean(probas)) if probas else 0.5
        
        pb_long = _infer(long_refs, row_np)
        pb_short = _infer(short_refs, row_np)
        
        print(f"\n  Per-bar prediction: long={pb_long:.4f} short={pb_short:.4f}")
else:
    print(f"Bar {bar_ts} not found in OOS index")
