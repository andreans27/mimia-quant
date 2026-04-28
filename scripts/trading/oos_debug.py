#!/usr/bin/env python3
"""Debug: check model feature_names vs DataFrame columns."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
import xgboost as xgb
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.utils.binance_client import BinanceRESTClient
from src.strategies.ml_features import compute_5m_features_5tf, _fetch_all_klines

model_dir = Path("data/oos_test/models")
sym = "BTCUSDT"

# Load meta to get expected feature count
with open(model_dir / f"{sym}_ensemble_meta.json") as f:
    meta = json.load(f)
meta_features = meta.get('features', [])
print(f"Meta has {len(meta_features)} features")
print(f"Sample: {meta_features[:3]}")

# Load model
m = xgb.XGBClassifier()
path = model_dir / f"{sym}_xgb_ens_42.json"
m.load_model(str(path))
mf = m.get_booster().feature_names
print(f"Model feature_names: {len(mf) if mf else 'EMPTY'}")

# Quick feature compute
client = BinanceRESTClient(testnet=True)
end = datetime.now()
start = end - timedelta(days=60)
df_5m = _fetch_all_klines(client, sym, "5m", int(start.timestamp()*1000), int(end.timestamp()*1000))
feat_df = compute_5m_features_5tf(df_5m, target_candle=9)
feature_cols = [c for c in feat_df.columns if c != 'target']
print(f"\nFeature DF has {len(feature_cols)} columns")

# Cross-check meta features vs DF columns
meta_set = set(meta_features)
cols_set = set(feat_df.columns)

common = cols_set & meta_set
missing = meta_set - cols_set
extra = cols_set - meta_set - {'target'}

print(f"\nMeta vs DF:")
print(f"  Common: {len(common)}")
print(f"  Missing (in meta but not in DF): {len(missing)}")
if missing:
    print(f"  Examples: {list(missing)[:5]}")
print(f"  Extra (in DF but not in meta): {len(extra)}")
if extra:
    print(f"  Examples: {list(extra)[:5]}")

# Simulate what compute_proba does
available = [c for c in meta_features if c in feat_df.columns]
print(f"\nAvailable for predict: {len(available)}")
X = feat_df[available].fillna(0).clip(-10, 10).values
print(f"X shape: {X.shape}")
print(f"Model expects 371 features")

if X.shape[1] == 371:
    print("✅ Shape match! Trying predict...")
    probs = m.predict_proba(X[:5])[:, 1]
    print(f"Sample probs: {probs}")
else:
    print(f"❌ MISMATCH: got {X.shape[1]}, expected 371")
