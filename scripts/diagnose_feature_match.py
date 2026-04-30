#!/usr/bin/env python3
"""Compare feature values between OOS (1-pass with shift) and per-bar computation."""

import sys, os, warnings
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategies.ml_features import (
    ensure_ohlcv_data, ensure_ohlcv_1h, compute_5m_features_5tf,
    compute_technical_features,
)

SYMBOL = "BTCUSDT"
TRAIN_DAYS = 45

# 1. Fetch data
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
df_1h = ensure_ohlcv_1h(SYMBOL, min_days=max(TRAIN_DAYS, 20))

# 2. OOS 1-pass features (with 1h shift — training mode)
print("=== OOS 1-pass (training mode, 1h shifted) ===")
feat_oos = compute_5m_features_5tf(
    df_5m, target_candle=9, target_threshold=0.005,
    intervals=['1h'], for_inference=False, df_1h=df_1h
)
print(f"  Rows: {len(feat_oos)}, Cols: {len(feat_oos.columns)}")

# 3. Per-bar: manually compute features like the backtest
print("\n=== Per-bar computation ===")
feats_5m = compute_technical_features(df_5m, prefix='m5_')
full_1h_feats = compute_technical_features(df_1h, prefix='h1_')

# Pick a test bar
test_start = datetime.utcnow().replace(second=0, microsecond=0) - timedelta(hours=24)
test_idx = int(np.searchsorted(df_5m.index, test_start))

# Test multiple bars
print(f"  Test range: bars {test_idx} to {test_idx + 12}")
for bi in range(test_idx, test_idx + 12):
    bar_ts = df_5m.index[bi]
    
    # Per-bar: n_complete_1h by calendar time
    n_complete = int(np.searchsorted(
        df_1h.index + timedelta(hours=1), bar_ts, side='right'
    ))
    
    if n_complete < 1:
        continue
    
    # Per-bar 1h features (from pre-computed, slice last complete)
    feats_1h_bar = full_1h_feats.iloc[[n_complete - 1]].copy()
    feats_1h_bar.index = [bar_ts]
    
    # OOS features
    if bar_ts not in feat_oos.index:
        continue
    oos_row = feat_oos.loc[bar_ts]
    
    # Per-bar combined
    perbar_row = pd.concat([feats_5m.loc[[bar_ts]], feats_1h_bar], axis=1).iloc[0]
    
    # Compare h1 features
    h1_cols = [c for c in oos_row.index if c.startswith('h1_') and c in perbar_row.index]
    
    diffs = []
    for c in h1_cols:
        oos_val = oos_row[c] if pd.notna(oos_row[c]) else 0.0
        pb_val = perbar_row[c] if pd.notna(perbar_row[c]) else 0.0
        diff = abs(oos_val - pb_val)
        if diff > 0.001:
            diffs.append((c, oos_val, pb_val, diff))
    
    if bi == test_idx or diffs:
        print(f"\n  Bar {bi} ({bar_ts}): n_complete={n_complete}")
        for c, o, p, d in diffs[:10]:
            print(f"    {c}: OOS={o:.4f}  PerBar={p:.4f}  diff={d:.4f}")
        if not diffs:
            print(f"    ✅ All {len(h1_cols)} h1 features match!")
        else:
            print(f"    ⚠️ {len(diffs)}/{len(h1_cols)} h1 features differ")
    
    # Also compare m5 features
    m5_cols = [c for c in oos_row.index if c.startswith('m5_') and c in perbar_row.index]
    m5_diffs = []
    for c in m5_cols:
        oos_val = oos_row[c] if pd.notna(oos_row[c]) else 0.0
        pb_val = perbar_row[c] if pd.notna(perbar_row[c]) else 0.0
        diff = abs(oos_val - pb_val)
        if diff > 0.001:
            m5_diffs.append((c, oos_val, pb_val, diff))
    
    if m5_diffs:
        print(f"  ⚠️ {len(m5_diffs)}/{len(m5_cols)} m5 features differ")
        for c, o, p, d in m5_diffs[:5]:
            print(f"    {c}: OOS={o:.6f}  PerBar={p:.6f}  diff={d:.6f}")
    else:
        print(f"  ✅ All {len(m5_cols)} m5 features match!")
    
    # Compare ALL features
    all_common = [c for c in oos_row.index if c in perbar_row.index]
    all_diffs = []
    for c in all_common:
        oos_val = oos_row[c] if pd.notna(oos_row[c]) else 0.0
        pb_val = perbar_row[c] if pd.notna(perbar_row[c]) else 0.0
        diff = abs(oos_val - pb_val)
        if diff > 0.001:
            all_diffs.append((c, oos_val, pb_val, diff))
    
    if all_diffs:
        print(f"\n  TOTAL: {len(all_diffs)}/{len(all_common)} features differ (>{0.001})")
        # Group by prefix
        from collections import Counter
        prefixes = Counter(c.split('_')[0] for c, _, _, _ in all_diffs)
        print(f"  Diffs by prefix: {dict(prefixes)}")
        # Show top 5 differences by magnitude
        all_diffs.sort(key=lambda x: -x[3])
        print(f"  Top 5 diffs by magnitude:")
        for c, o, p, d in all_diffs[:5]:
            print(f"    {c}: OOS={o:.6f}  PerBar={p:.6f}  diff={d:.6f}")
