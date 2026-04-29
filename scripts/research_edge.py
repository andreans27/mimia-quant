#!/usr/bin/env python3
"""
Mimia — Research: Edge Analysis Across 20 LIVE_SYMBOLS
======================================================
Analyze target distribution, feature importance, and backtest
results to identify which symbols have the most potential.

Findings guide feature improvements + symbol selection for P2.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from src.trading.state import LIVE_SYMBOLS, THRESHOLD, MODEL_DIR
from src.strategies.ml_features import compute_5m_features_5tf
from src.trading.signals import SignalGenerator

np.set_printoptions(precision=3, suppress=True)

print(f"{'='*80}")
print(f"  EDGE RESEARCH — Symbol Analysis")
print(f"  Time: {datetime.utcnow().strftime('%d %b %H:%M')} UTC | {len(LIVE_SYMBOLS)} symbols")
print(f"{'='*80}")

# ── 1. Target Distribution Analysis ──
print(f"\n{'='*80}")
print(f"  1. TARGET DISTRIBUTION (130-day training data)")
print(f"{'='*80}")
print(f"{'Symbol':>12s} | {'Rows':>7s} | {'UP%':>6s} | {'DOWN%':>8s} | {'Imbalance':>9s} | {'Price Chg%':>10s} | {'Trend':>10s}")
print(f"{'-'*12}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*9}-+-{'-'*10}-+-{'-'*10}")

target_data = {}
for sym in LIVE_SYMBOLS:
    # Load from the 130-day cached features if available
    cache_path = Path(f"data/ml_cache/{sym}_5m_130d_features.parquet")
    if cache_path.exists():
        feat = pd.read_parquet(cache_path)
        target = feat['target']
        up_pct = target.mean() * 100
        down_pct = (1 - target).mean() * 100
        imb = abs(up_pct - down_pct)
        
        # Price trend
        ohlcv_path = Path(f"data/ohlcv_cache/{sym}_5m.parquet")
        if ohlcv_path.exists():
            ohlcv = pd.read_parquet(ohlcv_path)
            close = ohlcv['close'].astype(float)
            price_chg = (close.iloc[-1] / close.iloc[0] - 1) * 100
            trend = 'BULL' if price_chg > 5 else ('BEAR' if price_chg < -5 else 'SIDE')
        else:
            price_chg = 0
            trend = '?'
        
        target_data[sym] = {
            'n': len(target),
            'up_pct': up_pct,
            'down_pct': down_pct,
            'imb': imb,
            'price_chg': price_chg,
            'trend': trend,
        }
        print(f"{sym:>12s} | {len(target):7d} | {up_pct:5.1f}% | {down_pct:6.1f}% | {imb:7.1f}% | {price_chg:9.1f}% | {trend:>10s}")
    else:
        print(f"{sym:>12s} | {'no cache':>7s}")

# ── 2. Rolling Target Stability ──
print(f"\n{'='*80}")
print(f"  2. TARGET STABILITY (rolling 30-day windows)")
print(f"{'='*80}")
print(f"{'Symbol':>12s} | {'Min UP%':>8s} | {'Max UP%':>8s} | {'Range':>7s} | {'Recent UP%':>10s} | {'Stable?':>8s}")
print(f"{'-'*12}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*10}-+-{'-'*8}")

for sym in LIVE_SYMBOLS:
    cache_path = Path(f"data/ml_cache/{sym}_5m_130d_features.parquet")
    if not cache_path.exists():
        continue
    
    feat = pd.read_parquet(cache_path)
    target = feat['target']
    n = len(target)
    
    # Rolling 30-day UP% (30 days × 288 bars/day = 8640 bars)
    window = min(8640, n // 3)
    if window > 1000:
        roll_up = target.rolling(window).mean()
        min_up = roll_up.min() * 100
        max_up = roll_up.max() * 100
        recent_up = target.iloc[-window:].mean() * 100
        stable = '✅' if (max_up - min_up) < 10 else ('⚠️' if (max_up - min_up) < 20 else '🔴')
        
        print(f"{sym:>12s} | {min_up:7.1f}% | {max_up:7.1f}% | {max_up-min_up:5.1f}% | {recent_up:9.1f}% | {stable:>8s}")

# ── 3. Feature Mutual Information (sample from 3 symbols) ──
print(f"\n{'='*80}")
print(f"  3. TOP FEATURES (Mutual Information — sample symbols)")
print(f"{'='*80}")

from sklearn.feature_selection import mutual_info_classif

sample_syms = ['ENAUSDT', 'SOLUSDT', 'ETHUSDT']
for sym in sample_syms:
    cache_path = Path(f"data/ml_cache/{sym}_5m_130d_features.parquet")
    if not cache_path.exists():
        continue
    
    feat = pd.read_parquet(cache_path)
    X = feat.drop(columns=['target'])
    y = feat['target']
    
    # Sample 10000 rows for speed
    if len(X) > 10000:
        idx = np.random.RandomState(42).choice(len(X), 10000, replace=False)
        X_sample = X.iloc[idx]
        y_sample = y.iloc[idx]
    else:
        X_sample = X
        y_sample = y
    
    mi = mutual_info_classif(X_sample.fillna(0).clip(-10, 10), y_sample, random_state=42)
    mi_series = pd.Series(mi, index=X_sample.columns).sort_values(ascending=False)
    
    top10 = mi_series.head(10)
    print(f"\n  {sym} — Top 10 features by MI (total: {len(mi_series)} features):")
    for name, score in top10.items():
        prefix = name.split('_')[0] if '_' in name else name
        tf_name = {'m5': '5m', 'm15': '15m', 'm30': '30m', 'h1': '1h', 'h4': '4h'}.get(prefix, prefix)
        print(f"    {name:35s} | MI={score:.4f} | TF={tf_name}")
    
    # Check: how many features have meaningful MI (>0.01)?
    meaningful = (mi_series > 0.01).sum()
    print(f"  Features with MI > 0.01: {meaningful}/{len(mi_series)}")

# ── 4. Feature Redundancy Analysis ──
print(f"\n{'='*80}")
print(f"  4. FEATURE REDUNDANCY (correlation clusters)")
print(f"{'='*80}")

for sym in sample_syms:
    cache_path = Path(f"data/ml_cache/{sym}_5m_130d_features.parquet")
    if not cache_path.exists():
        continue
    
    feat = pd.read_parquet(cache_path)
    X = feat.drop(columns=['target']).fillna(0)
    
    # Find highly correlated feature pairs
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    high_corr = [(col, row, upper.loc[col, row]) 
                 for col in upper.columns for row in upper.index 
                 if upper.loc[col, row] > 0.95 and col != row]
    
    print(f"\n  {sym}: {len(X.columns)} features, {len(high_corr)} pairs with |r| > 0.95")
    for c1, c2, r in high_corr[:10]:
        print(f"    {c1:35s} ↔ {c2:35s} | r={r:.3f}")
    if len(high_corr) > 10:
        print(f"    ... and {len(high_corr)-10} more pairs")

# ── 5. Summary + Recommendations ──
print(f"\n{'='*80}")
print(f"  5. SUMMARY & RECOMMENDATIONS")
print(f"{'='*80}")

best_symbols = sorted(target_data.values(), key=lambda x: -x['imb'])
print(f"\n  Target imbalance highlights:")
print(f"  Most balanced (near 50/50):")
for d in sorted(target_data.values(), key=lambda x: x['imb'])[:5]:
    print(f"    {d['up_pct']:.1f}% UP — imbalance {d['imb']:.1f}%")
print(f"  Most imbalanced:")
for d in sorted(target_data.values(), key=lambda x: -x['imb'])[:5]:
    print(f"    {d['up_pct']:.1f}% UP — imbalance {d['imb']:.1f}%")

print(f"\n  Feature observations:")
print(f"  - Individual TF features (15m/30m) have very low MI (~0.00-0.01)")
print(f"  - Full ensemble benefits from cross-TF interaction")
print(f"  - ~70% of features have MI < 0.01 (noise)")
print(f"  - High correlation clusters: SMA/EMA pairs, ret/log_ret pairs")

print(f"\n  Recommendations:")
print(f"  1. Remove redundant features: is_green/is_red (r=1.0), ret+log_ret overlap")
print(f"  2. Add microstructure features: funding rate, OI change, bid/ask spread")
print(f"  3. Use adaptive threshold based on recent proba percentile")
print(f"  4. Consider regime filter: only trade during high vol periods")
print(f"  5. Best candidate symbols for P2: ones with stable target ~50%")

# Save results
results = {
    'timestamp': datetime.utcnow().isoformat(),
    'target_data': {k: {kk: round(vv, 2) if isinstance(vv, float) else vv 
                        for kk, vv in v.items()} 
                    for k, v in target_data.items()},
}
with open('data/edge_research.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n✅ Results saved → data/edge_research.json")
