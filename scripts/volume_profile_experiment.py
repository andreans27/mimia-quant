#!/usr/bin/env python3
"""
Volume Profile / Market Profile experiment.
Features: VWAP, Delta/CVD, Volume Structure, Price-Volume confirmation.
Uses taker_buy data from Binance klines (already stored in cache).
"""

import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from pathlib import Path

from src.strategies.ml_features import ensure_ohlcv_data

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "1000PEPEUSDT"
TRAIN_DAYS = 45
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_SPLIT = 0.80

print(f"{'='*70}")
print(f"VOLUME PROFILE EXPERIMENT: {SYMBOL} | target={TARGET_BARS} bars")
print(f"{'='*70}")

# ─── 1. Fetch OHLCV with taker data ───
df_5m_raw = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)

print(f"\n📡 Data: {len(df_5m_raw)} bars ({df_5m_raw.index[0]} → {df_5m_raw.index[-1]})")
print(f"  Columns: {list(df_5m_raw.columns)}")

# Ensure we have taker data
has_taker = 'taker_buy_base' in df_5m_raw.columns
print(f"  Has taker data: {has_taker}")

if not has_taker:
    print("  ⚠️ No taker data — using volume-only features")

# ─── 2. Compute Volume Features ───

def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute VWAP at multiple periods."""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float)
    typical_price = (h + l + c) / 3
    
    feats = pd.DataFrame(index=df.index)
    
    # Daily VWAP
    daily_idx = df.index.date
    feats['vp_vwap_daily_num'] = (typical_price * v).groupby(daily_idx).cumsum()
    feats['vp_vwap_daily_den'] = v.groupby(daily_idx).cumsum()
    feats['vp_vwap_daily'] = feats['vp_vwap_daily_num'] / feats['vp_vwap_daily_den'].replace(0, np.nan)
    feats['vp_vwap_dev_daily'] = (c - feats['vp_vwap_daily']) / c  # % deviation from VWAP
    feats['vp_vwap_dev_z'] = feats['vp_vwap_dev_daily'] / feats['vp_vwap_dev_daily'].rolling(24).std().replace(0, np.nan)
    
    # Rolling VWAP (4h = 48 bars, 1h = 12 bars)
    for period, label in [(12, '1h'), (48, '4h'), (96, '8h')]:
        tpv = (typical_price * v).rolling(period).sum()
        tv = v.rolling(period).sum()
        feats[f'vp_vwap_{label}'] = tpv / tv.replace(0, np.nan)
        feats[f'vp_vwap_dev_{label}'] = (c - feats[f'vp_vwap_{label}']) / c
    
    # VWAP slope
    feats['vp_vwap_daily_slope'] = feats['vp_vwap_daily'].diff(12) / feats['vp_vwap_daily']  # 1h slope
    
    return feats


def compute_volume_structure(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features: spikes, trends, ratios."""
    v = df['volume'].astype(float)
    c = df['close'].astype(float)
    
    feats = pd.DataFrame(index=df.index)
    
    # Volume moving averages
    feats['vp_vol_sma_12'] = v.rolling(12).mean()
    feats['vp_vol_sma_24'] = v.rolling(24).mean()
    feats['vp_vol_sma_48'] = v.rolling(48).mean()
    
    # Volume ratios (current vs average)
    feats['vp_vol_ratio_12'] = v / feats['vp_vol_sma_12'].replace(0, np.nan)
    feats['vp_vol_ratio_24'] = v / feats['vp_vol_sma_24'].replace(0, np.nan)
    feats['vp_vol_ratio_48'] = v / feats['vp_vol_sma_48'].replace(0, np.nan)
    
    # Volume spike detection
    vol_std = v.rolling(24).std()
    vol_mean = v.rolling(24).mean()
    feats['vp_vol_zscore'] = (v - vol_mean) / vol_std.replace(0, np.nan)
    feats['vp_vol_spike'] = (feats['vp_vol_zscore'].abs() > 2).astype(float)
    
    # Volume trend (is volume expanding or contracting?)
    feats['vp_vol_trend_12'] = feats['vp_vol_sma_12'] / feats['vp_vol_sma_48'].replace(0, np.nan) - 1
    
    # Price-volume confirmation
    price_dir = np.sign(c.diff(1))
    vol_dir = np.sign(v.diff(1))
    feats['vp_pv_confirm'] = (price_dir * vol_dir).clip(lower=0)  # 1 if both same direction
    feats['vp_pv_diverg'] = ((-price_dir * vol_dir).clip(lower=0))  # 1 if opposite
    
    # Volume-weighted price change
    feats['vp_vw_ret_12'] = (c.pct_change(12) * v.rolling(12).sum()).rolling(12).mean()
    
    return feats


def compute_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Delta (buy-sell imbalance) and CVD features."""
    v = df['volume'].astype(float)
    c = df['close'].astype(float)
    
    feats = pd.DataFrame(index=df.index)
    
    if 'taker_buy_quote' in df.columns and 'quote_volume' in df.columns:
        tbb = df['taker_buy_quote'].astype(float)
        qv = df['quote_volume'].astype(float)
        # Delta = taker_buy_quote - taker_sell_quote = 2*taker_buy_quote - quote_volume
        # Using quote_volume (total quote volume) and taker_buy_quote (taker buy in quote)
        delta = 2 * tbb - qv
        feats['vp_delta'] = delta / qv.replace(0, np.nan)  # normalized delta
        feats['vp_delta_abs'] = delta.abs() / qv.replace(0, np.nan)
        
        # CVD — reset daily
        daily_idx = df.index.date
        feats['vp_cvd'] = delta.groupby(daily_idx).cumsum()
        feats['vp_cvd_norm'] = feats['vp_cvd'] / feats['vp_cvd'].rolling(48).std().replace(0, np.nan)
        
        # Delta divergence
        delta_trend = feats['vp_delta'].rolling(12).mean()
        price_trend = c.rolling(12).mean()
        feats['vp_delta_div_long'] = ((price_trend > price_trend.shift(6)) & (delta_trend < delta_trend.shift(6))).astype(float)
        feats['vp_delta_div_short'] = ((price_trend < price_trend.shift(6)) & (delta_trend > delta_trend.shift(6))).astype(float)
        
        # Buy/sell pressure ratio (using taker_buy_quote / quote_volume)
        feats['vp_buy_ratio'] = tbb / qv.replace(0, np.nan)
        feats['vp_buy_ratio_sma'] = feats['vp_buy_ratio'].rolling(12).mean()
        feats['vp_buy_ratio_z'] = (feats['vp_buy_ratio'] - feats['vp_buy_ratio'].rolling(24).mean()) / feats['vp_buy_ratio'].rolling(24).std().replace(0, np.nan)
        
        # Accumulation/distribution (using quote values)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        ad = ((c - l) - (h - c)) / (h - l).replace(0, np.nan) * qv
        feats['vp_ad_line'] = ad.rolling(48).sum()
        feats['vp_ad_z'] = (feats['vp_ad_line'] - feats['vp_ad_line'].rolling(24).mean()) / feats['vp_ad_line'].rolling(24).std().replace(0, np.nan)
        
        print(f"    Delta data: {tbb.notna().sum()}/{len(tbb)} rows have taker data")
    
    return feats


def compute_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Volume Profile: high/low volume nodes, value area."""
    v = df['volume'].astype(float)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    
    feats = pd.DataFrame(index=df.index)
    
    # Volume per price level (simplified: use rolling window)
    # Approximate HVN/LVN by looking at volume at current price vs recent average
    vol_at_price = v.copy()
    avg_vol = v.rolling(24).mean()
    feats['vp_vol_vs_avg'] = vol_at_price / avg_vol.replace(0, np.nan)  # > 1 = HVN level, < 1 = LVN level
    
    # Value Area: where 70% of volume occurred
    # Simplified: rolling volume distribution by price range
    price_range = h.rolling(24).max() - l.rolling(24).min()
    vol_per_range = v.rolling(24).sum() / price_range.replace(0, np.nan)
    feats['vp_vol_density'] = vol_per_range / vol_per_range.rolling(24).mean().replace(0, np.nan)
    
    # Volume exhaustion: high vol but small price movement
    price_move = abs(c.diff(1))
    vol_efficiency = price_move / v.replace(0, np.nan)
    feats['vp_vol_exhaust'] = (1 / vol_efficiency.rolling(24).mean()).replace(np.inf, np.nan)
    
    # Volume-weighted range
    vw_range = (v * (h - l)).rolling(24).sum() / v.rolling(24).sum().replace(0, np.nan)
    feats['vp_vw_range'] = vw_range / c
    
    return feats


# ─── 3. Compute all volume features ───
print(f"\n🔧 Computing volume profile features...")
vf_vwap = compute_vwap(df_5m_raw)
vf_vol = compute_volume_structure(df_5m_raw)
vf_delta = compute_delta_features(df_5m_raw)
vf_profile = compute_volume_profile(df_5m_raw)

print(f"  VWAP features: {len(vf_vwap.columns)}")
print(f"  Volume structure: {len(vf_vol.columns)}")
print(f"  Delta/CVD: {len(vf_delta.columns)}")
print(f"  Volume profile: {len(vf_profile.columns)}")

# ─── 4. Combine with baseline OHLCV features ───
# Compute baseline OHLCV features (price only, no volume)
c = df_5m_raw['close'].astype(float)
h = df_5m_raw['high'].astype(float)
l = df_5m_raw['low'].astype(float)

baseline_feats = pd.DataFrame(index=df_5m_raw.index)
for i in range(1, 4):
    baseline_feats[f'ret_{i}'] = c.pct_change(i)
    baseline_feats[f'log_ret_{i}'] = np.log(c / c.shift(i))
tr = pd.concat([h-l, abs(h-c.shift(1)), abs(l-c.shift(1))], axis=1).max(axis=1)
baseline_feats['atr_14'] = tr.rolling(14).mean()
delta = c.diff()
gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
ag, al = gain.rolling(14).mean(), loss.rolling(14).mean()
baseline_feats['rsi_14'] = 100 - (100 / (1 + ag / al.replace(0, np.nan)))
sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
baseline_feats['bb_width'] = (2*std20) / sma20
for p in [20, 50]:
    baseline_feats[f'close_div_sma{p}'] = c / c.rolling(p).mean() - 1

print(f"  Baseline (price): {len(baseline_feats.columns)} features")

# Combine
vol_feats_list = [vf_vwap, vf_vol, vf_delta, vf_profile]
vol_feats = pd.concat(vol_feats_list, axis=1)
combined = pd.concat([baseline_feats, vol_feats], axis=1)
combined = combined.dropna(how='all')
print(f"  TOTAL: {len(combined.columns)} features, {len(combined)} rows")

# ─── 5. Target ───
close_arr = c.reindex(combined.index)
target_long = np.zeros(len(combined), dtype=int)
target_short = np.zeros(len(combined), dtype=int)
for i in range(len(combined) - TARGET_BARS):
    fh = close_arr.iloc[i+1:i+TARGET_BARS+1].max()
    fl = close_arr.iloc[i+1:i+TARGET_BARS+1].min()
    target_long[i] = 1 if fh >= close_arr.iloc[i] * (1 + TARGET_THR) else 0
    target_short[i] = 1 if fl <= close_arr.iloc[i] * (1 - TARGET_THR) else 0

# ─── 6. Train: Baseline vs Volume Profile ───
n, split = len(combined), int(len(combined) * TRAIN_SPLIT)
feature_names = list(combined.columns)
X = combined[feature_names].fillna(0).clip(-10, 10)
baseline_cols = list(baseline_feats.columns)
vol_cols = list(vol_feats.columns)

HP = {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5}

print(f"\n{'='*70}")
print("COMPARISON: Baseline (price-only) vs +Volume Profile")
print(f"{'='*70}")
print(f"  Train: {split} rows | OOS: {n-split} rows")

all_results = {}
for exp_name, feat_cols in [("PRICE ONLY", baseline_cols), 
                              ("PRICE+VOLUME", feature_names)]:
    print(f"\n  ── {exp_name} ({len(feat_cols)} feats) ──")
    X_sub = X[feat_cols]
    results = []
    for side, y in [('long', target_long), ('short', target_short)]:
        yt, yo = y[:split], y[split:]
        sw = (len(yt) - yt.sum()) / max(yt.sum(), 1)
        model = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw,
                                   objective='binary:logistic', random_state=42, verbosity=0)
        model.fit(X_sub.iloc[:split], yt)
        p = model.predict_proba(X_sub.iloc[split:].values.astype(np.float32))[:, 1]
        auc = roc_auc_score(yo, p) if len(set(yo)) > 1 else 0.0
        
        best_wr, best_thr, best_n = 0, '-', 0
        for td in range(50, 90):
            thr = td / 100.0
            pred = p >= thr
            nt = int(pred.sum())
            if nt >= 5:
                wr = float(yo[pred].mean())
                if wr > best_wr:
                    best_wr, best_thr, best_n = wr, f'{thr:.2f}', nt
        
        print(f"    {side:>5s}: AUC={auc:.3f} WR={best_wr*100:.1f}% @ THR={best_thr} (n={best_n})")
        results.append({'side': side, 'auc': round(auc, 4), 'wr': round(best_wr*100, 1),
                        'thr': best_thr, 'n': best_n, 'oos_pos': round(yo.mean()*100, 1)})
    all_results[exp_name] = results

# ─── 7. Feature importance (Volume Profile features in Full model) ───
print(f"\n{'='*70}")
print("VOLUME PROFILE FEATURE IMPORTANCE (Full Model)")
print('='*70)
for side, y in [('long', target_long), ('short', target_short)]:
    yt = y[:split]
    sw = (len(yt)-yt.sum())/max(yt.sum(), 1)
    model = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw,
                               objective='binary:logistic', random_state=42, verbosity=0)
    model.fit(X.iloc[:split], yt)
    imp = pd.DataFrame({'feat': feature_names, 'imp': model.feature_importances_})
    imp = imp.sort_values('imp', ascending=False)
    
    print(f"\n  {side.upper()} — Volume features in top 30:")
    vol_imp = imp[imp['feat'].isin(vol_cols)]
    for _, r in vol_imp.head(15).iterrows():
        rank = imp[imp['feat'] == r['feat']].index[0] + 1
        print(f"    #{rank:2d} {r['feat']:<30s} {r['imp']:.4f}")
    print(f"    ... ({len(vol_imp)} total volume features)")

# ─── Summary ───
print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for exp, res in all_results.items():
    print(f"\n  {exp}:")
    for r in res:
        print(f"    {r['side']:>5s}: AUC={r['auc']:.3f} WR={r['wr']:.1f}% @ THR={r['thr']} (n={r['n']})")

Path('data/volume_profile_experiment.json').write_text(json.dumps(all_results, indent=2))
print(f"\n✅ Saved to data/volume_profile_experiment.json")
