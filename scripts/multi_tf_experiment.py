#!/usr/bin/env python3
"""
Multi-TF experiment: 5m + 15m + 30m + 1h with +1 bar shift.
Uses focused price-action features + cross-TF alignment.
All timeframes fetched DIRECTLY from Binance (no resample look-ahead).
"""

import sys, os, warnings, json, time
warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)) + '/..')
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from datetime import datetime, timedelta
from pathlib import Path

from src.strategies.ml_features import ensure_ohlcv_data
from src.trading.state import MODEL_DIR

# ─── Generic OHLCV fetcher (direct from Binance) ───
def fetch_ohlcv_tf(symbol: str, interval: str, min_days: int = 45):
    """Fetch klines for any timeframe DIRECTLY from Binance Futures API."""
    cache_dir = Path("data/ohlcv_cache")
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"
    
    # Check cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        interval_hours = {'15m': 0.25, '30m': 0.5, '1h': 1}[interval]
        min_bars = int(min_days * 24 / interval_hours)
        if len(df) >= min_bars:
            return df
    
    import requests
    url = "https://fapi.binance.com/fapi/v1/klines"
    end = datetime.now()
    start = end - timedelta(days=min_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    limit = 1000
    all_bars = []
    last_ts = start_ms
    
    while last_ts < end_ms:
        params = {'symbol': symbol, 'interval': interval, 'limit': limit,
                  'startTime': last_ts, 'endTime': end_ms}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_bars.extend(batch)
            last_ts = batch[-1][0] + 1
            if len(batch) < limit:
                break
        except Exception:
            break
    
    if len(all_bars) < 100:
        return None
    
    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    
    result = df[['open', 'high', 'low', 'close', 'volume']]
    result.to_parquet(cache_path)
    print(f"  💾 Cached {len(result)} {interval} bars → {cache_path}")
    return result


# ─── Core Price Features (~25 per TF) ───
def compute_price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Compute focused price-action features. NO market data, NO micro features."""
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    
    feats = pd.DataFrame(index=df.index)
    
    # Returns (1-5 bars)
    for i in range(1, 6):
        feats[f'{prefix}ret_{i}'] = c.pct_change(i)
        feats[f'{prefix}log_ret_{i}'] = np.log(c / c.shift(i))
    
    # Volatility
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    feats[f'{prefix}atr_14'] = tr.rolling(14).mean()
    feats[f'{prefix}atr_ratio'] = feats[f'{prefix}atr_14'] / c.rolling(14).mean()
    feats[f'{prefix}vol_20'] = c.rolling(20).std() / c.rolling(20).mean()
    
    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.rolling(14).mean()
    avg_l = loss.rolling(14).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    feats[f'{prefix}rsi_14'] = 100 - (100 / (1 + rs))
    
    # ADX (trend strength)
    tr_14 = tr.rolling(14).mean()
    dm_plus = (h - h.shift(1)).clip(lower=0)
    dm_minus = (l.shift(1) - l).clip(lower=0)
    di_plus = 100 * (dm_plus.rolling(14).mean() / tr_14.replace(0, np.nan))
    di_minus = 100 * (dm_minus.rolling(14).mean() / tr_14.replace(0, np.nan))
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus).replace(0, np.nan)
    feats[f'{prefix}adx_14'] = dx.rolling(14).mean()
    
    # Bollinger Bands
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats[f'{prefix}bb_width_20'] = (2 * std20) / sma20
    feats[f'{prefix}bb_pct_20'] = (c - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
    
    # SMA relationships
    for period in [20, 50]:
        sma = c.rolling(period).mean()
        feats[f'{prefix}sma_{period}'] = sma
        feats[f'{prefix}close_div_sma{period}'] = c / sma - 1
        feats[f'{prefix}price_vs_sma{period}'] = (c - sma) / c.rolling(period).std().replace(0, np.nan)
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    feats[f'{prefix}macd'] = macd / c
    feats[f'{prefix}macd_signal'] = signal / c
    feats[f'{prefix}macd_hist'] = (macd - signal) / c
    
    # Price range position
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    range_20 = high_20 - low_20
    feats[f'{prefix}range_20'] = range_20 / c
    feats[f'{prefix}pos_in_range_20'] = (c - low_20) / range_20.replace(0, np.nan)
    
    # Momentum features
    feats[f'{prefix}roc_10'] = c.pct_change(10)
    feats[f'{prefix}roc_20'] = c.pct_change(20)
    
    return feats


# ─── Cross-TF Alignment Features ───
def compute_cross_tf_features(aligned: dict) -> pd.DataFrame:
    """Compute alignment features between timeframes."""
    reference = list(aligned.keys())[0]
    idx = aligned[reference].index
    
    feats = pd.DataFrame(index=idx)
    tf_names = sorted(aligned.keys())
    
    # RSI divergence between consecutive TFs
    for i in range(len(tf_names) - 1):
        tf1, tf2 = tf_names[i], tf_names[i+1]
        p1, p2 = f'tf_{tf1}_', f'tf_{tf2}_'
        
        rsi1 = aligned[tf1].get(f'{p1}rsi_14')
        rsi2 = aligned[tf2].get(f'{p2}rsi_14')
        if rsi1 is not None and rsi2 is not None:
            feats[f'rsi_div_{tf1}_vs_{tf2}'] = rsi1 - rsi2
        
        adx1 = aligned[tf1].get(f'{p1}adx_14')
        adx2 = aligned[tf2].get(f'{p2}adx_14')
        if adx1 is not None and adx2 is not None:
            feats[f'adx_{tf1}_vs_{tf2}'] = adx1 - adx2
        
        # Trend alignment: sign(close - SMA20)
        c1 = aligned[tf1].get(f'{p1}close_div_sma20')
        c2 = aligned[tf2].get(f'{p2}close_div_sma20')
        if c1 is not None and c2 is not None:
            feats[f'trend_align_{tf1}_vs_{tf2}'] = np.sign(c1) * np.sign(c2)
        
        # Volatility ratio
        v1 = aligned[tf1].get(f'{p1}atr_ratio')
        v2 = aligned[tf2].get(f'{p2}atr_ratio')
        if v1 is not None and v2 is not None:
            feats[f'vol_ratio_{tf1}_vs_{tf2}'] = v1 / v2.replace(0, np.nan)
    
    # Regime: which TF has strongest trend (highest ADX)
    adx_cols = {}
    for tf in tf_names:
        p = f'tf_{tf}_'
        a = aligned[tf].get(f'{p}adx_14')
        if a is not None:
            adx_cols[tf] = a
    
    if len(adx_cols) >= 2:
        adx_df = pd.DataFrame(adx_cols)
        # Handle NaN: only compute where all ADX values are valid
        valid_mask = adx_df.notna().all(axis=1)
        feats['max_adx'] = adx_df.max(axis=1)
        feats['min_adx'] = adx_df.min(axis=1)
        feats['adx_spread'] = adx_df.max(axis=1) - adx_df.min(axis=1)
        # Best TF as numeric: 0=first, 1=second, 2=third
        if valid_mask.any():
            dominant = adx_df[valid_mask].idxmax(axis=1)
            tf_order = sorted(adx_cols.keys())
            tf_to_num = {tf: i for i, tf in enumerate(tf_order)}
            feats['dominant_tf_num'] = np.nan
            feats.loc[valid_mask, 'dominant_tf_num'] = dominant.map(tf_to_num)
    
    # Momentum harmony: MACD hist signs across TFs
    macd_cols = {}
    for tf in tf_names:
        p = f'tf_{tf}_'
        m = aligned[tf].get(f'{p}macd_hist')
        if m is not None:
            macd_cols[tf] = np.sign(m)
    
    if len(macd_cols) >= 2:
        macd_df = pd.DataFrame(macd_cols)
        feats['macd_unanimity'] = macd_df.sum(axis=1) / len(macd_cols)
        feats['macd_consensus'] = (macd_df.sum(axis=1).abs() >= len(macd_cols) - 1).astype(float)
    
    # BB width ranking (which TF most volatile)
    bb_cols = {}
    for tf in tf_names:
        p = f'tf_{tf}_'
        b = aligned[tf].get(f'{p}bb_width_20')
        if b is not None:
            bb_cols[tf] = b
    if len(bb_cols) >= 2:
        bb_df = pd.DataFrame(bb_cols)
        feats['bb_width_spread'] = bb_df.max(axis=1) - bb_df.min(axis=1)
        # Most volatile TF as numeric
        valid_mask = bb_df.notna().all(axis=1)
        feats['bb_width_max_tf_num'] = np.nan
        if valid_mask.any():
            dominant = bb_df[valid_mask].idxmax(axis=1)
            tf_order = sorted(bb_cols.keys())
            tf_to_num = {tf: i for i, tf in enumerate(tf_order)}
            feats.loc[valid_mask, 'bb_width_max_tf_num'] = dominant.map(tf_to_num)
    
    return feats


# ─── Config ───
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "SOLUSDT"
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_DAYS = 45
TRAIN_SPLIT = 0.80

HPARAMS = {42: {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5}}

print(f"{'='*70}")
print(f"MULTI-TF EXPERIMENT: {SYMBOL} | target={TARGET_BARS} bars ({TARGET_BARS*5}m)")
print(f"{'='*70}")

# ── 1. Fetch all timeframes ──
print(f"\n📡 Fetching {TRAIN_DAYS} days...")
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
if df_5m is None: sys.exit(1)

tfs = {'15m': 0.25, '30m': 0.5, '1h': 1}
extra_data = {}
for tf_name, hours in tfs.items():
    print(f"  {tf_name}...")
    df = fetch_ohlcv_tf(SYMBOL, tf_name, min_days=TRAIN_DAYS)
    if df is None:
        print(f"    ❌ No data for {tf_name}")
        continue
    extra_data[tf_name] = df
    print(f"    ✅ {len(df)} bars")

if len(extra_data) < 2:
    print("❌ Not enough timeframe data")
    sys.exit(1)

# ── 2. Compute price features per TF with +1 bar shift ──
print(f"\n🔧 Computing features...")
shift_map = {'15m': 1, '30m': 1, '1h': 1}  # +1 bar shift for each
fill_limit = {'15m': 3, '30m': 6, '1h': 12}
tf_feats = {}

for tf_name, df_tf in extra_data.items():
    prefix = f'tf_{tf_name}_'
    raw_feats = compute_price_features(df_tf, prefix)
    
    # Shift by 1 bar
    shifted = raw_feats.shift(shift_map[tf_name])
    
    # Forward-fill to 5m index
    aligned = shifted.reindex(df_5m.index, method='ffill', limit=fill_limit[tf_name])
    tf_feats[tf_name] = aligned
    n_feats = len([c for c in aligned.columns if c.startswith(prefix)])
    print(f"  {tf_name}: {n_feats} features (shift +{shift_map[tf_name]} bar)")

# ── 3. 5m price features (no shift) ──
feats_5m = compute_price_features(df_5m, 'tf_5m_')
print(f"  5m: {len([c for c in feats_5m.columns if c.startswith('tf_5m_')])} features (no shift)")

# ── 4. Cross-TF features ──
cross_feats = compute_cross_tf_features(tf_feats)
print(f"  Cross-TF: {len(cross_feats.columns)} features")

# ── 5. Combine all features ──
all_feats_list = [feats_5m] + list(tf_feats.values()) + [cross_feats]
combined = pd.concat(all_feats_list, axis=1)

# Drop NaN rows (warmup periods)
combined = combined.dropna(how='all')

print(f"\n  Total features: {len(combined.columns)}")
print(f"  Total rows: {len(combined)}")

# ── 6. Target ──
close = df_5m['close'].astype(float).reindex(combined.index)
target_long = np.zeros(len(combined), dtype=int)
target_short = np.zeros(len(combined), dtype=int)
for i in range(len(combined) - TARGET_BARS):
    future_high = close.iloc[i+1:i+TARGET_BARS+1].max()
    future_low = close.iloc[i+1:i+TARGET_BARS+1].min()
    target_long[i] = 1 if future_high >= close.iloc[i] * (1 + TARGET_THR) else 0
    target_short[i] = 1 if future_low <= close.iloc[i] * (1 - TARGET_THR) else 0

combined['target_long'] = target_long
combined['target_short'] = target_short

# ── 7. Train & evaluate ──
print(f"\n{'='*70}")
print(f"TRAINING & EVALUATION")
print(f"{'='*70}")

feature_names = [c for c in combined.columns if not c.startswith('target_')]
X = combined[feature_names].fillna(0).clip(-10, 10)
n = len(combined)
split = int(n * TRAIN_SPLIT)

results = []
for side in ['long', 'short']:
    y = combined[f'target_{side}'].values
    y_train, y_oos = y[:split], y[split:]
    pos_rate = y.mean() * 100
    oos_pos = y_oos.mean() * 100
    
    sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    
    model = xgb.XGBClassifier(n_estimators=400, **HPARAMS[42],
        scale_pos_weight=sw, objective='binary:logistic',
        random_state=42, verbosity=0)
    model.fit(X.iloc[:split], y_train)
    
    # Predict
    p = model.predict_proba(X.iloc[split:].values.astype(np.float32))[:, 1]
    
    try:
        auc = roc_auc_score(y_oos, p)
    except:
        auc = 0.0
    
    print(f"\n  {side.upper()} ({TARGET_BARS*5}m target):")
    print(f"    Positive rate: train={pos_rate:.1f}% OOS={oos_pos:.1f}%")
    print(f"    AUC: {auc:.4f}")
    
    # Threshold sweep
    best_wr, best_thr, best_n = 0, '-', 0
    for thr_dec in range(50, 90):
        thr = thr_dec / 100.0
        pred = p >= thr
        nt = int(pred.sum())
        if nt >= 5:
            wr = float(y_oos[pred].mean())
            if wr > best_wr:
                best_wr, best_thr, best_n = wr, f'{thr:.2f}', nt
    
    if best_thr != '-':
        print(f"    Best WR: {best_wr*100:.1f}% @ THR={best_thr} (n={best_n})")
    else:
        print(f"    No trades at any threshold")
    
    results.append({'side': side, 'auc': round(auc, 4), 'best_wr': round(best_wr*100, 1),
                    'best_thr': best_thr, 'best_n': best_n, 'oos_pos': round(oos_pos, 1)})

# ── 8. Feature importance (top 20) ──
print(f"\n{'='*70}")
print("TOP 20 FEATURES (by importance)")
print('='*70)
importances = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False).head(20)
for i, (_, row) in enumerate(importances.iterrows()):
    print(f"  {i+1:2d}. {row['feature']}: {row['importance']:.4f}")

# ── 9. Summary ──
print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for r in results:
    print(f"  {r['side']:>5s}: AUC={r['auc']:.3f} OOS_pos={r['oos_pos']:.1f}% "
          f"WR={r['best_wr']:.1f}% @ THR={r['best_thr']} (n={r['best_n']})")

# Save
output = {'symbol': SYMBOL, 'target_bars': TARGET_BARS, 'n_feats': len(feature_names),
          'tfs': list(tfs.keys()), 'results': results,
          'top_features': importances.to_dict('records')}
Path('data/multi_tf_experiment.json').write_text(json.dumps(output, indent=2))
print(f"\n✅ Saved to data/multi_tf_experiment.json")
