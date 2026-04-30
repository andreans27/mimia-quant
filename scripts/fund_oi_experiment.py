#!/usr/bin/env python3
"""
Fundamental + Market Structure features experiment.
Adds funding rate + OI features on top of multi-TF OHLCV features.
Tests if these improve predictive power.
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

# ─── 1. Data Fetching ───

def fetch_ohlcv_tf(symbol: str, interval: str, min_days: int = 45):
    cache_dir = Path("data/ohlcv_cache")
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        interval_hours = {'15m': 0.25, '30m': 0.5, '1h': 1}[interval]
        if len(df) >= int(min_days * 24 / interval_hours):
            return df
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    end = datetime.now()
    start = end - timedelta(days=min_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    all_bars = []
    last_ts = start_ms
    while last_ts < end_ms:
        params = {'symbol': symbol, 'interval': interval, 'limit': 1000,
                  'startTime': last_ts, 'endTime': end_ms}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200: break
            batch = r.json()
            if not batch: break
            all_bars.extend(batch)
            last_ts = batch[-1][0] + 1
            if len(batch) < 1000: break
        except: break
    
    if len(all_bars) < 100: return None
    df = pd.DataFrame(all_bars, columns=['open_time','open','high','low','close','volume',
                                         'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'])
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    result = df[['open','high','low','close','volume']]
    result.to_parquet(cache_path)
    return result


def fetch_funding_rate(symbol: str, limit: int = 1000):
    """Fetch funding rate history. Returns DataFrame indexed by funding time."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=30)
    if r.status_code != 200 or not r.json():
        return None
    data = r.json()
    df = pd.DataFrame(data)
    df['fundingRate'] = pd.to_numeric(df['fundingRate'])
    df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
    df.set_index('fundingTime', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['fundingRate']]


def fetch_all_oi(symbol: str, days: int = 45):
    """Fetch OI history via paginated API."""
    all_data = []
    end_ms = int(datetime.now().timestamp() * 1000)
    chunk_start = datetime.now() - timedelta(days=days)
    chunk_end = chunk_start + timedelta(days=7)
    
    while chunk_end < datetime.now():
        s = int(chunk_start.timestamp() * 1000)
        e = int(chunk_end.timestamp() * 1000)
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol": symbol, "period": "5m", "limit": 500,
                    "startTime": s, "endTime": e}, timeout=30)
        if r.status_code == 200 and r.json():
            all_data.extend(r.json())
        chunk_start += timedelta(days=7)
        chunk_end += timedelta(days=7)
    
    s = int(chunk_start.timestamp() * 1000)
    r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol": symbol, "period": "5m", "limit": 500,
                "startTime": s, "endTime": end_ms}, timeout=30)
    if r.status_code == 200 and r.json():
        all_data.extend(r.json())
    
    if not all_data: return None
    df = pd.DataFrame(all_data)
    for col in ['sumOpenInterest', 'sumOpenInterestValue']:
        df[col] = pd.to_numeric(df[col])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['sumOpenInterest', 'sumOpenInterestValue']]


# ─── 2. Features ───

def compute_price_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    feats = pd.DataFrame(index=df.index)
    
    for i in range(1, 4):
        feats[f'{prefix}ret_{i}'] = c.pct_change(i)
        feats[f'{prefix}log_ret_{i}'] = np.log(c / c.shift(i))
    
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    feats[f'{prefix}atr_14'] = tr.rolling(14).mean()
    feats[f'{prefix}atr_ratio'] = feats[f'{prefix}atr_14'] / c.rolling(14).mean()
    
    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_g = gain.rolling(14).mean()
    avg_l = loss.rolling(14).mean()
    rs = avg_g / avg_l.replace(0, np.nan)
    feats[f'{prefix}rsi_14'] = 100 - (100 / (1 + rs))
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feats[f'{prefix}bb_width_20'] = (2 * std20) / sma20
    feats[f'{prefix}bb_pct_20'] = (c - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
    
    for period in [20, 50]:
        sma = c.rolling(period).mean()
        feats[f'{prefix}close_div_sma{period}'] = c / sma - 1
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    feats[f'{prefix}macd'] = macd / c
    feats[f'{prefix}macd_signal'] = signal / c
    feats[f'{prefix}macd_hist'] = (macd - signal) / c
    
    high_20 = h.rolling(20).max()
    low_20 = l.rolling(20).min()
    range_20 = high_20 - low_20
    feats[f'{prefix}pos_in_range_20'] = (c - low_20) / range_20.replace(0, np.nan)
    feats[f'{prefix}range_20'] = range_20 / c
    
    return feats


def compute_cross_tf_features(aligned: dict) -> pd.DataFrame:
    ref = list(aligned.keys())[0]
    idx = aligned[ref].index
    feats = pd.DataFrame(index=idx)
    tf_names = sorted(aligned.keys())
    
    for i in range(len(tf_names) - 1):
        tf1, tf2 = tf_names[i], tf_names[i+1]
        p1, p2 = f'tf_{tf1}_', f'tf_{tf2}_'
        
        r1 = aligned[tf1].get(f'{p1}rsi_14')
        r2 = aligned[tf2].get(f'{p2}rsi_14')
        if r1 is not None and r2 is not None:
            feats[f'rsi_div_{tf1}_vs_{tf2}'] = r1 - r2
        
        a1 = aligned[tf1].get(f'{p1}close_div_sma20')
        a2 = aligned[tf2].get(f'{p2}close_div_sma20')
        if a1 is not None and a2 is not None:
            feats[f'trend_align_{tf1}_vs_{tf2}'] = np.sign(a1) * np.sign(a2)
        
        v1 = aligned[tf1].get(f'{p1}atr_ratio')
        v2 = aligned[tf2].get(f'{p2}atr_ratio')
        if v1 is not None and v2 is not None:
            feats[f'vol_ratio_{tf1}_vs_{tf2}'] = v1 / v2.replace(0, np.nan)
        
        m1 = aligned[tf1].get(f'{p1}macd')
        m2 = aligned[tf2].get(f'{p2}macd')
        if m1 is not None and m2 is not None:
            feats[f'macd_div_{tf1}_vs_{tf2}'] = m1 - m2
    
    return feats


def compute_funding_features(df_funding: pd.DataFrame, idx_5m: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute funding rate features and forward-fill to 5m index."""
    if df_funding is None or len(df_funding) < 10:
        return pd.DataFrame(index=idx_5m)
    
    fr = df_funding['fundingRate']
    
    # Funding features (on 8h intervals)
    fund_feats = pd.DataFrame(index=fr.index)
    fund_feats['funding_rate'] = fr
    fund_feats['funding_z3'] = (fr - fr.rolling(3).mean()) / fr.rolling(3).std().replace(0, np.nan)
    fund_feats['funding_z5'] = (fr - fr.rolling(5).mean()) / fr.rolling(5).std().replace(0, np.nan)
    fund_feats['funding_roc'] = fr.pct_change()
    fund_feats['funding_extreme'] = (fr.abs() > fr.rolling(10).std() * 2).astype(float)
    fund_feats['funding_sma3'] = fr.rolling(3).mean()
    
    # Shift by 1 funding period (8h) — data only available AFTER each funding event
    fund_feats = fund_feats.shift(1)
    
    # Forward-fill to 5m
    aligned = fund_feats.reindex(idx_5m, method='ffill', limit=96)  # 96 * 5m = 8h
    return aligned


def compute_oi_features(df_oi: pd.DataFrame) -> pd.DataFrame:
    """Compute OI features. df_oi is indexed by 5m timestamp."""
    if df_oi is None or len(df_oi) < 50:
        return pd.DataFrame(index=df_oi.index) if df_oi is not None else pd.DataFrame()
    
    oi = df_oi['sumOpenInterest'].astype(float)
    oi_usd = df_oi['sumOpenInterestValue'].astype(float)
    
    feats = pd.DataFrame(index=df_oi.index)
    
    # OI changes
    feats['oi_chg_1'] = oi.pct_change(1)
    feats['oi_chg_12'] = oi.pct_change(12)  # 1h
    feats['oi_chg_24'] = oi.pct_change(24)  # 2h
    feats['oi_chg_48'] = oi.pct_change(48)  # 4h
    
    # OI volatility
    feats['oi_vol_12'] = oi.pct_change(1).rolling(12).std()
    feats['oi_vol_24'] = oi.pct_change(1).rolling(24).std()
    
    # OI z-score
    feats['oi_zscore_12'] = (oi - oi.rolling(12).mean()) / oi.rolling(12).std().replace(0, np.nan)
    feats['oi_zscore_24'] = (oi - oi.rolling(24).mean()) / oi.rolling(24).std().replace(0, np.nan)
    feats['oi_zscore_48'] = (oi - oi.rolling(48).mean()) / oi.rolling(48).std().replace(0, np.nan)
    
    # OI extreme
    feats['oi_extreme'] = (feats['oi_zscore_24'].abs() > 2).astype(float)
    
    # OI vs price (requires close from OHLCV — added later)
    
    return feats


# ─── Config ───
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "1000PEPEUSDT"
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_DAYS = 45
TRAIN_SPLIT = 0.80

print(f"{'='*70}")
print(f"FUNDING + OI EXPERIMENT: {SYMBOL} | target={TARGET_BARS} bars")
print(f"{'='*70}")

# ─── Fetch ALL data ───
print(f"\n📡 Fetching data...")
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
if df_5m is None: sys.exit(1)

# Higher TFs
tfs = {'15m': 0.25, '30m': 0.5, '1h': 1}
extra_data = {}
for tf_name in tfs:
    df = fetch_ohlcv_tf(SYMBOL, tf_name, min_days=TRAIN_DAYS)
    if df is not None:
        extra_data[tf_name] = df
        print(f"  {tf_name}: {len(df)} bars")

# Funding rate
df_funding = fetch_funding_rate(SYMBOL, limit=1000)
print(f"  Funding: {len(df_funding) if df_funding is not None else 0} entries")

# OI
df_oi = fetch_all_oi(SYMBOL, days=max(TRAIN_DAYS, 30))
print(f"  OI: {len(df_oi) if df_oi is not None else 0} entries")

# ─── Compute OHLCV features (same as multi-TF experiment) ───
print(f"\n🔧 OHLCV features...")
shift_map = {'15m': 1, '30m': 1, '1h': 1}
fill_limit = {'15m': 3, '30m': 6, '1h': 12}
tf_feats = {}

for tf_name, df_tf in extra_data.items():
    prefix = f'tf_{tf_name}_'
    raw = compute_price_features(df_tf, prefix)
    shifted = raw.shift(shift_map[tf_name])
    aligned = shifted.reindex(df_5m.index, method='ffill', limit=fill_limit[tf_name])
    tf_feats[tf_name] = aligned

feats_5m = compute_price_features(df_5m, 'tf_5m_')
cross_feats = compute_cross_tf_features(tf_feats)

ohlcv_feats = pd.concat([feats_5m] + list(tf_feats.values()) + [cross_feats], axis=1)
print(f"  OHLCV features: {len(ohlcv_feats.columns)}")

# ─── Compute Funding + OI features ───
print(f"\n🔧 Fund + OI features...")
fund_feats = compute_funding_features(df_funding, df_5m.index)
print(f"  Funding features: {len(fund_feats.columns)}")

oi_feats = compute_oi_features(df_oi) if df_oi is not None else pd.DataFrame(index=df_5m.index)
print(f"  OI features: {len(oi_feats.columns)}")

# OI vs Price
close = df_5m['close'].astype(float).reindex(df_5m.index)
if len(oi_feats) > 0:
    oi_feats['oi_chg_12_div_price'] = oi_feats['oi_chg_12'] - close.pct_change(12)
    oi_feats['oi_chg_24_div_price'] = oi_feats['oi_chg_24'] - close.pct_change(24)

# ─── Combine ALL features ───
combined = pd.concat([ohlcv_feats, fund_feats, oi_feats], axis=1)
combined = combined.dropna(how='all')
print(f"  TOTAL features: {len(combined.columns)}")

# Count how many rows have OI data
oi_available = combined['oi_chg_1'].notna().sum() if 'oi_chg_1' in combined.columns else 0
fund_available = combined['funding_rate'].notna().sum() if 'funding_rate' in combined.columns else 0
print(f"  Rows with OI data: {oi_available}/{len(combined)}")
print(f"  Rows with funding: {fund_available}/{len(combined)}")

# ─── Target ───
close_arr = df_5m['close'].astype(float).reindex(combined.index)
target_long = np.zeros(len(combined), dtype=int)
target_short = np.zeros(len(combined), dtype=int)
for i in range(len(combined) - TARGET_BARS):
    fh = close_arr.iloc[i+1:i+TARGET_BARS+1].max()
    fl = close_arr.iloc[i+1:i+TARGET_BARS+1].min()
    target_long[i] = 1 if fh >= close_arr.iloc[i] * (1 + TARGET_THR) else 0
    target_short[i] = 1 if fl <= close_arr.iloc[i] * (1 - TARGET_THR) else 0

# ─── Train: BASELINE (OHLCV only) vs FULL (OHLCV + Fund + OI) ───
print(f"\n{'='*70}")
print(f"TRAINING: Baseline (OHLCV-only) vs Full (OHLCV + Fund + OI)")
print(f"{'='*70}")

n = len(combined)
split = int(n * TRAIN_SPLIT)
feature_names = [c for c in combined.columns]
X = combined[feature_names].fillna(0).clip(-10, 10)

# Identify feature groups
ohlcv_cols = list(ohlcv_feats.columns)
fund_cols = list(fund_feats.columns) if len(fund_feats.columns) > 0 else []
oi_cols = list(oi_feats.columns) if len(oi_feats.columns) > 0 else []
fund_oi_cols = fund_cols + oi_cols

HPARAMS = {42: {'max_depth': 6, 'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5}}

all_results = {}

for exp_name, feat_cols in [("BASELINE (OHLCV)", ohlcv_cols), 
                              ("FULL (+Fund+OI)", feature_names)]:
    print(f"\n  ── {exp_name} ──")
    X_sub = X[feat_cols]
    
    results = []
    for side in ['long', 'short']:
        y = target_long if side == 'long' else target_short
        y_train, y_oos = y[:split], y[split:]
        
        sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
        
        model = xgb.XGBClassifier(n_estimators=400, **HPARAMS[42],
            scale_pos_weight=sw, objective='binary:logistic',
            random_state=42, verbosity=0)
        model.fit(X_sub.iloc[:split], y_train)
        
        p = model.predict_proba(X_sub.iloc[split:].values.astype(np.float32))[:, 1]
        
        try: auc = roc_auc_score(y_oos, p)
        except: auc = 0.0
        
        best_wr, best_thr, best_n = 0, '-', 0
        for thr_dec in range(50, 90):
            thr = thr_dec / 100.0
            pred = p >= thr
            nt = int(pred.sum())
            if nt >= 5:
                wr = float(y_oos[pred].mean())
                if wr > best_wr:
                    best_wr, best_thr, best_n = wr, f'{thr:.2f}', nt
        
        pos_rate = y_oos.mean() * 100
        print(f"    {side:>5s}: AUC={auc:.3f} OOS_pos={pos_rate:.1f}% WR={best_wr*100:.1f}% @ THR={best_thr} (n={best_n})")
        
        results.append({'side': side, 'auc': round(auc, 4), 'best_wr': round(best_wr*100, 1),
                        'best_thr': best_thr, 'best_n': best_n, 'oos_pos': round(pos_rate, 1)})
    
    all_results[exp_name] = results

# ─── Feature Importance (Full model) ───
print(f"\n{'='*70}")
print("TOP 20 FEATURES (Full Model)")
print('='*70)
# Retrain full model on FULL features
for side in ['long', 'short']:
    y = target_long if side == 'long' else target_short
    y_train = y[:split]
    sw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)
    m = xgb.XGBClassifier(n_estimators=400, **HPARAMS[42],
        scale_pos_weight=sw, objective='binary:logistic',
        random_state=42, verbosity=0)
    m.fit(X.iloc[:split], y_train)
    
    imp = pd.DataFrame({'feature': feature_names, 'importance': m.feature_importances_})
    imp = imp.sort_values('importance', ascending=False).head(20)
    
    print(f"\n  {side.upper()}:")
    for i, (_, row) in enumerate(imp.iterrows()):
        tag = ""
        if row['feature'] in fund_cols: tag = " [FUND]"
        elif row['feature'] in oi_cols: tag = " [OI]"
        print(f"    {i+1:2d}. {row['feature']:<30s} {row['importance']:.4f}{tag}")

# ─── Summary ───
print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for exp, results in all_results.items():
    print(f"\n  {exp}:")
    for r in results:
        lr = "⬆" if r['auc'] > 0.55 else "⬇" if r['auc'] < 0.45 else "→"
        print(f"    {r['side']:>5s}: AUC={r['auc']:.3f} {lr} WR={r['best_wr']:.1f}% @ THR={r['best_thr']} (n={r['best_n']})")

# Save
Path('data/fund_oi_experiment.json').write_text(json.dumps(all_results, indent=2))
print(f"\n✅ Saved to data/fund_oi_experiment.json")
