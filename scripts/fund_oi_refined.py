#!/usr/bin/env python3
"""
Refined Fund+OI experiment: same training window for both models.
Uses recent window (14 or 7 days) where funding covers fully.
Compares OHLCV-only vs OHLCV+Funding+OI on EQUAL data.
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

# ─── Config ───
SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "1000PEPEUSDT"
TRAIN_DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 14  # shorter window
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_SPLIT = 0.75  # higher split ratio for smaller data

print(f"{'='*70}")
print(f"REFINED EXPERIMENT: {SYMBOL} | window={TRAIN_DAYS}d | target={TARGET_BARS} bars")
print(f"{'='*70}")

# ─── Fetch functions (same as fund_oi_experiment) ───

def fetch_ohlcv_tf(symbol, interval, min_days):
    cache_dir = Path("data/ohlcv_cache")
    cache_path = cache_dir / f"{symbol}_{interval}.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        ih = {'15m':0.25,'30m':0.5,'1h':1}[interval]
        if len(df) >= int(min_days * 24 / ih): return df
    
    url = "https://fapi.binance.com/fapi/v1/klines"
    end = datetime.now()
    start = end - timedelta(days=min_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    all_bars, last_ts = [], start_ms
    while last_ts < end_ms:
        try:
            r = requests.get(url, params={'symbol':symbol,'interval':interval,'limit':1000,
                                          'startTime':last_ts,'endTime':end_ms}, timeout=30)
            if r.status_code!=200: break
            batch = r.json()
            if not batch: break
            all_bars.extend(batch)
            last_ts = batch[-1][0]+1
            if len(batch)<1000: break
        except: break
    if len(all_bars)<100: return None
    df = pd.DataFrame(all_bars, columns=['open_time','open','high','low','close','volume',
                                         'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'])
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c],errors='coerce').fillna(0)
    df['open_time'] = pd.to_datetime(df['open_time'],unit='ms')
    df.set_index('open_time',inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    result = df[['open','high','low','close','volume']]
    result.to_parquet(cache_path)
    return result

def fetch_funding(symbol):
    r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate", params={"symbol":symbol,"limit":1000}, timeout=30)
    if r.status_code!=200 or not r.json(): return None
    df = pd.DataFrame(r.json())
    df['fundingRate'] = pd.to_numeric(df['fundingRate'])
    df['fundingTime'] = pd.to_datetime(df['fundingTime'],unit='ms')
    df.set_index('fundingTime',inplace=True)
    return df[['fundingRate']]

def fetch_all_oi(symbol, days):
    all_data = []
    end_ms = int(datetime.now().timestamp()*1000)
    cs = datetime.now()-timedelta(days=days)
    ce = cs+timedelta(days=7)
    while ce < datetime.now():
        r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
            params={"symbol":symbol,"period":"5m","limit":500,"startTime":int(cs.timestamp()*1000),"endTime":int(ce.timestamp()*1000)}, timeout=30)
        if r.status_code==200 and r.json(): all_data.extend(r.json())
        cs+=timedelta(days=7); ce+=timedelta(days=7)
    s = int(cs.timestamp()*1000)
    r = requests.get("https://fapi.binance.com/futures/data/openInterestHist",
        params={"symbol":symbol,"period":"5m","limit":500,"startTime":s,"endTime":end_ms}, timeout=30)
    if r.status_code==200 and r.json(): all_data.extend(r.json())
    if not all_data: return None
    df = pd.DataFrame(all_data)
    for c in ['sumOpenInterest','sumOpenInterestValue']: df[c] = pd.to_numeric(df[c])
    df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')
    df.set_index('timestamp',inplace=True)
    return df[['sumOpenInterest','sumOpenInterestValue']]

# ─── Feature functions (same) ───
def compute_price_features(df, prefix):
    c,h,l = df['close'].astype(float), df['high'].astype(float), df['low'].astype(float)
    feats = pd.DataFrame(index=df.index)
    for i in range(1,4):
        feats[f'{prefix}ret_{i}'] = c.pct_change(i)
        feats[f'{prefix}log_ret_{i}'] = np.log(c/c.shift(i))
    tr = pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
    feats[f'{prefix}atr_14'] = tr.rolling(14).mean()
    feats[f'{prefix}atr_ratio'] = feats[f'{prefix}atr_14'] / c.rolling(14).mean()
    delta = c.diff()
    gain, loss = delta.clip(lower=0), (-delta).clip(lower=0)
    ag, al = gain.rolling(14).mean(), loss.rolling(14).mean()
    feats[f'{prefix}rsi_14'] = 100 - (100/(1+ag/al.replace(0,np.nan)))
    sma20, std20 = c.rolling(20).mean(), c.rolling(20).std()
    feats[f'{prefix}bb_width_20'] = (2*std20)/sma20
    for p in [20,50]: feats[f'{prefix}close_div_sma{p}'] = c/c.rolling(p).mean()-1
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    feats[f'{prefix}macd'] = macd/c; feats[f'{prefix}macd_signal'] = sig/c; feats[f'{prefix}macd_hist'] = (macd-sig)/c
    h20,l20 = h.rolling(20).max(), l.rolling(20).min()
    r20 = h20-l20
    feats[f'{prefix}pos_in_range_20'] = (c-l20)/r20.replace(0,np.nan)
    return feats

def compute_cross_tf(aligned):
    if not aligned: return pd.DataFrame()
    ref = list(aligned.keys())[0]
    idx = aligned[ref].index
    feats = pd.DataFrame(index=idx)
    tn = sorted(aligned.keys())
    for i in range(len(tn)-1):
        tf1,tf2 = tn[i],tn[i+1]
        p1,p2 = f'tf_{tf1}_', f'tf_{tf2}_'
        r1 = aligned[tf1].get(f'{p1}rsi_14') if f'{p1}rsi_14' in aligned[tf1].columns else None
        r2 = aligned[tf2].get(f'{p2}rsi_14') if f'{p2}rsi_14' in aligned[tf2].columns else None
        if r1 is not None and r2 is not None: feats[f'rsi_div_{tf1}_vs_{tf2}'] = r1 - r2
        a1 = aligned[tf1].get(f'{p1}close_div_sma20') if f'{p1}close_div_sma20' in aligned[tf1].columns else None
        a2 = aligned[tf2].get(f'{p2}close_div_sma20') if f'{p2}close_div_sma20' in aligned[tf2].columns else None
        if a1 is not None and a2 is not None: feats[f'trend_align_{tf1}_vs_{tf2}'] = np.sign(a1) * np.sign(a2)
        v1 = aligned[tf1].get(f'{p1}atr_ratio') if f'{p1}atr_ratio' in aligned[tf1].columns else None
        v2 = aligned[tf2].get(f'{p2}atr_ratio') if f'{p2}atr_ratio' in aligned[tf2].columns else None
        if v1 is not None and v2 is not None: feats[f'vol_ratio_{tf1}_vs_{tf2}'] = v1 / v2.replace(0, np.nan)
    return feats

# ─── Fetch all data ───
print(f"\n📡 Fetching {TRAIN_DAYS} days of data...")
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)
if df_5m is None: sys.exit(1)
# Trim to last TRAIN_DAYS (cache returns all, we need actual window)
cutoff = df_5m.index[-1] - timedelta(days=TRAIN_DAYS)
df_5m = df_5m[df_5m.index >= cutoff].copy()
print(f"  5m: {len(df_5m)} bars ({df_5m.index[0]:%Y-%m-%d %H:%M} → {df_5m.index[-1]:%Y-%m-%d %H:%M})")
tfs = {'15m': '15m', '30m': '30m', '1h': '1h'}
extra = {}
for tf in tfs: 
    d = fetch_ohlcv_tf(SYMBOL, tf, TRAIN_DAYS)
    if d is not None: extra[tf] = d

df_funding = fetch_funding(SYMBOL)
df_oi = fetch_all_oi(SYMBOL, TRAIN_DAYS)

print(f"  5m: {len(df_5m)} bars ({df_5m.index[0]:%Y-%m-%d} → {df_5m.index[-1]:%Y-%m-%d})")
print(f"  Funding: {len(df_funding) if df_funding is not None else 0} entries")
print(f"  OI: {len(df_oi) if df_oi is not None else 0} entries")

# ─── Feature computation ───
shift_map, fill_limit = {'15m':1,'30m':1,'1h':1}, {'15m':3,'30m':6,'1h':12}
tf_feats = {}
for tf_name, df_tf in extra.items():
    r = compute_price_features(df_tf, f'tf_{tf_name}_').shift(1)
    a = r.reindex(df_5m.index, method='ffill', limit=fill_limit[tf_name])
    tf_feats[tf_name] = a

feats_5m = compute_price_features(df_5m, 'tf_5m_')
cross = compute_cross_tf(tf_feats)

# Funding features
fund_feats = pd.DataFrame(index=df_5m.index)
if df_funding is not None and len(df_funding) > 10:
    fr = df_funding['fundingRate']
    ff = pd.DataFrame(index=fr.index)
    ff['funding_rate'] = fr
    ff['funding_z3'] = (fr-fr.rolling(3).mean())/fr.rolling(3).std().replace(0,np.nan)
    ff['funding_z5'] = (fr-fr.rolling(5).mean())/fr.rolling(5).std().replace(0,np.nan)
    ff['funding_roc'] = fr.pct_change()
    ff['funding_extreme'] = (fr.abs() > fr.rolling(10).std()*2).astype(float)
    ff['funding_sma3'] = fr.rolling(3).mean()
    ff = ff.shift(1)  # available after funding event
    fund_feats = ff.reindex(df_5m.index, method='ffill', limit=96)

# OI features
oi_feats = pd.DataFrame(index=df_5m.index)
if df_oi is not None and len(df_oi) > 50:
    oi = df_oi['sumOpenInterest'].astype(float)
    of = pd.DataFrame(index=df_oi.index)
    of['oi_chg_1'] = oi.pct_change(1)
    of['oi_chg_12'] = oi.pct_change(12)
    of['oi_chg_24'] = oi.pct_change(24)
    of['oi_z12'] = (oi-oi.rolling(12).mean())/oi.rolling(12).std().replace(0,np.nan)
    of['oi_z24'] = (oi-oi.rolling(24).mean())/oi.rolling(24).std().replace(0,np.nan)
    of['oi_extreme'] = (of['oi_z24'].abs()>2).astype(float)
    close = df_5m['close'].astype(float).reindex(df_5m.index)
    of['oi_div_price_12'] = of['oi_chg_12'] - close.pct_change(12)
    oi_feats = of.reindex(df_5m.index)

# Combine
ohlev_feats_list = [feats_5m] + list(tf_feats.values()) + [cross]
ohlcv = pd.concat(ohlev_feats_list, axis=1)
combined = pd.concat([ohlcv, fund_feats, oi_feats], axis=1)
combined = combined.dropna(how='all')

print(f"\n  OHLCV: {len(ohlcv.columns)} features")
print(f"  Funding: {len(fund_feats.columns)} features")
print(f"  OI: {len(oi_feats.columns)} features")
print(f"  TOTAL: {len(combined.columns)} features, {len(combined)} rows")

# ─── Target ───
close_arr = df_5m['close'].astype(float).reindex(combined.index)
for side, arr in [('long', target_long := np.zeros(len(combined), dtype=int)),
                  ('short', target_short := np.zeros(len(combined), dtype=int))]:
    for i in range(len(combined)-TARGET_BARS):
        fh = close_arr.iloc[i+1:i+TARGET_BARS+1].max()
        fl = close_arr.iloc[i+1:i+TARGET_BARS+1].min()
        target_long[i] = 1 if fh >= close_arr.iloc[i]*(1+TARGET_THR) else 0
        target_short[i] = 1 if fl <= close_arr.iloc[i]*(1-TARGET_THR) else 0

# ─── Train & Compare ───
n, split = len(combined), int(len(combined)*TRAIN_SPLIT)
feature_names = list(combined.columns)
X = combined[feature_names].fillna(0).clip(-10,10)
ohlcv_cols = list(ohlcv.columns)
fund_oi_cols = list(fund_feats.columns) + list(oi_feats.columns)

HP = {'max_depth':6,'subsample':0.80,'colsample_bytree':0.8,'learning_rate':0.05,'min_child_weight':5}

print(f"\n{'='*70}")
print("COMPARISON: Same training window, same data length")
print(f"{'='*70}")
print(f"  Training rows: {split} | OOS rows: {n-split}")

all_results = {}
for exp_name, feat_cols in [("OHLCV ONLY", ohlcv_cols), 
                              ("OHLCV+FUND+OI", feature_names)]:
    print(f"\n  ── {exp_name} ({len(feat_cols)} feats) ──")
    X_sub = X[feat_cols]
    res = []
    for side, y in [('long', target_long), ('short', target_short)]:
        yt, yo = y[:split], y[split:]
        sw = (len(yt)-yt.sum())/max(yt.sum(),1)
        m = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw, objective='binary:logistic', random_state=42, verbosity=0)
        m.fit(X_sub.iloc[:split], yt)
        p = m.predict_proba(X_sub.iloc[split:].values.astype(np.float32))[:,1]
        auc = roc_auc_score(yo, p) if len(set(yo))>1 else 0.0
        
        best_wr,best_thr,best_n = 0,'-',0
        for td in range(50,90):
            thr = td/100.0; pred = p>=thr; nt = int(pred.sum())
            if nt>=5:
                wr = float(yo[pred].mean())
                if wr>best_wr: best_wr,best_thr,best_n = wr,f'{thr:.2f}',nt
        
        print(f"    {side:>5s}: AUC={auc:.3f} WR={best_wr*100:.1f}% @ THR={best_thr} (n={best_n})")
        res.append({'side':side,'auc':round(auc,4),'wr':round(best_wr*100,1),
                    'thr':best_thr,'n':best_n,'oos_pos':round(yo.mean()*100,1)})
    all_results[exp_name] = res

# ─── Feature importance ───
print(f"\n{'='*70}")
print("FUNDING+OI FEATURE IMPORTANCE (Full Model)")
print('='*70)
for side, y in [('long', target_long), ('short', target_short)]:
    yt = y[:split]
    sw = (len(yt)-yt.sum())/max(yt.sum(),1)
    m = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw, objective='binary:logistic', random_state=42, verbosity=0)
    m.fit(X.iloc[:split], yt)
    imp = pd.DataFrame({'feat':feature_names,'imp':m.feature_importances_}).sort_values('imp',ascending=False)
    fund_oi_imp = imp[imp['feat'].isin(fund_oi_cols)]
    print(f"\n  {side.upper()} — Funding+OI features in top 50:")
    for _, r in fund_oi_imp.head(10).iterrows():
        rank = imp[imp['feat']==r['feat']].index[0]+1
        print(f"    #{rank:2d} {r['feat']:<30s} {r['imp']:.4f}")
    print(f"    ... ({len(fund_oi_imp)} total fund/oi features)")

# Summary
print(f"\n{'='*70}")
print("SUMMARY")
print('='*70)
for exp, res in all_results.items():
    print(f"\n  {exp}:")
    for r in res:
        print(f"    {r['side']:>5s}: AUC={r['auc']:.3f} WR={r['wr']:.1f}% @ THR={r['thr']} (n={r['n']})")

Path('data/fund_oi_refined.json').write_text(json.dumps(all_results, indent=2))
print(f"\n✅ Done")
