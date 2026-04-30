#!/usr/bin/env python3
"""
Combined experiment: Multi-TF OHLCV + Volume Profile (VWAP + volume structure).
Tests if VWAP enhances multi-TF performance.
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

SYMBOL = sys.argv[1] if len(sys.argv) > 1 else "1000PEPEUSDT"
TRAIN_DAYS = 45
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_SPLIT = 0.80

print(f"{'='*70}")
print(f"COMBO EXPERIMENT: {SYMBOL} | Multi-TF + Volume Profile")
print(f"{'='*70}")

# ─── Data fetching functions ───
from src.strategies.ml_features import ensure_ohlcv_data

def fetch_tf(symbol, interval, min_days):
    cache_dir = Path("data/ohlcv_cache")
    cp = cache_dir / f"{symbol}_{interval}.parquet"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if cp.exists():
        df = pd.read_parquet(cp)
        ih = {'15m':0.25,'30m':0.5,'1h':1}[interval]
        if len(df) >= int(min_days*24/ih): return df
    url = "https://fapi.binance.com/fapi/v1/klines"
    end = datetime.now(); start = end - timedelta(days=min_days)
    sm = int(start.timestamp()*1000); em = int(end.timestamp()*1000)
    bars, lt = [], sm
    while lt < em:
        try:
            r = requests.get(url, params={'symbol':symbol,'interval':interval,'limit':1000,'startTime':lt,'endTime':em}, timeout=30)
            if r.status_code!=200 or not r.json(): break
            b = r.json(); bars.extend(b); lt = b[-1][0]+1
            if len(b)<1000: break
        except: break
    if len(bars)<100: return None
    df = pd.DataFrame(bars, columns=['open_time','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'])
    for c in ['open','high','low','close','volume']: df[c] = pd.to_numeric(df[c],errors='coerce').fillna(0)
    df['open_time'] = pd.to_datetime(df['open_time'],unit='ms'); df.set_index('open_time',inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    r = df[['open','high','low','close','volume']]; r.to_parquet(cp); return r

# ─── Multi-TF features ───
def price_feats(df, prefix):
    c,h,l = df['close'].astype(float), df['high'].astype(float), df['low'].astype(float)
    f = pd.DataFrame(index=df.index)
    for i in range(1,3): f[f'{prefix}ret_{i}'] = c.pct_change(i)
    tr = pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
    f[f'{prefix}atr_14'] = tr.rolling(14).mean()
    f[f'{prefix}atr_ratio'] = f[f'{prefix}atr_14']/c.rolling(14).mean()
    delta = c.diff(); g,l_ = delta.clip(lower=0), (-delta).clip(lower=0)
    ag, al = g.rolling(14).mean(), l_.rolling(14).mean()
    f[f'{prefix}rsi_14'] = 100-(100/(1+ag/al.replace(0,np.nan)))
    for p in [20]: f[f'{prefix}close_div_sma{p}'] = c/c.rolling(p).mean()-1
    ema12,ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    macd = ema12-ema26; sig = macd.ewm(span=9).mean()
    f[f'{prefix}macd'] = macd/c; f[f'{prefix}macd_h'] = (macd-sig)/c
    h20,l20 = h.rolling(20).max(), l.rolling(20).min()
    r20 = h20-l20; f[f'{prefix}pos_range'] = (c-l20)/r20.replace(0,np.nan)
    return f

# ─── VWAP features ───
def vwap_feats(df):
    c,h,l = df['close'].astype(float), df['high'].astype(float), df['low'].astype(float)
    v = df['volume'].astype(float)
    tp = (h+l+c)/3
    f = pd.DataFrame(index=df.index)
    for p,lb in [(12,'1h'),(48,'4h')]:
        tpv = (tp*v).rolling(p).sum(); tv = v.rolling(p).sum()
        f[f'vp_vwap_{lb}'] = tpv/tv.replace(0,np.nan)
        f[f'vp_vwap_dev_{lb}'] = (c-f[f'vp_vwap_{lb}'])/c
    # Daily VWAP
    di = df.index.date
    f['vp_vwap_d'] = (tp*v).groupby(di).cumsum()/v.groupby(di).cumsum().replace(0,np.nan)
    f['vp_vwap_dev_d'] = (c-f['vp_vwap_d'])/c
    f['vp_vwap_dev_z'] = f['vp_vwap_dev_d']/f['vp_vwap_dev_d'].rolling(24).std().replace(0,np.nan)
    f['vp_vwap_slope'] = f['vp_vwap_d'].diff(12)/f['vp_vwap_d']
    return f

# ─── Volume structure ───
def vol_feats(df):
    v = df['volume'].astype(float); c = df['close'].astype(float)
    f = pd.DataFrame(index=df.index)
    f['vp_vol_ratio_12'] = v/v.rolling(12).mean().replace(0,np.nan)
    f['vp_vol_ratio_24'] = v/v.rolling(24).mean().replace(0,np.nan)
    f['vp_vol_trend'] = v.rolling(12).mean()/v.rolling(48).mean().replace(0,np.nan)-1
    f['vp_vol_z'] = (v-v.rolling(24).mean())/v.rolling(24).std().replace(0,np.nan)
    f['vp_vol_spike'] = (f['vp_vol_z'].abs()>2).astype(float)
    # Volume-weighted range
    h,l = df['high'].astype(float), df['low'].astype(float)
    f['vp_vw_range'] = (v*(h-l)).rolling(24).sum()/v.rolling(24).sum().replace(0,np.nan)/c
    return f

# ─── Fetch ───
print(f"\n📡 Data...")
df_5m = ensure_ohlcv_data(SYMBOL, min_days=TRAIN_DAYS)

# Multi-TF: get 15m, 30m, 1h
tfs = {'15m':'15m','30m':'30m','1h':'1h'}
extra = {}
for tf in tfs:
    d = fetch_tf(SYMBOL, tf, TRAIN_DAYS)
    if d is not None: extra[tf] = d

print(f"  5m: {len(df_5m)} bars")
for k,v in extra.items(): print(f"  {k}: {len(v)} bars")

# ─── Compute features ───
print(f"\n🔧 Features...")

# Multi-TF features (price only)
shift_map, fill_limit = {'15m':1,'30m':1,'1h':1}, {'15m':3,'30m':6,'1h':12}
tf_feats = {}
for tf_name, df_tf in extra.items():
    r = price_feats(df_tf, f'tf_{tf_name}_').shift(1)
    a = r.reindex(df_5m.index, method='ffill', limit=fill_limit[tf_name])
    tf_feats[tf_name] = a

f_5m = price_feats(df_5m, 'tf_5m_')
# Cross-TF
idx = list(tf_feats.values())[0].index
cross = pd.DataFrame(index=idx)
tn = sorted(tf_feats.keys())
for i in range(len(tn)-1):
    t1,t2 = tn[i],tn[i+1]
    p1,p2 = f'tf_{t1}_', f'tf_{t2}_'
    for feat, name in [('rsi_14','rsi_div'),('close_div_sma20','trend_align'),('macd','macd_div')]:
        c1 = tf_feats[t1].get(f'{p1}{feat}') if f'{p1}{feat}' in tf_feats[t1].columns else None
        c2 = tf_feats[t2].get(f'{p2}{feat}') if f'{p2}{feat}' in tf_feats[t2].columns else None
        if c1 is not None and c2 is not None:
            if i==0: cross[f'vol_ratio_{t1}_vs_{t2}'] = tf_feats[t1].get(f'{p1}atr_ratio', pd.Series(index=idx))/tf_feats[t2].get(f'{p2}atr_ratio', pd.Series(index=idx)).replace(0,np.nan)
            cross[f'{name}_{t1}_vs_{t2}'] = c1-c2

mtf_feats = pd.concat([f_5m] + list(tf_feats.values()) + [cross], axis=1)
print(f"  Multi-TF: {len(mtf_feats.columns)} features")

# Volume Profile features
v_vwap = vwap_feats(df_5m)
v_vol = vol_feats(df_5m)
vp_feats = pd.concat([v_vwap, v_vol], axis=1)
print(f"  Volume Profile: {len(vp_feats.columns)} features")

# Combined
combined = pd.concat([mtf_feats, vp_feats], axis=1).dropna(how='all')
mtf_cols = list(mtf_feats.columns)
vp_cols = list(vp_feats.columns)
print(f"  TOTAL: {len(combined.columns)} features, {len(combined)} rows")

# ─── Target ───
c = df_5m['close'].astype(float).reindex(combined.index)
tl = np.zeros(len(combined), dtype=int)
ts = np.zeros(len(combined), dtype=int)
for i in range(len(combined)-TARGET_BARS):
    fh = c.iloc[i+1:i+TARGET_BARS+1].max()
    fl = c.iloc[i+1:i+TARGET_BARS+1].min()
    tl[i] = 1 if fh >= c.iloc[i]*(1+TARGET_THR) else 0
    ts[i] = 1 if fl <= c.iloc[i]*(1-TARGET_THR) else 0

# ─── Train: MTF vs MTF+VP ───
n, sp = len(combined), int(len(combined)*TRAIN_SPLIT)
fn = list(combined.columns)
X = combined[fn].fillna(0).clip(-10,10)
HP = {'max_depth':6,'subsample':0.80,'colsample_bytree':0.8,'learning_rate':0.05,'min_child_weight':5}

print(f"\n{'='*70}")
print("COMPARISON: Multi-TF vs Multi-TF + Volume Profile")
print(f"{'='*70}")
print(f"  Train: {sp} | OOS: {n-sp}")

all_res = {}
for exp, fc in [("MTF ONLY", mtf_cols), ("MTF+VOLPROF", fn)]:
    print(f"\n  ── {exp} ({len(fc)} feats) ──")
    Xs = X[fc]
    res = []
    for side, y in [('long', tl), ('short', ts)]:
        yt, yo = y[:sp], y[sp:]
        sw = (len(yt)-yt.sum())/max(yt.sum(),1)
        m = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw, objective='binary:logistic', random_state=42, verbosity=0)
        m.fit(Xs.iloc[:sp], yt)
        p = m.predict_proba(Xs.iloc[sp:].values.astype(np.float32))[:,1]
        auc = roc_auc_score(yo, p) if len(set(yo))>1 else 0.0
        bw,bt,bn = 0,'-',0
        for td in range(50,90):
            thr = td/100.0; pred = p>=thr; nt = int(pred.sum())
            if nt>=5:
                wr = float(yo[pred].mean())
                if wr>bw: bw,bt,bn = wr,f'{thr:.2f}',nt
        # Also show at thr=0.70 (broader)
        pred70 = p>=0.70; n70 = int(pred70.sum()); wr70 = float(yo[pred70].mean())*100 if n70>=5 else 0
        print(f"    {side:>5s}: AUC={auc:.3f} WR={bw*100:.1f}% @ THR={bt} (n={bn}) | WR@0.70={wr70:.1f}% (n={n70})")
        res.append({'side':side,'auc':round(auc,3),'wr':round(bw*100,1),'thr':bt,'n':bn,'wr70':round(wr70,1),'n70':n70})
    all_res[exp] = res

# ─── Feature importance ───
print(f"\n{'='*70}")
print("TOP FEATURES")
print('='*70)
for side, y in [('long', tl), ('short', ts)]:
    yt=y[:sp]; sw=(len(yt)-yt.sum())/max(yt.sum(),1)
    m=xgb.XGBClassifier(n_estimators=400,**HP,scale_pos_weight=sw,objective='binary:logistic',random_state=42,verbosity=0)
    m.fit(X.iloc[:sp], yt)
    imp = pd.DataFrame({'feat':fn,'imp':m.feature_importances_}).sort_values('imp',ascending=False)
    print(f"\n  {side.upper()} TOP 15:")
    for _, r in imp.head(15).iterrows():
        tag = ' [VP]' if r['feat'] in vp_cols else ''
        print(f"    {r['feat']:<35s} {r['imp']:.4f}{tag}")
    # VP feature ranks
    vp_imp = imp[imp['feat'].isin(vp_cols)]
    if len(vp_imp)>0:
        vp_ranks = []
        for idx, (_, row) in enumerate(vp_imp.head(5).iterrows()):
            vp_ranks.append(f"#{idx+1}-{row['feat']}")
        print(f"  Volume Profile ranks: {', '.join(vp_ranks)}")

# ─── Summary ───
print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
for exp, res in all_res.items():
    print(f"\n  {exp}:")
    for r in res:
        print(f"    {r['side']:>5s}: AUC={r['auc']:.3f} WR={r['wr']:.1f}% (n={r['n']}) | WR@0.70={r['wr70']:.1f}% (n={r['n70']})")

Path('data/combo_experiment.json').write_text(json.dumps(all_res, indent=2))
print(f"\n✅ Done")
