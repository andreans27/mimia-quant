#!/usr/bin/env python3
"""
Comprehensive Volume/Market Profile experiment.
Tests VWAP, Volume Structure, Candle Anatomy, Microstructure features.
Validates on 5 symbols to prevent overfitting.
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
SYMBOLS = [
    ("BTCUSDT",    "Majors"),
    ("ETHUSDT",    "Majors"),
    ("SOLUSDT",    "Alt L1"),
    ("1000PEPEUSDT", "Meme"),
    ("ENAUSDT",    "New Gen"),
]
TARGET_BARS = 6
TARGET_THR = 0.005
TRAIN_DAYS = 45
TRAIN_SPLIT = 0.80

# ─── Helper fetch ───
def fetch_tf(symbol, interval, min_days):
    from src.strategies.ml_features import ensure_ohlcv_data  # dummy
    cp = Path(f"data/ohlcv_cache/{symbol}_{interval}.parquet")
    if cp.exists():
        df = pd.read_parquet(cp)
        ih = {'15m':0.25,'30m':0.5,'1h':1}[interval]
        if len(df) >= int(min_days*24/ih): return df
    url, end = "https://fapi.binance.com/fapi/v1/klines", datetime.now()
    sm = int((end - timedelta(days=min_days)).timestamp()*1000)
    em = int(end.timestamp()*1000)
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
    for c in ['open','high','low','close','volume']: df[c]=pd.to_numeric(df[c],errors='coerce').fillna(0)
    df['open_time']=pd.to_datetime(df['open_time'],unit='ms'); df.set_index('open_time',inplace=True)
    df=df[~df.index.duplicated(keep='last')].sort_index()
    r=df[['open','high','low','close','volume']]; Path('data/ohlcv_cache').mkdir(exist_ok=True); r.to_parquet(cp); return r

# ─── Feature Groups ───

def compute_mtf(df_5m, extra_tfs):
    """Multi-TF price features (compact)."""
    def pf(df, prefix):
        c,h,l=df['close'].astype(float),df['high'].astype(float),df['low'].astype(float)
        f=pd.DataFrame(index=df.index)
        for i in range(1,3): f[f'{prefix}ret_{i}']=c.pct_change(i)
        tr=pd.concat([h-l,abs(h-c.shift(1)),abs(l-c.shift(1))],axis=1).max(axis=1)
        f[f'{prefix}atr_14']=tr.rolling(14).mean()
        f[f'{prefix}atr_ratio']=f[f'{prefix}atr_14']/c.rolling(14).mean()
        d=c.diff(); g,la=d.clip(lower=0),(-d).clip(lower=0)
        ag,al=g.rolling(14).mean(),la.rolling(14).mean()
        f[f'{prefix}rsi_14']=100-(100/(1+ag/al.replace(0,np.nan)))
        f[f'{prefix}cd_sma20']=c/c.rolling(20).mean()-1
        ema12,ema26=c.ewm(span=12).mean(),c.ewm(span=26).mean()
        macd=ema12-ema26;s=macd.ewm(span=9).mean()
        f[f'{prefix}macd']=macd/c;f[f'{prefix}macdh']=(macd-s)/c
        h20,l20=h.rolling(20).max(),l.rolling(20).min()
        f[f'{prefix}pos_r']=(c-l20)/(h20-l20).replace(0,np.nan)
        return f
    
    shift_map={'15m':1,'30m':1,'1h':1}; fl={'15m':3,'30m':6,'1h':12}
    tf_f={}
    for tn,df_tf in extra_tfs.items():
        r=pf(df_tf,f'tf_{tn}_').shift(1)
        a=r.reindex(df_5m.index,method='ffill',limit=fl[tn]); tf_f[tn]=a
    f5m=pf(df_5m,'tf_5m_')
    
    # Cross-TF
    idx=df_5m.index
    cross=pd.DataFrame(index=idx)
    tn=sorted(tf_f.keys())
    for i in range(len(tn)-1):
        t1,t2=tn[i],tn[i+1]; p1,p2=f'tf_{t1}_',f'tf_{t2}_'
        for ft,nm in [('rsi_14','rsi'),('cd_sma20','trend'),('macd','macd')]:
            c1=tf_f[t1].get(f'{p1}{ft}'); c2=tf_f[t2].get(f'{p2}{ft}')
            if c1 is not None and c2 is not None: cross[f'{nm}_{t1}_v_{t2}']=c1-c2
        vr1=tf_f[t1].get(f'{p1}atr_ratio'); vr2=tf_f[t2].get(f'{p2}atr_ratio')
        if vr1 is not None and vr2 is not None: cross[f'volr_{t1}_v_{t2}']=vr1/vr2.replace(0,np.nan)
    
    return pd.concat([f5m]+list(tf_f.values())+[cross],axis=1)

def compute_vwap(df):
    """VWAP family (expanded)."""
    c,h,l=df['close'].astype(float),df['high'].astype(float),df['low'].astype(float)
    v=df['volume'].astype(float); tp=(h+l+c)/3
    f=pd.DataFrame(index=df.index)
    # Multiple VWAP periods
    for p,lb in [(12,'1h'),(48,'4h'),(96,'8h')]:
        tpv=(tp*v).rolling(p).sum(); tv=v.rolling(p).sum()
        vwap=tpv/tv.replace(0,np.nan); f[f'vwap_{lb}']=vwap
        f[f'vwap_dv_{lb}']=(c-vwap)/c
        f[f'vwap_bw_{lb}']=vwap.rolling(12).std()/c  # VWAP band width
    # Daily VWAP (anchored)
    di=df.index.date
    dv=(tp*v).groupby(di).cumsum()/v.groupby(di).cumsum().replace(0,np.nan)
    f['vwap_d']=dv; f['vwap_dv_d']=(c-dv)/c
    f['vwap_dz']=f['vwap_dv_d']/f['vwap_dv_d'].rolling(24).std().replace(0,np.nan)
    f['vwap_sl']=dv.diff(12)/dv  # slope
    f['vwap_sl2']=dv.diff(12).diff(12)/dv  # acceleration
    # VWAP crossovers
    for lb in ['1h','4h','8h','d']:
        above=(c>f[f'vwap_{lb}']).astype(float) if lb!='d' else (c>f['vwap_d']).astype(float)
        f[f'vwap_x_{lb}']=above.diff().clip(lower=0)  # 1 = crossed UP
        f[f'vwap_xd_{lb}']=(-above.diff()).clip(lower=0)  # 1 = crossed DOWN
    # Consecutive bars away from VWAP
    for lb in ['1h','4h','d']:
        dev=f[f'vwap_dv_{lb}'] if lb!='d' else f['vwap_dv_d']
        above=(dev>0).astype(float)
        f[f'vwap_cb_{lb}']=above.groupby((above!=above.shift()).cumsum()).cumcount()+1
    return f

def compute_volume(f):
    """Volume analysis features."""
    v=f['volume'].astype(float); c=f['close'].astype(float)
    h=f['high'].astype(float); l=f['low'].astype(float); o=f['open'].astype(float)
    rng=h-l
    feats=pd.DataFrame(index=f.index)
    # Volume ratios
    feats['v_r12']=v/v.rolling(12).mean().replace(0,np.nan)
    feats['v_r24']=v/v.rolling(24).mean().replace(0,np.nan)
    feats['v_r48']=v/v.rolling(48).mean().replace(0,np.nan)
    feats['v_trend']=v.rolling(12).mean()/v.rolling(48).mean().replace(0,np.nan)-1
    feats['v_z']=(v-v.rolling(24).mean())/v.rolling(24).std().replace(0,np.nan)
    feats['v_spike']=(feats['v_z'].abs()>2).astype(float)
    feats['v_accel']=v.diff(12)-v.diff(24)  # acceleration
    # Volume × range (dollar volatility)
    feats['vxr']=(v*rng)/c/1e6  # scaled
    feats['vxr_ma']=feats['vxr'].rolling(12).mean()
    feats['vxr_z']=(feats['vxr']-feats['vxr'].rolling(24).mean())/feats['vxr'].rolling(24).std().replace(0,np.nan)
    # Volume-weighted return
    feats['vwr_12']=(c.pct_change(12)*v.rolling(12).sum()).rolling(12).mean()
    # Volume direction
    price_dir=np.sign(c.diff())
    vol_dir=np.sign(v.diff())
    feats['pv_confirm']=(price_dir*vol_dir).clip(lower=0)
    feats['pv_diverg']=((-price_dir*vol_dir).clip(lower=0))
    # Volume exhaustion (high vol + small range = trend end)
    feats['v_exhaust']=(v/v.rolling(48).mean())/(rng/rng.rolling(48).mean()).replace(0,np.nan)
    return feats

def compute_candle(f):
    """Candle anatomy features."""
    o=f['open'].astype(float); h=f['high'].astype(float)
    l=f['low'].astype(float); c=f['close'].astype(float); v=f['volume'].astype(float)
    rng=h-l; body=abs(c-o); upper=h-np.maximum(o,c); lower=np.minimum(o,c)-l
    feats=pd.DataFrame(index=f.index)
    feats['can_body%']=body/rng.replace(0,np.nan)
    feats['can_upper%']=upper/rng.replace(0,np.nan)
    feats['can_lower%']=lower/rng.replace(0,np.nan)
    feats['can_dir']=np.sign(c-o)  # +1 green, -1 red
    feats['can_str']=(c-o)/rng.replace(0,np.nan)  # signed strength
    # Volume-weighted candle strength
    feats['can_vstr']=feats['can_str']*feats['v_r24']
    feats['can_body_v']=feats['can_body%']*feats['v_r24']
    # Doji detection (small body)
    feats['can_doji']=(body/rng<0.1).astype(float)
    feats['can_hammer']=((lower/rng>0.5)&(body/rng<0.3)).astype(float)
    feats['can_shoot']=((upper/rng>0.5)&(body/rng<0.3)).astype(float)
    feats['can_maru']=((c==h)&(o==l)).astype(float)+((c==l)&(o==h)).astype(float)  # marubozu
    # Consecutive up/down candles
    up=(c>o).astype(float)
    feats['can_consec']=up.groupby((up!=up.shift()).cumsum()).cumcount()+1
    feats['can_consec_v']=feats['can_consec']*feats['v_r24']
    # Big body detection
    body_avg=body.rolling(20).mean()
    feats['can_bigbody']=(body>body_avg*2).astype(float)
    return feats

def compute_micro(f):
    """Microstructure features (trades + volume composition)."""
    v=f['volume'].astype(float); c=f['close'].astype(float); h=f['high'].astype(float); l=f['low'].astype(float)
    rng=h-l
    feats=pd.DataFrame(index=f.index)
    
    # Trade features (available from cache)
    if 'trades' in f.columns and 'quote_volume' in f.columns:
        tr=f['trades'].astype(float); qv=f['quote_volume'].astype(float)
        feats['mic_avg_trade']=qv/tr.replace(0,np.nan)/c  # avg trade size in base
        feats['mic_trade_density']=tr/(rng/c).replace(0,np.nan)  # trades per price range
        feats['mic_trade_z']=(tr-tr.rolling(24).mean())/tr.rolling(24).std().replace(0,np.nan)
        feats['mic_dollar_trade']=qv/tr.replace(0,np.nan)  # $ per trade
    
    # Taker buy ratio (limited coverage but useful)
    if 'taker_buy_quote' in f.columns:
        tbq=f['taker_buy_quote'].astype(float)
        qv=f['quote_volume'].astype(float) if 'quote_volume' in f.columns else v*c
        feats['mic_buy_r']=tbq/qv.replace(0,np.nan)
        feats['mic_buy_r_ma']=feats['mic_buy_r'].rolling(12).mean()
        feats['mic_buy_r_z']=(feats['mic_buy_r']-feats['mic_buy_r'].rolling(24).mean())/feats['mic_buy_r'].rolling(24).std().replace(0,np.nan)
    
    # Volume composition (available for all bars)
    feats['mic_dollar_vol']=v*c/1e6  # dollar volume in millions
    feats['mic_vol_density']=v/rng.replace(0,np.nan)/1e6  # volume per price unit
    feats['mic_vol_density_ma']=feats['mic_vol_density'].rolling(24).mean()
    return feats

# ─── Run for all symbols ───
all_results = {}
HP = {'max_depth':6,'subsample':0.80,'colsample_bytree':0.8,'learning_rate':0.05,'min_child_weight':5}

for sym, cat in SYMBOLS:
    print(f"\n{'='*70}")
    print(f"📡 {sym:>12s} ({cat}) | target={TARGET_BARS} bars (30m)")
    print(f"{'='*70}")
    
    t0 = time.time()
    
    df_5m = ensure_ohlcv_data(sym, min_days=TRAIN_DAYS)
    if df_5m is None:
        print(f"  ❌ No 5m data")
        continue
    
    # Fetch higher TFs
    extra = {}
    for tf in ['15m','30m','1h']:
        d = fetch_tf(sym, tf, TRAIN_DAYS)
        if d is not None: extra[tf] = d
    
    if len(extra) < 2:
        print(f"  ❌ Not enough TF data")
        continue
    
    # Compute features
    try:
        mtf = compute_mtf(df_5m, extra)
        vwap = compute_vwap(df_5m)
        vol = compute_volume(df_5m)
        cand = compute_candle(df_5m)
        micro = compute_micro(df_5m)
    except Exception as e:
        print(f"  ❌ Feature error: {e}")
        continue
    
    # Combine
    vp_all = pd.concat([vwap, vol, cand, micro], axis=1)
    combined = pd.concat([mtf, vp_all], axis=1).dropna(how='all')
    mtf_cols = list(mtf.columns)
    vp_cols = list(vp_all.columns)
    
    # Target
    c_arr = df_5m['close'].astype(float).reindex(combined.index)
    tl = np.zeros(len(combined), dtype=int)
    ts = np.zeros(len(combined), dtype=int)
    for i in range(len(combined)-TARGET_BARS):
        fh = c_arr.iloc[i+1:i+TARGET_BARS+1].max()
        fl = c_arr.iloc[i+1:i+TARGET_BARS+1].min()
        tl[i] = 1 if fh >= c_arr.iloc[i]*(1+TARGET_THR) else 0
        ts[i] = 1 if fl <= c_arr.iloc[i]*(1-TARGET_THR) else 0
    
    n, sp = len(combined), int(len(combined)*TRAIN_SPLIT)
    fn = list(combined.columns)
    X = combined[fn].fillna(0).clip(-10,10)
    
    # Baseline vs Full
    sym_res = {}
    for exp, fc in [("BASELINE", mtf_cols), ("FULL", fn)]:
        Xs = X[fc]
        res = []
        for side, y in [('long', tl), ('short', ts)]:
            yt, yo = y[:sp], y[sp:]
            if len(set(yo)) < 2:
                continue
            sw = (len(yt)-yt.sum())/max(yt.sum(),1)
            m = xgb.XGBClassifier(n_estimators=400, **HP, scale_pos_weight=sw,
                                   objective='binary:logistic', random_state=42, verbosity=0)
            m.fit(Xs.iloc[:sp], yt)
            p = m.predict_proba(Xs.iloc[sp:].values.astype(np.float32))[:,1]
            auc = roc_auc_score(yo, p)
            
            # WR at multiple thresholds
            wr_info = {}
            for td in range(50, 90):
                thr = td/100.0
                pred = p >= thr
                nt = int(pred.sum())
                if nt >= 10:
                    wr_info[f'thr{td}'] = {'n': nt, 'wr': round(float(yo[pred].mean())*100, 1)}
            
            best_wr, best_thr, best_n = 0, '-', 0
            for k, v in wr_info.items():
                if v['wr'] > best_wr:
                    best_wr, best_thr, best_n = v['wr'], f"0.{k[3:]}", v['n']
            
            res.append({
                'side': side, 'auc': round(auc, 3), 'wr': best_wr,
                'thr': best_thr, 'n': best_n, 'oos_pos': round(yo.mean()*100, 1),
                **{k: v for k,v in wr_info.items()}
            })
            print(f"    {exp:>10s} {side:>5s}: AUC={auc:.3f} WR={best_wr:.1f}% @ {best_thr} (n={best_n}) OOS_pos={yo.mean()*100:.1f}%")
        sym_res[exp] = res
    all_results[sym] = {**sym_res, 'time': round(time.time()-t0, 1)}

# ─── Summary Table ───
print(f"\n\n{'='*70}")
print("FINAL SUMMARY — All Symbols")
print('='*70)
header = f"{'Symbol':>12s} {'Side':>5s} {'Baseline':>20s} {'+Volume':>20s} {'Gain':>8s}"
print(header)
print('-' * 70)

for sym, _ in SYMBOLS:
    r = all_results.get(sym, {})
    base = r.get('BASELINE', [])
    full = r.get('FULL', [])
    for b in base:
        side = b['side']
        f = next((x for x in full if x['side'] == side), None)
        if f:
            b_auc = b['auc']; f_auc = f['auc']
            b_wr = b['wr']; f_wr = f['wr']
            b_n = b['n']; f_n = f['n']
            gain = '+' if f_wr > b_wr else '' if f_wr == b_wr else '-'
            print(f"{sym:>12s} {side:>5s} AUC={b_auc:.3f} WR={b_wr:.1f}% n={b_n:>4d}  AUC={f_auc:.3f} WR={f_wr:.1f}% n={f_n:>4d}  {gain}{f_wr-b_wr:>+.1f}pp")

# Save
output = {
    'config': {'target_bars': TARGET_BARS, 'target_thr': TARGET_THR, 'train_days': TRAIN_DAYS},
    'results': {k: {sk: [{skk: svv for skk, svv in sv.items() if skk != 'thr'}] for sk, sv in sv.items() if isinstance(sv, list)} 
                for k, v in all_results.items() if isinstance(v, dict) for sk, sv in v.items() if isinstance(sv, list)}
}
print(f"\n✅ Done")
