#!/usr/bin/env python3
"""Quick diagnostic: compare resample outputs between backtest and live feature paths."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.trading.signals import SignalGenerator
from src.strategies.ml_features import resample_to_timeframes, compute_technical_features

# Config
TF_RULE = {'15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h'}
TF_PREFIX = {'15m': 'm15_', '30m': 'm30_', '1h': 'h1_', '4h': 'h4_'}
TF_DUR = {'15m': timedelta(minutes=15), '30m': timedelta(minutes=30),
          '1h': timedelta(hours=1), '4h': timedelta(hours=4)}

symbol = 'ENAUSDT'

# Load OHLCV
gen = SignalGenerator(symbol)
df_5m = gen._ensure_ohlcv_data(symbol)
print(f"OHLCV: {len(df_5m)} bars, last={df_5m.index[-1]}")

# Method 1 (Backtest): manual resample + dropna
td_backtest = df_5m.resample('1h').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum',
}).dropna()
td_backtest['low'] = td_backtest[['open', 'close', 'low']].min(axis=1)
td_backtest['high'] = td_backtest[['open', 'close', 'high']].max(axis=1)

# Method 2 (Live): resample_to_timeframes (drops incomplete last bar)
tf_live = resample_to_timeframes(df_5m)
td_live = tf_live['1h']

print(f"\n=== h1 bar comparison ===")
print(f"Backtest: {len(td_backtest)} bars, last={td_backtest.index[-1]} close={td_backtest['close'].iloc[-1]:.4f}")
print(f"Live:     {len(td_live)} bars, last={td_live.index[-1]} close={td_live['close'].iloc[-1]:.4f}")

# Check if they differ
if len(td_backtest) != len(td_live):
    print(f"\n🔴 LENGTH MISMATCH: BT={len(td_backtest)} Live={len(td_live)}")
    print(f"  BT last bar: {td_backtest.index[-1]} (may be incomplete — bar_end={td_backtest.index[-1] + timedelta(hours=1)} > last_ts={df_5m.index[-1]})")
    print(f"  Live: drops incomplete bar → last complete: {td_live.index[-1]}")
    
    # Compare features on the common bars
    common_idx = td_backtest.index[:-1]  # all except last
    feats_bt = compute_technical_features(td_backtest, prefix='h1_')
    feats_live = compute_technical_features(td_live, prefix='h1_')
    
    # Only compare bars present in both
    common_both = feats_bt.index.intersection(feats_live.index)
    print(f"\n  Common bars: {len(common_both)}")
    
    # Check close values
    close_bt = td_backtest.loc[common_both, 'close']
    close_live = td_live.loc[common_both, 'close']
    close_match = (close_bt.values == close_live.values).all()
    print(f"  OHLC close identical on common bars: {close_match}")
    
    # Check upper_wick feature
    if 'h1_upper_wick' in feats_bt.columns and 'h1_upper_wick' in feats_live.columns:
        uv_bt = feats_bt.loc[common_both, 'h1_upper_wick']
        uv_live = feats_live.loc[common_both, 'h1_upper_wick']
        
        # Find which bars differ
        diffs = np.where(~np.isclose(uv_bt.values, uv_live.values, rtol=1e-10, atol=1e-10))[0]
        if len(diffs) > 0:
            print(f"\n  🔴 h1_upper_wick differs on {len(diffs)}/{len(common_both)} bars!")
            for d in diffs[:3]:
                print(f"    @ {common_both[d]}: BT={uv_bt.values[d]:.4f} Live={uv_live.values[d]:.4f}")
        else:
            print(f"\n  ✅ h1_upper_wick identical on all {len(common_both)} common bars")
else:
    print(f"\n  ✅ Same number of bars")
    # Check values
    feats_bt = compute_technical_features(td_backtest, prefix='h1_')
    feats_live = compute_technical_features(td_live, prefix='h1_')
    for col in ['h1_upper_wick', 'h1_lower_wick', 'h1_body_pct']:
        if col in feats_bt.columns and col in feats_live.columns:
            same = (feats_bt[col].values == feats_live[col].values).all()
            print(f"  {col}: {'✅' if same else '🔴'} identical")

# Also check: does bar count difference = 1?
diff = abs(len(td_backtest) - len(td_live))
if diff == 1:
    print(f"\n✅ CONFIRMED: Single incomplete bar difference at the end")
    print(f"   Resample_to_timeframes drops the last incomplete h1 bar")
    print(f"   Backtest keeps it → uses incomplete bar values for last 5m bars")
elif diff > 1:
    print(f"\n❌ UNEXPECTED: {diff} bars difference")

# Now check: for h4
td4_bt = df_5m.resample('4h').agg({
    'open': 'first', 'high': 'max', 'low': 'min',
    'close': 'last', 'volume': 'sum',
}).dropna()
td4_bt['low'] = td4_bt[['open', 'close', 'low']].min(axis=1)
td4_bt['high'] = td4_bt[['open', 'close', 'high']].max(axis=1)
td4_live = tf_live['4h']

print(f"\n=== h4 bar comparison ===")
print(f"Backtest: {len(td4_bt)} bars, last={td4_bt.index[-1]}")
print(f"Live:     {len(td4_live)} bars, last={td4_live.index[-1]}")
if len(td4_bt) != len(td4_live):
    print(f"🔴 LENGTH MISMATCH: BT={len(td4_bt)} Live={len(td4_live)}")
else:
    print(f"✅ Same length")
