#!/usr/bin/env python3
"""
Mimia — Signal Replay Engine
==============================
Replay a specific historical signal with FROZEN context:
1. Frozen OHLCV data (truncated to data_cutoff)
2. Frozen model version (checksum-matched)

Usage:
  python scripts/replay_signal.py --symbol ENAUSDT --trade-id 1104
  python scripts/replay_signal.py --last 5          # Last 5 signals
  python scripts/replay_signal.py --trade-id 1104   # Specific trade ID
"""
import sys, json, time as ttime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import warnings; warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from src.trading.state import (
    DB_PATH, THRESHOLD, HOLD_BARS, COOLDOWN_BARS,
    TAKER_FEE, SLIPPAGE, POSITION_PCT, INITIAL_CAPITAL,
    get_model_info,
)
from src.strategies.ml_features import (
    resample_to_timeframes, compute_technical_features
)
from src.trading.signals import SignalGenerator

# ─── Config ────────────────────────────────
TF_RULES = {'15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h'}
FILL_LIMITS = {'15m': 3, '30m': 6, '1h': 24, '4h': 96}
TF_ORDER = ['15m', '30m', '1h', '4h']
TF_PREFIXES = {'15m': 'm15_', '30m': 'm30_', '1h': 'h1_', '4h': 'h4_'}


def get_valid_bar_start(T: datetime, tf_name: str) -> datetime:
    """Get START of the last COMPLETE higher-TF bar at cutoff T."""
    rule = pd.Timedelta(TF_RULES[tf_name])
    T_minus_rule = T - rule
    m = T_minus_rule.minute
    if tf_name == '15m':
        return T_minus_rule.replace(minute=(m // 15) * 15, second=0, microsecond=0)
    elif tf_name == '30m':
        return T_minus_rule.replace(minute=(m // 30) * 30, second=0, microsecond=0)
    elif tf_name == '1h':
        return T_minus_rule.replace(minute=0, second=0, microsecond=0)
    elif tf_name == '4h':
        return T_minus_rule.replace(hour=(T_minus_rule.hour // 4) * 4, minute=0, second=0, microsecond=0)
    return T_minus_rule


def replay_signal(symbol: str, entry_time_ms: int, live_proba: float,
                  data_cutoff_ms: Optional[int] = None,
                  model_info_json: Optional[str] = None) -> Dict:
    """Replay one signal with frozen data and model context."""
    symbol_clean = symbol.replace('_fwd', '')
    
    # Check model drift
    current_models = get_model_info()
    models_match = True
    if model_info_json:
        try:
            old = json.loads(model_info_json)
            cur = json.loads(current_models)
            # Check if ALL model files from old are still the same
            for fname, old_mtime in old.items():
                if fname not in cur or cur[fname] != old_mtime:
                    models_match = False
                    break
        except:
            models_match = False
    
    # Determine data cutoff
    cutoff_dt: Optional[datetime] = None
    if data_cutoff_ms:
        cutoff_dt = datetime.utcfromtimestamp(data_cutoff_ms / 1000)
    else:
        # Fallback: entry_time minus 1 bar (best estimate for old signals)
        cutoff_dt = datetime.utcfromtimestamp(entry_time_ms / 1000) - timedelta(minutes=5)
    
    # Round down to 5-min boundary
    cm = cutoff_dt.minute
    cutoff_dt = cutoff_dt.replace(minute=(cm // 5) * 5, second=0, microsecond=0)
    
    # Load data and models
    gen = SignalGenerator(symbol_clean)
    cached = gen._load_models(symbol_clean)
    if cached is None:
        return {'error': f'Cannot load models for {symbol_clean}'}
    
    model_groups = cached['groups']
    df_5m = gen._ensure_ohlcv_data(symbol_clean)
    if df_5m is None:
        return {'error': 'Cannot load OHLCV data'}
    
    # Pre-compute features
    feats_5m = compute_technical_features(df_5m, prefix='m5_')
    tf_data = resample_to_timeframes(df_5m, intervals=list(TF_RULES.keys()))
    tf_feats = {}
    for tf_name in TF_ORDER:
        tf_feats[tf_name] = compute_technical_features(tf_data[tf_name], prefix=TF_PREFIXES[tf_name])
    
    # Compute proba with frozen data cutoff
    idx_5m = feats_5m.index
    idx_up_to_T = idx_5m[idx_5m <= cutoff_dt]
    
    if len(idx_up_to_T) == 0:
        return {'error': f'No data up to cutoff {cutoff_dt}'}
    
    # Build aligned features
    feat_parts = [feats_5m.loc[idx_up_to_T]]
    for tf_name in TF_ORDER:
        tf_feat = tf_feats[tf_name]
        last_valid = get_valid_bar_start(cutoff_dt, tf_name)
        valid_tf = tf_feat[tf_feat.index <= last_valid]
        if len(valid_tf) > 0:
            aligned = valid_tf.reindex(idx_up_to_T, method='ffill',
                                       limit=FILL_LIMITS.get(tf_name, 12))
        else:
            aligned = pd.DataFrame(index=idx_up_to_T, columns=tf_feat.columns)
            aligned[:] = np.nan
        feat_parts.append(aligned)
    
    combined = pd.concat(feat_parts, axis=1)
    
    # Truncate to valid rows
    valid_mask = combined.notna().all(axis=1)
    if not valid_mask.any():
        return {'error': 'All NaN features after alignment'}
    last_valid_idx = combined.index[valid_mask][-1]
    combined = combined[combined.index <= last_valid_idx]
    last_row = combined.iloc[-1]
    
    # Model inference
    group_probas = []
    for tf_name_g, models in model_groups.items():
        tfp = []
        for _, m, mf in models:
            avail = [c for c in mf if c in combined.columns]
            if len(avail) < 5: continue
            X = last_row[avail].fillna(0).clip(-10, 10).values.reshape(1, -1)
            try:
                tfp.append(float(m.predict_proba(X)[0, 1]))
            except Exception as e:
                continue
        if tfp:
            group_probas.append(np.mean(tfp))
    
    replayed_proba = float(np.mean(group_probas)) if group_probas else None
    
    sig_replay = 0
    if replayed_proba is not None:
        if replayed_proba >= THRESHOLD: sig_replay = 1
        elif replayed_proba <= (1 - THRESHOLD): sig_replay = -1
    
    sig_live = 0
    if live_proba >= THRESHOLD: sig_live = 1
    elif live_proba <= (1 - THRESHOLD): sig_live = -1
    
    diff = abs((replayed_proba or 0) - live_proba)
    dir_match = (sig_replay == sig_live)
    
    return {
        'symbol': symbol_clean,
        'entry_time': datetime.utcfromtimestamp(entry_time_ms / 1000).strftime('%H:%M:%S'),
        'live_proba': live_proba,
        'replayed_proba': replayed_proba,
        'diff': diff,
        'live_signal': 'LONG' if sig_live == 1 else ('SHORT' if sig_live == -1 else 'FLAT'),
        'replayed_signal': 'LONG' if sig_replay == 1 else ('SHORT' if sig_replay == -1 else 'FLAT'),
        'direction_match': dir_match,
        'data_cutoff': cutoff_dt.strftime('%H:%M:%S'),
        'models_match': models_match,
        'n_bars_used': len(idx_up_to_T),
    }


# ─── CLI ───────────────────────────────────
if __name__ == '__main__':
    import sqlite3
    import argparse
    
    parser = argparse.ArgumentParser(description='Replay a historical signal with frozen context')
    parser.add_argument('--symbol', type=str, help='Symbol to replay')
    parser.add_argument('--trade-id', type=int, help='Trade ID from live_trades table')
    parser.add_argument('--last', type=int, help='Replay last N signals across all symbols')
    args = parser.parse_args()
    
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    
    targets = []
    
    if args.trade_id:
        # Get specific trade's entry signal
        cur.execute('''
            SELECT t.id, t.symbol, t.entry_time, t.entry_proba, 
                   COALESCE(s.data_cutoff, 0) as data_cutoff,
                   COALESCE(s.model_info, '') as model_info
            FROM live_trades t
            LEFT JOIN live_signals s ON s.symbol = t.symbol 
                AND abs(s.timestamp - t.entry_time) < 300000  -- within 5 min
            WHERE t.id = ?
            LIMIT 1
        ''', (args.trade_id,))
        row = cur.fetchone()
        if row:
            targets.append(row)
    
    if args.last:
        cur.execute('''
            SELECT t.id, t.symbol, t.entry_time, t.entry_proba, 
                   COALESCE(s.data_cutoff, 0) as data_cutoff,
                   COALESCE(s.model_info, '') as model_info
            FROM live_trades t
            LEFT JOIN live_signals s ON s.symbol = t.symbol 
                AND abs(s.timestamp - t.entry_time) < 300000
            WHERE t.entry_proba IS NOT NULL AND t.entry_proba != 0
            ORDER BY t.entry_time DESC
            LIMIT ?
        ''', (args.last,))
        targets = cur.fetchall()
    
    if args.symbol and not args.trade_id and not args.last:
        # Get all signals for a specific symbol
        cur.execute('''
            SELECT t.id, t.symbol, t.entry_time, t.entry_proba,
                   COALESCE(s.data_cutoff, 0) as data_cutoff,
                   COALESCE(s.model_info, '') as model_info
            FROM live_trades t
            LEFT JOIN live_signals s ON s.symbol = t.symbol 
                AND abs(s.timestamp - t.entry_time) < 300000
            WHERE t.symbol = ? AND t.entry_proba IS NOT NULL AND t.entry_proba != 0
            ORDER BY t.entry_time DESC
        ''', (args.symbol,))
        targets = cur.fetchall()
    
    conn.close()
    
    if not targets:
        print("❌ No matching signals found. Use --trade-id, --symbol, or --last")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"  SIGNAL REPLAY — {len(targets)} signal(s)")
    print(f"{'='*70}")
    
    results = []
    for tid, sym, et_ms, proba, dc, mi in targets:
        t0 = ttime.time()
        r = replay_signal(sym, et_ms, proba, 
                          data_cutoff_ms=dc if dc else None,
                          model_info_json=mi if mi else None)
        elapsed = ttime.time() - t0
        
        if 'error' in r:
            print(f"\n  ❌ Trade #{tid} {sym}: {r['error']}")
            continue
        
        results.append(r)
        
        icon = '✅' if r['direction_match'] and r['diff'] < 0.01 else \
               '⚠️' if r['direction_match'] else '❌'
        mi_icon = '🟢' if r['models_match'] else '🟡'
        
        print(f"\n  {icon} Trade #{tid} | {sym:10s} | {r['entry_time']}")
        print(f"     LIVE: proba={r['live_proba']:.4f} ({r['live_signal']})")
        print(f"     REPLAY: proba={r['replayed_proba']:.4f} ({r['replayed_signal']})")
        print(f"     Diff: {r['diff']:.4f} | Dir match: {r['direction_match']} | {elapsed:.1f}s")
        print(f"     Cutoff: {r['data_cutoff']} | {r['n_bars_used']} bars")
        print(f"     Models: {mi_icon} {'same' if r['models_match'] else 'CHANGED'}")
    
    # Summary
    if results:
        match_dir = sum(1 for r in results if r['direction_match'])
        match_proba = sum(1 for r in results if r['diff'] < 0.01)
        avg_diff = np.mean([r['diff'] for r in results])
        max_diff = max(r['diff'] for r in results)
        
        print(f"\n{'='*70}")
        print(f"  SUMMARY")
        print(f"{'='*70}")
        print(f"  Direction match: {match_dir}/{len(results)}")
        print(f"  Proba < 0.01:    {match_proba}/{len(results)}")
        print(f"  Avg diff:        {avg_diff:.4f}")
        print(f"  Max diff:        {max_diff:.4f}")
