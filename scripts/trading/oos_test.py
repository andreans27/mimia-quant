#!/usr/bin/env python3
"""
Mimia OOS Test — 6 Month Walk-Forward
5 months training (in-sample) → 1 month test (out-of-sample)
No interference with existing code or live trading.
"""
import sys, json, os, warnings, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent  # = /root/projects/mimia-quant
sys.path.insert(0, str(ROOT))
os.chdir(str(ROOT))  # Ensure working dir is project root
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta

# ── OOS Temp Directory (no touch existing) ──
OOS_DIR = Path("data/oos_test")
OOS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = OOS_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
CACHE_DIR = OOS_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── Reuse existing functions ──
from src.strategies.ml_features import compute_5m_features_5tf
from src.backtesting.compare_exit_strategies import compute_proba, run_backtest
from src.utils.binance_client import BinanceRESTClient

# ── Constants ──
SEEDS = [42, 101, 202, 303, 404]
TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']
TAKER_FEE = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0

HPARAMS = {
    42:  {'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'DOGEUSDT', '1000PEPEUSDT']

def train_models(symbol, feat_df, output_dir):
    """Train 5 TF groups × 5 seeds = 25 models."""
    print(f"\n{'='*60}")
    print(f"  Training {symbol} — {len(feat_df)} rows on IS")
    print(f"{'='*60}")
    
    os.makedirs(output_dir, exist_ok=True)
    full_meta = {'model_features': {}, 'features': [], 'symbol': symbol}
    
    for tf_group in TF_GROUPS:
        print(f"\n  [{tf_group}] Training 5 models...")
        
        if tf_group == 'full':
            # Full group: use ALL feature columns — save as primary feature set
            feature_cols = [c for c in feat_df.columns if c != 'target']
            if not full_meta['features']:
                full_meta['features'] = feature_cols  # Save once for full group
            prefix = ''
        else:
            prefix = f'{tf_group}_'
            feature_cols = [c for c in feat_df.columns if c.startswith(prefix) and c != 'target']
        
        if len(feature_cols) < 10:
            print(f"    ⚠️ Only {len(feature_cols)} features for {tf_group}, skip")
            continue
        
        X = feat_df[feature_cols].fillna(0).clip(-10, 10).values
        y = feat_df['target'].values
        
        for seed in SEEDS:
            hparams = HPARAMS[seed].copy()
            hparams['random_state'] = seed
            hparams['eval_metric'] = 'logloss'
            hparams['early_stopping_rounds'] = 10
            hparams['verbosity'] = 0
            
            # Split 20% validation for early stopping
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            model = xgb.XGBClassifier(**hparams, n_estimators=500)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            # Save model
            if tf_group == 'full':
                fname = f'{symbol}_xgb_ens_{seed}.json'
            else:
                fname = f'{symbol}_{tf_group}_xgb_ens_{seed}.json'
            
            model.save_model(str(output_dir / fname))
            
            # Store feature names — use TF-qualified key to avoid overwrites
            full_meta['model_features'][f'{tf_group}_{seed}'] = feature_cols
            if tf_group == 'full':
                full_meta['features'] = feature_cols  # Full group = primary feature set
            
            val_probs = model.predict_proba(X_val)[:, 1]
            val_acc = (val_probs > 0.5).astype(int) == y_val
            print(f"    Seed {seed}: val_acc={val_acc.mean():.3f} ({model.best_iteration} trees)")
    
    # Save meta
    full_meta['n_models'] = len(SEEDS) * len(TF_GROUPS)
    with open(output_dir / f'{symbol}_ensemble_meta.json', 'w') as f:
        json.dump(full_meta, f, indent=2)
    
    return model


def load_oos_models(symbol, model_dir):
    """Load models from OOS temp dir, grouped by TF."""
    groups = {}
    meta_path = model_dir / f'{symbol}_ensemble_meta.json'
    if not meta_path.exists():
        print(f"    ❌ No meta at {meta_path}")
        return None
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    for tf in TF_GROUPS:
        models = []
        for seed in SEEDS:
            if tf == 'full':
                path = model_dir / f'{symbol}_xgb_ens_{seed}.json'
            else:
                path = model_dir / f'{symbol}_{tf}_xgb_ens_{seed}.json'
            
            if not path.exists():
                continue
            
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            mf = m.get_booster().feature_names or []
            if not mf:
                # Fallback using TF-qualified key
                mf = meta.get('model_features', {}).get(f'{tf}_{seed}', [])
            if not mf:
                mf = meta.get('features', [])  # Ultimate fallback
            
            # Fix prefix for TF-specific models
            if tf != 'full' and mf and not mf[0].startswith(f'{tf}_'):
                mf = [f'{tf}_{f}' for f in mf]
            
            models.append((str(seed), m, mf))
        
        if len(models) >= 2:
            groups[tf] = models
    
    return groups if len(groups) >= 2 else None


def run_oos(symbol):
    """Run full OOS test for one symbol."""
    print(f"\n{'#'*70}")
    print(f"  OOS TEST: {symbol}")
    print(f"{'#'*70}")
    
    # ── 1. Fetch 6 months data (isolated — no touch shared cache) ──
    print(f"\n📡 Fetching 6 months of 5m data (direct from Binance Futures)...")
    t0 = time.time()
    client = BinanceRESTClient(testnet=True)
    end = datetime.now()
    start = end - timedelta(days=185)
    
    from src.strategies.ml_features import _fetch_all_klines
    df_5m = _fetch_all_klines(client, symbol, "5m", int(start.timestamp()*1000), int(end.timestamp()*1000))
    if df_5m is None or len(df_5m) < 10000:
        print(f"  ❌ Insufficient data: {len(df_5m) if df_5m is not None else 0} bars")
        return None
    
    print(f"  ✅ {len(df_5m)} bars ({time.time()-t0:.0f}s)")
    
    # ── 2. Compute features ──
    print(f"🔧 Computing features...")
    t0 = time.time()
    feat_df = compute_5m_features_5tf(df_5m, target_candle=9)
    print(f"  ✅ {len(feat_df)} feature rows, {len([c for c in feat_df.columns if c != 'target'])} features ({time.time()-t0:.0f}s)")
    
    # ── 3. Split: 5 months IS, 1 month OOS ──
    timestamps = feat_df.index
    split_date = timestamps[-1] - timedelta(days=30)  # Last 30 days = OOS
    
    is_df = feat_df[feat_df.index < split_date].copy()
    oos_df = feat_df[feat_df.index >= split_date].copy()
    
    print(f"\n📊 Split:")
    print(f"  In-Sample:  {len(is_df)} rows ({is_df.index[0].strftime('%Y-%m-%d')} → {is_df.index[-1].strftime('%Y-%m-%d')})")
    print(f"  Out-of-Sample: {len(oos_df)} rows ({oos_df.index[0].strftime('%Y-%m-%d')} → {oos_df.index[-1].strftime('%Y-%m-%d')})")
    
    if len(is_df) < 5000 or len(oos_df) < 1000:
        print(f"  ❌ Split too small: IS={len(is_df)} OOS={len(oos_df)}")
        return None
    
    # ── 4. Train on IS ──
    print(f"\n🎯 Training on In-Sample...")
    t0 = time.time()
    train_models(symbol, is_df, MODEL_DIR)
    print(f"  ⏱ Training: {time.time()-t0:.0f}s")
    
    # ── 5. Load models ──
    print(f"\n📦 Loading models...")
    is_groups = load_oos_models(symbol, MODEL_DIR)
    if is_groups is None:
        print(f"  ❌ Model loading failed")
        return None
    print(f"  ✅ {len(is_groups)} TF groups loaded")
    
    # ── 6. Compute proba for both IS and OOS ──
    print(f"\n🔮 Predicting...")
    t0 = time.time()
    is_proba = compute_proba(symbol, is_groups, is_df)
    oos_proba = compute_proba(symbol, is_groups, oos_df)
    print(f"  ⏱ Prediction: {time.time()-t0:.0f}s")
    
    if is_proba is None or oos_proba is None:
        print(f"  ❌ Prediction failed")
        return None
    
    # ── 7. Backtest both ──
    print(f"\n📈 Backtesting...")
    
    # Build signal dfs
    is_sig = pd.DataFrame({'proba': is_proba}, index=is_df.index)
    oos_sig = pd.DataFrame({'proba': oos_proba}, index=oos_df.index)
    
    # Open prices for backtest
    from src.strategies.ml_features import resample_to_timeframes
    tf_data = resample_to_timeframes(df_5m)
    ohlcv_5m = tf_data['5m']
    
    # Align
    is_aligned = ohlcv_5m.join(is_sig, how='inner')
    oos_aligned = ohlcv_5m.join(oos_sig, how='inner')
    
    # Skip warmup
    is_aligned = is_aligned.iloc[100:].copy()
    oos_aligned = oos_aligned.iloc[100:].copy()
    
    print(f"  IS aligned: {len(is_aligned)} bars")
    print(f"  OOS aligned: {len(oos_aligned)} bars")
    
    # Run backtest with standard hold_exit strategy
    exit_config = {
        'sl_pct': 0, 'tp_pct': 0, 'trail_pct': 0,
        'signal_exit': False, 'trail_activation_pct': 0,
        'sl_activation_bar': 0, 'max_hold_bars': 0,
    }
    
    # Run for multiple thresholds
    results = {}
    for thresh in [0.55, 0.60, 0.65]:
        print(f"\n  Threshold = {thresh}")
        is_result = run_backtest(symbol, is_aligned, thresh, hold_bars=9, cooldown_bars=3, exit_config=exit_config)
        oos_result = run_backtest(symbol, oos_aligned, thresh, hold_bars=9, cooldown_bars=3, exit_config=exit_config)
        
        if is_result and oos_result:
            results[thresh] = {
                'is': {
                    'wr': is_result.get('win_rate_pct', 0),
                    'pf': is_result.get('profit_factor', 0),
                    'dd': is_result.get('max_drawdown_pct', 0),
                    'monthly': is_result.get('avg_monthly_return_pct', 0),
                    'trades': is_result.get('total_trades', 0),
                    'sharpe': is_result.get('sharpe_ratio', 0),
                    'sortino': is_result.get('sortino_ratio', 0),
                },
                'oos': {
                    'wr': oos_result.get('win_rate_pct', 0),
                    'pf': oos_result.get('profit_factor', 0),
                    'dd': oos_result.get('max_drawdown_pct', 0),
                    'monthly': oos_result.get('avg_monthly_return_pct', 0),
                    'trades': oos_result.get('total_trades', 0),
                    'sharpe': oos_result.get('sharpe_ratio', 0),
                    'sortino': oos_result.get('sortino_ratio', 0),
                },
            }
            print(f"    IS:  WR={is_result.get('win_rate_pct',0):.1f}% PF={is_result.get('profit_factor',0):.2f} DD={is_result.get('max_drawdown_pct',0):.2f}% Trades={is_result.get('total_trades',0)}")
            print(f"    OOS: WR={oos_result.get('win_rate_pct',0):.1f}% PF={oos_result.get('profit_factor',0):.2f} DD={oos_result.get('max_drawdown_pct',0):.2f}% Trades={oos_result.get('total_trades',0)}")
    
    return results


# ── MAIN ──
print("=" * 70)
print("  MIMIA OOS TEST — 5 Months Train / 1 Month Test")
print(f"  Temp Dir: {OOS_DIR.resolve()}")
print(f"  Symbols: {SYMBOLS}")
print("=" * 70)

all_results = {}
for sym in SYMBOLS:
    try:
        r = run_oos(sym)
        if r:
            all_results[sym] = r
        print(f"\n{'~'*70}")
    except Exception as e:
        print(f"\n  ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

# ── Final Summary Table ──
print("\n\n")
print("=" * 70)
print("  OOS TEST RESULTS — FINAL SUMMARY")
print("=" * 70)

for sym, results in all_results.items():
    print(f"\n{sym}:")
    print(f"{'Threshold':>10} {'IS WR':>8} {'IS PF':>8} {'IS DD':>8} {'OOS WR':>8} {'OOS PF':>8} {'OOS DD':>8} {'Degrad':>8}")
    print("-" * 78)
    for thresh, data in sorted(results.items()):
        is_d = data['is']
        oos_d = data['oos']
        degrad = is_d['wr'] - oos_d['wr']
        print(f"{thresh:>10.2f} {is_d['wr']:>7.1f}% {is_d['pf']:>7.2f} {is_d['dd']:>7.2f}% {oos_d['wr']:>7.1f}% {oos_d['pf']:>7.2f} {oos_d['dd']:>7.2f}% {degrad:>+7.1f}%")

# Save results
with open(OOS_DIR / 'results_summary.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

print(f"\n\n📁 Full results saved to: {OOS_DIR}/results_summary.json")
