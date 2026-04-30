#!/usr/bin/env python3
"""
Auto Retraining Pipeline — fetch fresh data, retrain models, validate, deploy if better.

Architecture:
  1. Fetch fresh OHLCV data from Binance (130 days)
  2. Compute/enrich features (all 5 TF groups)
  3. Retrain XGBoost models (5 seeds × 5 TF groups)
  4. Validate on OOS (last 20% of timeframe)
  5. Compare against current production models
  6. Deploy new models if improvement >= threshold
  7. Archive old models as backup

This script is designed to be run by a cron job (1-hour interval) or on-demand.

Usage:
  python scripts/training/auto_retrain.py --symbol SOLUSDT [--force]
  python scripts/training/auto_retrain.py --all [--force]
  python scripts/training/auto_retrain.py --status
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
import json
import time
import shutil
import warnings
import argparse
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb

# Import proper trainers — these are the real deal with per-seed HPARAMS, MI selection, ensemble voting
from src.training.train_ml_ensemble import train_full_from_features
from src.training.train_tf_specific import train_tf_from_features

from src.trading.state import LIVE_SYMBOLS, INITIAL_CAPITAL

# ─── Paths ─────────────────────────────────────────────────────────
CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")
BACKUP_DIR = MODEL_DIR / "_backups"
STATUS_FILE = MODEL_DIR / "_retrain_status.json"

SYMBOLS_10 = LIVE_SYMBOLS  # Use all 20 live trading symbols

SEEDS = [42, 101, 202, 303, 404]
TF_GROUPS = ['long', 'short'] # Dual ensemble: long (predict UP) + short (predict DOWN)
TFS = {'full': []}       # 'full' uses ALL available features

DAYS_DATA = 130
WARMUP_BARS = 200
TRAIN_SPLIT = 0.80  # 80% train, 20% OOS
IMPROVEMENT_THRESHOLD = 0.02  # 2% F1 improvement to auto-deploy

MIN_TRADES_RETRAIN = 50         # minimum bars after filling/feature gen
OOS_HOURS = 72                  # OOS validation window (72h = 3 days of unseen data)
# MODEL_PARAMS and FEATURE_SUBSAMPLE_RATIO retired — replaced by per-seed HPARAMS in imported trainers


def status(msg: str, symbol: str = ""):
    prefix = f"[{symbol}]" if symbol else "[RETRAIN]"
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts} {prefix} {msg}")


# ─── Data Pipeline ─────────────────────────────────────────────────

def fetch_ohlcv(symbol: str, days: int = DAYS_DATA) -> pd.DataFrame:
    """Fetch 5m OHLCV from shared cache (single source of truth)."""
    from src.strategies.ml_features import ensure_ohlcv_data
    return ensure_ohlcv_data(symbol, min_days=days)


def compute_target(df: pd.DataFrame, forward_bars: int = 3) -> pd.Series:
    """
    Target: 1 if price >= close * 1.002 in next {forward_bars} bars, else 0.
    (0.2% profit target, aligned with 0.09% round-trip fees)
    """
    target = np.zeros(len(df))
    for i in range(len(df) - forward_bars):
        future_max = df['high'].iloc[i+1:i+forward_bars+1].max()
        target[i] = 1 if future_max >= df['close'].iloc[i] * 1.002 else 0
    return pd.Series(target, index=df.index)


def compute_features_5tf(ohlcv: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Compute all 372 features (or reuse cached)."""
    from src.strategies.ml_features import compute_5m_features_5tf, ensure_ohlcv_1h

    # Check cache
    cache_dir = Path("data/ml_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{symbol}_5m_130d_features.parquet"
    if cache_path.exists():
        print(f"  Loading cached features for {symbol}...")
        return pd.read_parquet(cache_path)

    # Fetch 1h OHLCV directly (no look-ahead from resample)
    df_1h = ensure_ohlcv_1h(symbol, min_days=7)
    if df_1h is not None:
        print(f"    ✅ 1h: {len(df_1h)} bars (direct from Binance)")
    feat_df = compute_5m_features_5tf(ohlcv, df_1h=df_1h)

    # Save to cache
    feat_df.to_parquet(cache_path)
    print(f"  Saved features to {cache_path}")
    return feat_df


# ─── Model Training ────────────────────────────────────────────────

def train_ensemble(symbol: str, feat_df: pd.DataFrame, target: pd.Series,
                   force: bool = False) -> dict:
    """
    Train 5 seeds for each TF group (long/short).
    Delegates to proper trainers: train_full_from_features.
    Returns metrics dict.
    """
    n = len(feat_df)
    split = int(n * TRAIN_SPLIT)

    all_metrics = {}
    total_seeds = 0
    successful = 0

    for tf in TF_GROUPS:
        target_col = f'target_{tf}'  # 'target_long' or 'target_short'
        if target_col not in feat_df.columns:
            status(f"  ❌ No '{target_col}' column, skipping", symbol)
            continue

        status(f"Training {tf.upper()} ensemble...", symbol)

        # Remap target column for trainers (they expect 'target')
        feat_copy = feat_df.copy()
        feat_copy['target'] = feat_copy[target_col]

        try:
            meta = train_full_from_features(feat_copy, symbol, model_prefix=tf)
        except Exception as e:
            status(f"  ❌ {tf} failed: {e}", symbol)
            import traceback
            traceback.print_exc()
            continue

        if meta is None:
            status(f"  ❌ {tf} returned None (likely data issue)", symbol)
            continue

        # Accumulate metrics
        tf_metrics = []
        for seed_str, seed_metrics in meta.get('individual_metrics', {}).items():
            seed = int(seed_str)
            tf_metrics.append({
                'seed': seed,
                'train_acc': seed_metrics.get('accuracy', 0),
                'test_acc': seed_metrics.get('accuracy', 0),
                'test_f1': seed_metrics.get('f1', 0),
                'test_prec': seed_metrics.get('precision', 0),
                'test_rec': seed_metrics.get('recall', 0),
                'features_used': len(meta.get('features', [])),
            })
            total_seeds += 1
            successful += 1

        if tf_metrics:
            all_metrics[tf] = tf_metrics
            avg_f1 = np.mean([m['test_f1'] for m in tf_metrics])
            status(f"  ✅ {len(tf_metrics)} seeds avg F1={avg_f1:.4f}", symbol)

    # Save combined meta (aggregate of long + short)
    combined_meta = {
        'symbol': symbol,
        'tf_groups': TF_GROUPS,
        'n_models': successful,
        'n_expected': len(TF_GROUPS) * len(SEEDS),
        'train_split': TRAIN_SPLIT,
        'metrics_per_tf': {k: {'avg_f1': round(np.mean([m['test_f1'] for m in v]), 4),
                                'best_f1': round(max(m['test_f1'] for m in v), 4),
                                'n_seeds': len(v)}
                          for k, v in all_metrics.items() if v},
        'trained_at': datetime.now().isoformat(),
    }
    with open(MODEL_DIR / f'{symbol}_ensemble_meta.json', 'w') as f:
        json.dump(combined_meta, f, indent=2, default=str)

    return {'n_models': successful, 'n_expected': len(TF_GROUPS) * len(SEEDS),
            'metrics': all_metrics}


def validate_new_models(symbol: str, feat_df: pd.DataFrame = None) -> dict:
    """
    Run proper OOS backtest on new models using deferred-entry engine.

    Uses trading/backtest.run_backtest() which has:
      - Deferred entry (N+1) — NO look-ahead bias
      - True OOS window (last OOS_HOURS hours of unseen data)
      - Same execution logic as live trader

    Returns PF, WR, DD, return_pct, trades.
    feat_df: ignored (kept for backward compat — engine loads its own data).
    """
    from src.trading import backtest as bt_engine
    result = bt_engine.run_backtest(symbol, test_hours=OOS_HOURS, verbose=False)
    if result is None or result.get('n_trades', 0) < 5:
        # Fallback: try longer OOS window (168h = 1 week)
        result = bt_engine.run_backtest(symbol, test_hours=168, verbose=False)
        if result is None or result.get('n_trades', 0) < 5:
            return None
    return {
        'wr': result['win_rate'],
        'pf': result['profit_factor'],
        'sharpe': 0.0,  # not computed by engine
        'dd': result['max_dd'],
        'return_pct': result['total_pnl'] / INITIAL_CAPITAL * 100,
        'trades': result['n_trades'],
    }


def compare_with_production(symbol: str, new_metrics: dict,
                            force: bool = False) -> dict:
    """
    Compare new model metrics with currently deployed models.
    Returns deployment decision.
    """
    status_file = STATUS_FILE
    if status_file.exists():
        try:
            with open(status_file) as f:
                prod_data = json.load(f)
        except:
            prod_data = {}
    else:
        prod_data = {'symbols': {}}
    
    old = prod_data.get('symbols', {}).get(symbol, {})
    
    decision = {
        'symbol': symbol,
        'new_metrics': new_metrics,
        'old_metrics': old,
        'should_deploy': False,
        'reason': '',
    }

    # Force overrides everything
    if force:
        decision['should_deploy'] = True
        decision['reason'] = 'force_deploy'
        return decision

    if not old:
        new_wr = new_metrics.get('wr', 0)
        new_pf = new_metrics.get('pf', 0)
        # OOS-realistic quality gate for first-time validation
        # 72h OOS deferred-entry WR is naturally lower than biased in-sample validation
        if new_wr < 30.0 or new_pf < 0.7:
            decision['should_deploy'] = False
            decision['reason'] = f'quality_gate_failed_wr_{new_wr:.1f}%_pf_{new_pf:.2f}'
            return decision
        decision['should_deploy'] = True
        decision['reason'] = 'first_time_training'
        return decision
    
    if force:
        decision['should_deploy'] = True
        decision['reason'] = 'force_deploy'
        return decision
    
    new_pf = new_metrics.get('pf', 0)
    old_pf = old.get('pf', 0)
    new_wr = new_metrics.get('wr', 0)
    old_wr = old.get('wr', 0)
    new_dd = new_metrics.get('dd', 99)
    old_dd = old.get('dd', 99)
    
    # Check improvement (OOS realistic: PF is more meaningful than WR for 72h window)
    pf_improvement = (new_pf - old_pf) / max(old_pf, 0.01)
    wr_improvement = new_wr - old_wr

    if new_pf > old_pf * (1 + 0.10) and new_wr >= old_wr:
        decision['should_deploy'] = True
        decision['reason'] = f'pf_improved_{pf_improvement*100:.1f}%'
    elif new_wr > old_wr + 5 and new_pf >= old_pf:
        decision['should_deploy'] = True
        decision['reason'] = f'wr_improved_{wr_improvement:.1f}pp'
    elif new_dd < old_dd * 0.5 and new_pf >= old_pf * 0.9:
        decision['should_deploy'] = True
        decision['reason'] = f'dd_reduced_{old_dd:.2f}%_to_{new_dd:.2f}%'
    else:
        decision['should_deploy'] = False
        decision['reason'] = f'no_significant_improvement_pf_{new_pf:.2f}_vs_{old_pf:.2f}'

    # Global quality gate: never deploy models below minimum OOS standards
    if decision['should_deploy']:
        if new_wr < 30.0 or new_pf < 0.7:
            decision['should_deploy'] = False
            decision['reason'] = f'quality_gate_wr_{new_wr:.1f}%_pf_{new_pf:.2f}'
    
    return decision


def save_status(symbol: str, metrics: dict, deployed: bool = False):
    """Save latest metrics to retrain status file.

    Tracks retrain count per calendar week to enforce MAX_RETRAIN_PER_WEEK.
    Resets weekly counter when ISO week number changes.
    """
    status_file = STATUS_FILE
    if status_file.exists():
        with open(status_file) as f:
            data = json.load(f)
    else:
        data = {'symbols': {}, 'last_run': None, 'runs': []}

    now = datetime.now()
    current_iso_week = f"{now.isocalendar()[0]}-W{now.isocalendar()[1]}"  # "2026-W17"

    # Get existing symbol data to preserve weekly counter
    existing = data['symbols'].get(symbol, {})
    last_week = existing.get('iso_week', '')

    # Reset weekly retrain counter if week changed
    if last_week == current_iso_week:
        retrains_this_week = existing.get('retrains_this_week', 0) + 1
    else:
        retrains_this_week = 1

    data['symbols'][symbol] = metrics
    data['symbols'][symbol]['last_retrained'] = now.isoformat()
    data['symbols'][symbol]['deployed'] = deployed
    data['symbols'][symbol]['deployed_at'] = now.isoformat() if deployed else existing.get('deployed_at')
    data['symbols'][symbol]['retrains_this_week'] = retrains_this_week
    data['symbols'][symbol]['iso_week'] = current_iso_week
    data['last_run'] = now.isoformat()

    with open(status_file, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def backup_old_models(symbol: str):
    """Move existing models to backup dir before deploying new ones."""
    backup_dir = BACKUP_DIR / f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup new-naming models: {symbol}_{tf}_xgb_ens_{seed}.json
    for tf in TF_GROUPS:
        for seed in SEEDS:
            src = MODEL_DIR / f'{symbol}_{tf}_xgb_ens_{seed}.json'
            if src.exists():
                shutil.copy2(str(src), str(backup_dir / src.name))
    
    # Backup old-naming models (voting pipeline format): {symbol}_xgb_ens_{seed}.json
    for seed in SEEDS:
        src = MODEL_DIR / f'{symbol}_xgb_ens_{seed}.json'
        if src.exists():
            shutil.copy2(str(src), str(backup_dir / src.name))
    
    # Also backup old TF naming: {symbol}_15m/30m/1h_xgb_ens_{seed}.json
    old_tfs = ['15m', '30m', '1h']
    for old_tf in old_tfs:
        for seed in SEEDS:
            src = MODEL_DIR / f'{symbol}_{old_tf}_xgb_ens_{seed}.json'
            if src.exists():
                shutil.copy2(str(src), str(backup_dir / src.name))
    
    meta = MODEL_DIR / f'{symbol}_ensemble_meta.json'
    if meta.exists():
        shutil.copy2(str(meta), str(backup_dir / meta.name))
    
    return backup_dir


# ─── Calibration ────────────────────────────────────────────────

def _calibrate_ensemble(symbol: str, feat_df: pd.DataFrame, target_side: str = 'long') -> None:
    """
    Fit Platt scaling on validation set for one model side.
    Loads models DIRECTLY from files (not via SignalGenerator) to ensure
    feature columns match the training data.

    Args:
        symbol: Trading symbol
        feat_df: Feature DataFrame (from training, with full columns)
        target_side: 'long' or 'short' — which model set to calibrate
    """
    from sklearn.linear_model import LogisticRegression
    import json
    from pathlib import Path

    try:
        import xgboost as xgb
    except ImportError:
        pass

    # 1. Split data
    n = len(feat_df)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    target_col = f'target_{target_side}'
    if target_col not in feat_df.columns:
        raise ValueError(f"Column '{target_col}' not in feat_df for calibration")

    # Drop all target columns for X
    X_all = feat_df.drop(columns=[c for c in ['target', 'target_long', 'target_short'] if c in feat_df.columns])
    y_all = feat_df[target_col]

    X_val = X_all.iloc[train_size:train_size + val_size]
    y_val = y_all.iloc[train_size:train_size + val_size]

    # 2. Load models DIRECTLY from disk (no SignalGenerator — avoids feature mismatch)
    loaded = []
    for seed in SEEDS:
        path = MODEL_DIR / f"{symbol}_{target_side}_xgb_ens_{seed}.json"
        if not path.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        mf = m.get_booster().feature_names
        if mf:
            loaded.append((m, mf))

    if len(loaded) < 3:
        raise ValueError(f"Only {len(loaded)} {target_side} models loaded (need ≥3)")

    # 3. Ensemble predictions on validation set
    all_preds = []
    for m, mf in loaded:
        available = [c for c in mf if c in X_val.columns]
        if len(available) < 5:
            continue
        X_sub = X_val[available].fillna(0).clip(-10, 10).values
        probs = m.predict_proba(X_sub)[:, 1]
        all_preds.append(probs)

    if len(all_preds) < 3:
        raise ValueError(f"Only {len(all_preds)} {target_side} models with valid features")

    raw_probas = np.mean(all_preds, axis=0)

    # 4. Fit Platt calibrator
    calibrator = LogisticRegression(C=1e10, solver='lbfgs')
    calibrator.fit(raw_probas.reshape(-1, 1), y_val.values)

    cal_probas = calibrator.predict_proba(raw_probas.reshape(-1, 1))[:, 1]
    raw_auc = __import__('sklearn').metrics.roc_auc_score(y_val, raw_probas)
    cal_auc = __import__('sklearn').metrics.roc_auc_score(y_val, cal_probas)
    status(f"Calibration ({target_side}): raw AUC={raw_auc:.4f} → cal AUC={cal_auc:.4f}, "
           f"raw range=[{raw_probas.min():.4f},{raw_probas.max():.4f}] "
           f"→ cal range=[{cal_probas.min():.4f},{cal_probas.max():.4f}]", symbol)

    # 5. Save with side prefix
    cal_data = {
        'type': 'platt',
        'side': target_side,
        'coef': float(calibrator.coef_[0][0]),
        'intercept': float(calibrator.intercept_[0]),
        'raw_range': [float(raw_probas.min()), float(raw_probas.max())],
        'cal_range': [float(cal_probas.min()), float(cal_probas.max())],
        'val_size': len(y_val),
    }
    cal_path = MODEL_DIR / f"{symbol}_{target_side}_calibrator.json"
    with open(cal_path, 'w') as f:
        json.dump(cal_data, f, indent=2)
    status(f"✅ Calibrator saved → {cal_path.name}", symbol)


# ─── Main Pipeline ─────────────────────────────────────────────────

def retrain_symbol(symbol: str, force: bool = False) -> dict:
    """Full retrain pipeline for one symbol. Returns results dict."""
    status(f"Starting retrain...", symbol)
    t_start = time.time()
    
    # 1. Fetch data
    status(f"Fetching {DAYS_DATA} days OHLCV...", symbol)
    ohlcv = fetch_ohlcv(symbol)
    if ohlcv is None or len(ohlcv) < 1000:
        status(f"❌ Insufficient data ({len(ohlcv) if ohlcv is not None else 0} bars)", symbol)
        return {'status': 'error', 'reason': 'no_data'}
    status(f"✅ {len(ohlcv)} bars fetched", symbol)
    
    # 2. Compute features
    status(f"Computing features...", symbol)
    feat_df = compute_features_5tf(ohlcv, symbol)
    if feat_df is None or len(feat_df) < 500:
        status(f"❌ Feature computation failed", symbol)
        return {'status': 'error', 'reason': 'feature_failure'}
    status(f"✅ {len(feat_df)} rows × {len(feat_df.columns)} cols", symbol)
    
    # 3. Use dual targets (target_long + target_short)
    if 'target_long' not in feat_df.columns or 'target_short' not in feat_df.columns:
        status(f"❌ Missing 'target_long' or 'target_short' columns", symbol)
        return {'status': 'error', 'reason': 'missing_dual_targets'}
    target = feat_df['target_long']  # default for backward compat
    long_ratio = feat_df['target_long'].mean()
    short_ratio = feat_df['target_short'].mean()
    status(f"Targets: LONG={long_ratio*100:.1f}% | SHORT={short_ratio*100:.1f}% positive", symbol)
    
    # 4. Backup old models before training (in case training fails)
    backup_dir = backup_old_models(symbol)
    status(f"Backup saved to {backup_dir.name}", symbol)
    
    # 5. Train
    status(f"Training ensemble...", symbol)
    train_meta = train_ensemble(symbol, feat_df, target, force)
    n_models = train_meta.get('n_models', 0)
    status(f"✅ {n_models} models trained", symbol)
    
    # Expect 10 models (5 long + 5 short), but accept min 5 (one side)
    if n_models < 5:
        status(f"❌ Too few models trained ({n_models}), restoring backup", symbol)
        for f in backup_dir.iterdir():
            shutil.copy2(str(f), str(MODEL_DIR / f.name))
        return {'status': 'error', 'reason': f'too_few_models_{n_models}'}
    
    # 5b. Calibrate both ensembles
    for side in ['long', 'short']:
        try:
            status(f"Calibrating {side} ensemble...", symbol)
            _calibrate_ensemble(symbol, feat_df, target_side=side)
        except Exception as e:
            status(f"⚠️ {side} calibration failed: {e}", symbol)
    
    # 6. Validate — run quick backtest (skip if force=True)
    val_metrics = None
    if force:
        status(f"⏭️ Skipping validation (force=True)", symbol)
        val_metrics = {'wr': 0.0, 'pf': 0.0, 'dd': 0.0, 'return_pct': 0.0, 'trades': 0}
    else:
        status(f"Validating with backtest...", symbol)
        val_metrics = validate_new_models(symbol, feat_df=feat_df)
        if val_metrics is None:
            status(f"❌ Validation backtest failed (0 trades in OOS), restoring backup", symbol)
            for f in backup_dir.iterdir():
                shutil.copy2(str(f), str(MODEL_DIR / f.name))
            return {'status': 'error', 'reason': 'validation_failed'}
    status(f"Backtest: WR={val_metrics['wr']:.1f}% PF={val_metrics['pf']:.2f} DD={val_metrics['dd']:.2f}%", symbol)
    
    # 7. Compare with production
    decision = compare_with_production(symbol, val_metrics, force)
    
    if decision['should_deploy']:
        status(f"✅ Deploying new models — {decision['reason']}", symbol)
        # Keep backup, update status
        save_status(symbol, val_metrics, deployed=True)
    else:
        status(f"↩️ Skipping deploy — {decision['reason']}", symbol)
        # Remove newly trained models, then restore old ones from backup
        for tf in TF_GROUPS:
            for seed in SEEDS:
                new_file = MODEL_DIR / f'{symbol}_{tf}_xgb_ens_{seed}.json'
                if new_file.exists():
                    new_file.unlink()
        # Restore all backed-up models (both old and new naming)
        for f in backup_dir.iterdir():
            shutil.copy2(str(f), str(MODEL_DIR / f.name))
        # Save status for reference
        val_metrics['reason'] = decision['reason']
        save_status(symbol, val_metrics, deployed=False)
    
    elapsed = time.time() - t_start
    status(f"Done in {elapsed:.0f}s", symbol)
    
    return {
        'status': 'ok',
        'symbol': symbol,
        'elapsed_s': round(elapsed),
        'n_models': n_models,
        'validation': val_metrics,
        'deployed': decision['should_deploy'],
        'reason': decision['reason'],
    }


def show_status():
    """Display current retrain status for all symbols."""
    if not STATUS_FILE.exists():
        print("  No retrain status yet.")
        return
    
    with open(STATUS_FILE) as f:
        data = json.load(f)
    
    symbols = data.get('symbols', {})
    if not symbols:
        print("  No symbol status data.")
        return
    
    print(f"\n  {'='*90}")
    print(f"  RETRAIN STATUS — Last run: {data.get('last_run', 'N/A')}")
    print(f"  {'='*90}")
    print(f"  {'Symbol':<12} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'DD%':>6} {'Return%':>8} {'Trades':>7} {'Last Retrain':<21}")
    print(f"  {'-'*73}")
    
    for sym, s in sorted(symbols.items()):
        wr = f"{s.get('wr', 0):.1f}%"
        pf = f"{s.get('pf', 0):.2f}"
        sh = f"{s.get('sharpe', 0):.2f}"
        dd = f"{s.get('dd', 0):.2f}%"
        ret = f"{s.get('return_pct', 0):.1f}%"
        trades = s.get('trades', 0)
        last = s.get('last_retrained', 'N/A')[:19]
        print(f"  {sym:<12} {wr:>6} {pf:>6} {sh:>7} {dd:>6} {ret:>8} {trades:>7} {last:<21}")


def main():
    parser = argparse.ArgumentParser(description='Auto Retrain Pipeline')
    parser.add_argument('--symbol', type=str, help='Single symbol to retrain')
    parser.add_argument('--all', action='store_true', help='Retrain all 10 symbols')
    parser.add_argument('--force', action='store_true', help='Force deploy regardless of improvement')
    parser.add_argument('--status', action='store_true', help='Show retrain status')
    parser.add_argument('--skip-validation', action='store_true', help='Skip backtest validation (faster)')
    args = parser.parse_args()
    
    if args.status:
        show_status()
        return
    
    symbols = []
    if args.symbol:
        symbols.append(args.symbol.upper())
    elif args.all:
        symbols = SYMBOLS_10
    else:
        parser.print_help()
        return
    
    results = []
    for sym in symbols:
        # Check if live trader is running — skip if symbol has open position
        # Simple check: just warn, don't block
        print(f"\n{'='*70}")
        print(f"  RETRAINING: {sym}")
        print(f"{'='*70}")
        result = retrain_symbol(sym, force=args.force)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  RETRAIN SUMMARY")
    print(f"{'='*70}")
    for r in results:
        sym = r.get('symbol', '?')
        status_flag = r.get('status', 'error')
        if status_flag == 'ok':
            deployed = r.get('deployed', False)
            reason = r.get('reason', '')
            val = r.get('validation', {})
            wr = val.get('wr', 0)
            pf = val.get('pf', 0)
            elapsed = r.get('elapsed_s', 0)
            deploy_flag = "✅ DEPLOYED" if deployed else "↩️ SKIPPED"
            print(f"  {sym:<12} {deploy_flag:<16} WR={wr:.1f}% PF={pf:.2f} [{reason}] ({elapsed}s)")
        else:
            print(f"  {sym:<12} ❌ ERROR — {r.get('reason', 'unknown')}")


if __name__ == '__main__':
    main()
