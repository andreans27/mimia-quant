#!/usr/bin/env python3
"""
Batch 45-day training optimization — train + OOS validate all 20 symbols.
Runs symbols in parallel using subprocess for speed.

Usage: python scripts/training/batch_45d_optimize.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
import json
import time
import argparse
import subprocess
import warnings
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

from src.strategies.ml_features import (
    ensure_ohlcv_data, compute_5m_features_5tf,
    prepare_ml_dataset
)
from src.strategies.market_data_cache import ensure_all_market_data
from src.trading.state import LIVE_SYMBOLS, MODEL_DIR, SEEDS, THRESHOLD

# ─── Config ───
TRAIN_DAYS = 45           # Use 45 days of recent data
TRAIN_SPLIT = 0.80        # 80% train, 20% OOS
TARGET_CANDLE = 9
TARGET_THRESHOLD = 0.005

# HPARAMS matching auto_retrain's train_tf_ensemble
HPARAMS = {
    42:  {'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}


def train_symbol_45d(symbol: str) -> dict:
    """Train + validate ONE symbol with 45-day window. Returns metrics dict."""
    result = {'symbol': symbol, 'status': 'ok'}
    t0 = time.time()

    try:
        # 1. Fetch OHLCV (45 days)
        df_5m = ensure_ohlcv_data(symbol, min_days=TRAIN_DAYS)
        if df_5m is None or len(df_5m) < 1000:
            return {'symbol': symbol, 'status': 'fail', 'error': 'insufficient OHLCV'}

        # 2. Compute features
        feat_df = compute_5m_features_5tf(
            df_5m, target_candle=TARGET_CANDLE,
            target_threshold=TARGET_THRESHOLD,
            intervals=['1h'], for_inference=False
        )
        if len(feat_df) < 500:
            return {'symbol': symbol, 'status': 'fail', 'error': f'insufficient features: {len(feat_df)}'}

        result['rows'] = len(feat_df)
        result['features'] = len([c for c in feat_df.columns if c not in ('target','target_long','target_short')])

        # 3. Chronological split
        n = len(feat_df)
        split_idx = int(n * TRAIN_SPLIT)

        exclude = {'target', 'target_long', 'target_short'}
        feature_names = [c for c in feat_df.columns if c not in exclude]

        X = feat_df[feature_names].fillna(0).clip(-10, 10)

        # 4. Train LONG and SHORT models
        for side in ['long', 'short']:
            target_col = f'target_{side}'
            y = feat_df[target_col].values

            # Split
            X_train = X.iloc[:split_idx].values.astype(np.float32)
            y_train = y[:split_idx]
            X_test = X.iloc[split_idx:].values.astype(np.float32)
            y_test = y[split_idx:]

            # Class balance
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            scale_pos = n_neg / max(n_pos, 1)

            # Train 5 seeds
            models = []
            for seed in SEEDS:
                hp = HPARAMS[seed]
                model = xgb.XGBClassifier(
                    n_estimators=400,
                    max_depth=hp['max_depth'],
                    learning_rate=hp['learning_rate'],
                    subsample=hp['subsample'],
                    colsample_bytree=hp['colsample_bytree'],
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    min_child_weight=hp['min_child_weight'],
                    scale_pos_weight=scale_pos,
                    objective='binary:logistic',
                    eval_metric='auc',
                    random_state=seed,
                    verbosity=0,
                    use_label_encoder=False,
                )
                # Feature subsampling for diversity
                np.random.seed(seed)
                n_feats = X_train.shape[1]
                feat_mask = np.random.choice([True, False], size=n_feats, p=[0.85, 0.15])
                if feat_mask.sum() < 10:
                    feat_mask[:10] = True
                X_train_sub = X_train[:, feat_mask]
                X_test_sub = X_test[:, feat_mask]
                feats_used = [f for f, m in zip(feature_names, feat_mask) if m]

                model.fit(X_train_sub, y_train)

                # Save model
                model_path = MODEL_DIR / f"{symbol}_{side}_xgb_ens_{seed}.json"
                model.save_model(str(model_path))
                models.append((seed, model, feats_used))

            # Ensemble inference
            test_probas = np.zeros((len(X_test), len(models)))
            for i, (_, m, mf) in enumerate(models):
                feat_idx = [feature_names.index(f) for f in mf if f in feature_names]
                if len(feat_idx) > 0:
                    test_probas[:, i] = m.predict_proba(X_test[:, feat_idx])[:, 1]
                else:
                    test_probas[:, i] = 0.5
            ensemble_proba = test_probas.mean(axis=1)

            # Metrics
            test_auc = roc_auc_score(y_test, ensemble_proba)

            # Best threshold by F1
            best_f1 = 0
            best_thr = 0.50
            for thr_dec in range(35, 80):
                thr = thr_dec / 100.0
                pred = (ensemble_proba >= thr).astype(int)
                f1 = f1_score(y_test, pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thr = thr

            # WR at THR=0.50
            pred_50 = (ensemble_proba >= 0.50).astype(int)
            n_pred = pred_50.sum()
            wr_50 = y_test[pred_50].mean() if n_pred > 0 else 0

            # WR at multiple thresholds
            threshold_results = {}
            for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
                pred_t = (ensemble_proba >= thr).astype(int)
                nt = pred_t.sum()
                wt = y_test[pred_t].mean() if nt > 3 else 0
                pt = precision_score(y_test, pred_t, zero_division=0) if nt > 0 else 0
                threshold_results[f'thr_{thr:.2f}'] = {
                    'n_trades': int(nt),
                    'wr': float(wt),
                    'precision': float(pt),
                }

            result[side] = {
                'test_auc': float(test_auc),
                'n_test': len(y_test),
                'test_pos_pct': float(y_test.mean()),
                'n_train': len(y_train),
                'train_pos_pct': float(y_train.mean()),
                'best_thr': float(best_thr),
                'best_f1': float(best_f1),
                'wr_50': float(wr_50),
                'n_pred_50': int(n_pred),
                'thresholds': threshold_results,
            }

        result['time_s'] = round(time.time() - t0, 1)
        result['status'] = 'ok'

    except Exception as e:
        import traceback
        result['status'] = 'fail'
        result['error'] = str(e)
        result['traceback'] = traceback.format_exc()

    return result


def format_result(r: dict) -> str:
    """Format single result as readable string."""
    if r['status'] != 'ok':
        return f"  ❌ {r['symbol']:>15s}: FAIL — {r.get('error', 'unknown')}"

    lines = [f"  {r['symbol']:>15s}: {r['rows']:5d} rows, {r['features']:3d} feats, {r['time_s']:5.1f}s"]
    for side in ['long', 'short']:
        if side not in r:
            continue
        s = r[side]
        # Best threshold trade-off
        best = max(
            [(k, v) for k, v in s['thresholds'].items() if v['n_trades'] >= 10],
            key=lambda x: x[1]['wr'], default=(None, {})
        )
        if best[0]:
            thr_label, thr_data = best
            lines.append(f"    {side:5s}: AUC={s['test_auc']:.3f} | BEST={thr_label} WR={thr_data['wr']:.1%} n={thr_data['n_trades']} | THR50 WR={s['wr_50']:.1%} n={s['n_pred_50']}")
        else:
            lines.append(f"    {side:5s}: AUC={s['test_auc']:.3f} | THR50 WR={s['wr_50']:.1%} n={s['n_pred_50']}")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to train')
    parser.add_argument('--parallel', type=int, default=4, help='Parallel workers (default: 4)')
    args = parser.parse_args()

    symbols = args.symbols or LIVE_SYMBOLS
    print(f"\n{'='*60}")
    print(f"🔥 BATCH 45-DAY TRAINING OPTIMIZATION")
    print(f"{'='*60}")
    print(f"Symbols: {len(symbols)}")
    print(f"Train days: {TRAIN_DAYS}")
    print(f"Split: {TRAIN_SPLIT*100:.0f}/{100-TRAIN_SPLIT*100:.0f} (train/OOS)")
    print(f"Seeds: {len(SEEDS)} per side")
    print(f"Parallel: {args.parallel}")
    print()

    all_results = []
    t_start = time.time()

    # Run in parallel
    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        futures = {executor.submit(train_symbol_45d, sym): sym for sym in symbols}
        for f in as_completed(futures):
            r = f.result()
            all_results.append(r)
            print(format_result(r))
            print()
            # Save incremental results
            with open(MODEL_DIR / '_batch_45d_results.json', 'w') as f_out:
                json.dump(all_results, f_out, indent=2, default=str)

    # ─── Summary ───
    elapsed = time.time() - t_start
    ok = [r for r in all_results if r['status'] == 'ok']
    fail = [r for r in all_results if r['status'] != 'ok']

    print(f"\n{'='*60}")
    print(f"SUMMARY — {len(ok)} OK, {len(fail)} FAIL in {elapsed:.0f}s")
    print(f"{'='*60}")

    # Rank by AUC
    print(f"\n📊 RANKED BY LONG AUC:")
    ranked = sorted(ok, key=lambda r: r.get('long', {}).get('test_auc', 0), reverse=True)
    print(f"   {'Symbol':>15s} | {'AUC':>5s} | {'Best THR':>9s} | {'WR':>5s} | {'n':>5s} | {'Short AUC':>10s}")
    print(f"   {'-'*15} | {'-'*5} | {'-'*9} | {'-'*5} | {'-'*5} | {'-'*10}")
    for r in ranked:
        s = r.get('long', {})
        ss = r.get('short', {})
        best = max(
            [(k, v) for k, v in s.get('thresholds', {}).items() if v.get('n_trades', 0) >= 10],
            key=lambda x: x[1]['wr'], default=(None, {})
        )
        if best[0]:
            thr_lbl = best[0].replace('thr_', '')
            wr_val = f"{best[1]['wr']:.0%}"
            n_val = best[1]['n_trades']
        else:
            thr_lbl = '-'
            wr_val = '-'
            n_val = 0
        print(f"   {r['symbol']:>15s} | {s.get('test_auc', 0):.3f} | {thr_lbl:>9s} | {wr_val:>5s} | {n_val:>5d} | {ss.get('test_auc', 0):.3f}")

    # Symbols meeting WR >= 70%
    wr70_symbols = []
    for r in ok:
        for side in ['long', 'short']:
            s = r.get(side, {})
            # Check if ANY threshold gives WR >= 70% with >= 10 trades
            for k, v in s.get('thresholds', {}).items():
                if v.get('wr', 0) >= 0.70 and v.get('n_trades', 0) >= 10:
                    wr70_symbols.append((r['symbol'], side, k, v['wr'], v['n_trades']))
                    break

    print(f"\n📊 SYMBOLS WITH WR >= 70% (>=10 OOS trades):")
    if wr70_symbols:
        for sym, side, thr, wr_, n_ in sorted(wr70_symbols):
            print(f"   ✅ {sym:>15s} {side:5s} @ {thr.replace('thr_',''):>4s} WR={wr_:.0%} n={n_}")
    else:
        print(f"   ❌ None")

    # Save final results
    with open(MODEL_DIR / '_batch_45d_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n📁 Results saved to {MODEL_DIR / '_batch_45d_results.json'}")


if __name__ == '__main__':
    main()
