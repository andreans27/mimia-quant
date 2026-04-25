"""
Train XGBoost ensemble (5 models per symbol) for majority voting.
Different random seeds + hyperparameter diversity for decorrelated predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import json
from datetime import datetime

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

from src.strategies.ml_features import prepare_ml_dataset

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT",
    "APTUSDT", "FETUSDT", "TIAUSDT", "OPUSDT",
    "1000PEPEUSDT", "SUIUSDT", "ARBUSDT", "INJUSDT"
]

MODEL_DIR = Path("data/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 101, 202, 303, 404]

# Diverse hyperparameter configs per seed (different subsample/colsample/depth combos)
HPARAMS = {
    42:  {'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}


def select_features(X_train, y_train, X_val=None):
    """Feature selection: remove low-variance, non-numeric + top 80 by mutual info."""
    # Filter to numeric columns only (Arrow-backed string columns break var/reduction)
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    if X_val is not None:
        X_val = X_val[numeric_cols]
    X_train = X_train[numeric_cols]

    variances = X_train.var()
    low_var = variances[variances < 1e-8].index.tolist()
    if low_var:
        X_train = X_train.drop(columns=low_var)
        if X_val is not None:
            X_val = X_val.drop(columns=low_var, errors='ignore')
    
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        X_train = X_train.drop(columns=nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=nan_cols, errors='ignore')
    
    mi = mutual_info_classif(X_train.fillna(0).clip(-10, 10), y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    top_features = mi_series.head(100).index.tolist()  # More features for 5-TF ensemble
    
    X_train = X_train[top_features]
    if X_val is not None:
        X_val = X_val[top_features]
    
    return X_train, X_val, top_features


def train_full_from_features(feat_df: pd.DataFrame, symbol: str) -> dict:
    """
    Train 'full' ensemble using pre-computed features (ALL features from all TFs).
    Saves as {symbol}_full_xgb_ens_{seed}.json — compatible with load_models() fallback.

    This is the core training function used by auto_retrain.py for the 'full' TF group.
    """
    print(f"\n{'='*60}")
    print(f"Training FULL ensemble for {symbol} (5 models, all features)...")
    print(f"{'='*60}")

    if 'target' not in feat_df.columns:
        print(f"  ❌ No 'target' column in pre-computed features")
        return None

    # Use all feature columns (exclude target and non-feature columns)
    exclude_cols = {'target', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    feature_names = [c for c in feat_df.columns if c not in exclude_cols]

    X = feat_df[feature_names]
    y = feat_df['target']

    print(f"  Dataset: {len(X)} samples, {len(feature_names)} features")

    # Split 70/15/15
    n = len(X)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Feature selection (MI top 100)
    X_train_fs, X_val_fs, selected_features = select_features(X_train, y_train, X_val)
    X_test_fs = X_test[selected_features]

    for df_ in [X_train_fs, X_val_fs, X_test_fs]:
        df_.fillna(0, inplace=True)
        df_.clip(-10, 10, inplace=True)

    models = []
    all_metrics = {}

    for seed in SEEDS:
        hp = HPARAMS[seed]

        # Feature diversity per seed
        np.random.seed(seed)
        n_features = len(selected_features)
        if n_features > 60:
            feat_mask = np.random.choice([True, False], size=n_features, p=[0.75, 0.25])
            model_features = [f for f, m in zip(selected_features, feat_mask) if m]
        else:
            model_features = selected_features

        # Class balance
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

        params = {
            'n_estimators': 400,
            'max_depth': hp['max_depth'],
            'learning_rate': hp['learning_rate'],
            'subsample': hp['subsample'],
            'colsample_bytree': hp['colsample_bytree'],
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': hp['min_child_weight'],
            'scale_pos_weight': scale_pos,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'early_stopping_rounds': 30,
            'random_state': seed,
            'verbosity': 0,
        }

        X_train_m = X_train_fs[model_features] if len(model_features) < len(selected_features) else X_train_fs
        X_val_m = X_val_fs[model_features] if len(model_features) < len(selected_features) else X_val_fs

        model = xgb.XGBClassifier(**params)
        model.fit(X_train_m, y_train.loc[X_train_m.index],
                  eval_set=[(X_val_m, y_val.loc[X_val_m.index])],
                  verbose=False)

        # Evaluate
        X_test_m = X_test_fs[model_features] if len(model_features) < len(selected_features) else X_test_fs
        y_prob = model.predict_proba(X_test_m)[:, 1]
        y_pred = (y_prob >= 0.55).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1_v = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"    Model {seed}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1_v:.3f} AUC={auc:.3f} (features={len(model_features)})")
        all_metrics[seed] = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1_v), 'auc': float(auc)}
        models.append((str(seed), model, model_features))

    # Ensemble evaluation
    all_probs = []
    for seed, model, mf in models:
        X_test_m = X_test_fs[mf]
        probs = model.predict_proba(X_test_m)[:, 1]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    for thresh in [0.50, 0.55, 0.60, 0.65]:
        ensemble_pred = (avg_probs >= thresh).astype(int)
        ens_acc = accuracy_score(y_test, ensemble_pred)
        ens_prec = precision_score(y_test, ensemble_pred, zero_division=0)
        ens_rec = recall_score(y_test, ensemble_pred, zero_division=0)
        ens_auc = roc_auc_score(y_test, avg_probs)
        print(f"    ENSEMBLE @{thresh:.2f}: Acc={ens_acc:.3f} Prec={ens_prec:.3f} Rec={ens_rec:.3f} AUC={ens_auc:.3f}")

    # Best threshold by F1
    best_f1 = 0
    best_thresh = 0.55
    for thresh in np.arange(0.40, 0.80, 0.01):
        ensemble_pred = (avg_probs >= thresh).astype(int)
        f1_val = f1_score(y_test, ensemble_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh

    print(f"    Best threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # Save models with 'full' prefix (compatible with load_models)
    model_paths = {}
    for seed, model, mf in models:
        path = MODEL_DIR / f"{symbol}_full_xgb_ens_{seed}.json"
        model.save_model(str(path))
        model_paths[str(seed)] = str(path)

    # Save meta with same name as original (load_models looks for this)
    meta = {
        'symbol': symbol,
        'ensemble_size': len(SEEDS),
        'seeds': SEEDS,
        'model_paths': model_paths,
        'features': selected_features,
        'model_features': {str(s): mf for s, mf in zip(SEEDS, [m[2] for m in models])},
        'individual_metrics': all_metrics,
        'ensemble_metrics': {
            'best_threshold': float(best_thresh),
            'test_auc': float(roc_auc_score(y_test, avg_probs)),
            'test_precision_at_55': float(precision_score(y_test, (avg_probs >= 0.55).astype(int), zero_division=0)),
            'test_accuracy_at_55': float(accuracy_score(y_test, (avg_probs >= 0.55).astype(int))),
        },
        'training_date': datetime.now().isoformat(),
    }

    meta_path = MODEL_DIR / f"{symbol}_ensemble_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\n  ✅ FULL ensemble saved: {len(SEEDS)} models + metadata")
    print(f"    Best thr={best_thresh:.2f}, AUC={meta['ensemble_metrics']['test_auc']:.3f}")
    return meta


# ─── Refactor original train_ensemble_symbol to reuse new function ───

def train_ensemble_symbol(symbol: str, days: int = 120):
    """Original API — fetches data via prepare_ml_dataset, then delegates to train_full_from_features."""
    X, y, idx = prepare_ml_dataset(
        symbol, days=days,
        target_candle=9,
        intervals=['15m', '30m', '1h', '4h']
    )
    if X is None:
        print(f"  ❌ No data for {symbol}")
        return None

    # Combine X & y back into a DataFrame for train_full_from_features
    feat_df = X.copy()
    feat_df['target'] = y
    return train_full_from_features(feat_df, symbol)


def main():
    # First, train on high-cap pairs only (faster turnaround)
    results = {}
    
    for symbol in SYMBOLS:
        try:
            meta = train_ensemble_symbol(symbol)
            if meta:
                results[symbol] = meta['ensemble_metrics']
        except Exception as e:
            print(f"\n  ❌ {symbol} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'AUC':>6} {'Acc@55':>8} {'Prec@55':>8} {'BestThr':>8}")
    print("-" * 45)
    for symbol, metrics in results.items():
        print(f"{symbol:<10} {metrics['test_auc']:>6.3f} {metrics['test_accuracy_at_55']:>6.3f} "
              f"{metrics['test_precision_at_55']:>6.3f} {metrics['best_threshold']:>7.2f}")
    
    avg_auc = np.mean([m['test_auc'] for m in results.values()])
    avg_acc = np.mean([m['test_accuracy_at_55'] for m in results.values()])
    print(f"\n  Average Test AUC: {avg_auc:.3f}")
    print(f"  Average Test Acc: {avg_acc:.3f}")
    
    summary_path = MODEL_DIR / "ensemble_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv for kk, vv in v.items()} for k, v in results.items()},
                    "avg_auc": float(avg_auc), "avg_acc": float(avg_acc)}, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
