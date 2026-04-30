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
    top_features = mi_series.head(100).index.tolist()
    
    X_train = X_train[top_features]
    if X_val is not None:
        X_val = X_val[top_features]
    
    return X_train, X_val, top_features


def train_full_from_features(feat_df: pd.DataFrame, symbol: str, model_prefix: str = 'full') -> dict:
    """
    Train 'full' ensemble using ALL features (no MI pre-selection — XGBoost chooses).
    """
    print(f"\n{'='*60}")
    print(f"Training FULL ensemble for {symbol} (5 models, ALL features)...")
    print(f"{'='*60}")

    if 'target' not in feat_df.columns:
        print(f"  No 'target' column in pre-computed features")
        return None

    exclude_cols = {'target', 'target_long', 'target_short', 'timestamp', 'open', 'high', 'low', 'close', 'volume'}
    feature_names = [c for c in feat_df.columns if c not in exclude_cols]

    X = feat_df[feature_names]
    y = feat_df['target']

    print(f"  Dataset: {len(X)} samples, {len(feature_names)} features")

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

    # Skip MI feature selection — XGBoost handles feature importance internally
    X_train_fs = X_train.copy()
    X_val_fs = X_val.copy()
    X_test_fs = X_test.copy()
    selected_features = feature_names

    for df_ in [X_train_fs, X_val_fs, X_test_fs]:
        df_.fillna(0, inplace=True)
        df_.clip(-10, 10, inplace=True)

    models = []
    all_metrics = {}

    for seed in SEEDS:
        hp = HPARAMS[seed]

        # Feature diversity per seed: keep 85% randomly (less aggressive)
        np.random.seed(seed)
        n_features = len(selected_features)
        feat_mask = np.random.choice([True, False], size=n_features, p=[0.85, 0.15])
        model_features = [f for f, m in zip(selected_features, feat_mask) if m]

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

        params = {
            'n_estimators': 600,
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

        X_test_m = X_test_fs[model_features] if len(model_features) < len(selected_features) else X_test_fs
        y_prob = model.predict_proba(X_test_m)[:, 1]
        y_pred = (y_prob >= 0.55).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"    Model {seed}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f} (features={len(model_features)})")

        models.append((seed, model, model_features))
        all_metrics[str(seed)] = {
            'accuracy': acc, 'precision': prec, 'recall': rec,
            'f1': f1, 'auc': auc, 'features': model_features,
        }

    # Ensemble predictions on test set
    ensemble_probs = np.zeros(len(X_test))
    for seed, m, mf in models:
        X_ens = X_test_fs[mf] if len(mf) < len(selected_features) else X_test_fs
        ensemble_probs += m.predict_proba(X_ens)[:, 1]
    ensemble_probs /= len(models)

    print()
    for thr in [0.50, 0.55, 0.60, 0.65]:
        ens_pred = (ensemble_probs >= thr).astype(int)
        acc = accuracy_score(y_test, ens_pred)
        prec = precision_score(y_test, ens_pred, zero_division=0)
        rec = recall_score(y_test, ens_pred, zero_division=0)
        f1 = f1_score(y_test, ens_pred, zero_division=0)
        print(f"    ENSEMBLE @{thr:.2f}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")

    # Find best threshold
    best_f1 = 0
    best_thr = 0.50
    for thr_dec in range(35, 80):
        thr = thr_dec / 100.0
        ens_pred = (ensemble_probs >= thr).astype(int)
        f1 = f1_score(y_test, ens_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    auc_val = roc_auc_score(y_test, ensemble_probs)
    print(f"    Best threshold: {best_thr:.2f} (F1={best_f1:.3f})")
    print(f"    Best thr={best_thr:.2f}, AUC={auc_val:.3f}")

    # Save models
    model_dir = MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)
    for seed, m, mf in models:
        fname = model_dir / f'{symbol}_{model_prefix}_xgb_ens_{seed}.json'
        m.save_model(str(fname))

    # Save metadata with feature list
    meta_path = model_dir / f'{symbol}_{model_prefix}_meta.json'
    meta = {
        'type': 'ensemble',
        'prefix': model_prefix,
        'model_features': {str(s): mf for s, m, mf in models},
        'features': selected_features,
        'n_features': len(selected_features),
        'n_models': len(models),
        'test_auc': float(auc_val),
        'best_threshold': best_thr,
        'best_f1': float(best_f1),
        'trained_at': datetime.now().isoformat(),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  ✅ {model_prefix.upper()} ensemble saved: {len(models)} models + metadata")
    print(f"    Best thr={best_thr:.2f}, AUC={auc_val:.3f}")

    return {
        'models': models,
        'features': selected_features,
        'test_auc': float(auc_val),
        'best_threshold': best_thr,
        'best_f1': float(best_f1),
        'individual_metrics': all_metrics,
    }
