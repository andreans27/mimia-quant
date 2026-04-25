"""
Train TF-specific XGBoost models using cached 5m multi-TF features.
Each model only uses features from ONE timeframe prefix (m5_, m15_, m30_, h1_, h4_).
All models predict at 5m resolution → aligned for voting.

Usage: python scripts/train_tf_specific.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

MODEL_DIR = Path("data/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT",
    "APTUSDT", "FETUSDT", "TIAUSDT", "OPUSDT",
    "1000PEPEUSDT", "SUIUSDT", "ARBUSDT", "INJUSDT"
]

SEEDS = [42, 101, 202, 303, 404]

HPARAMS = {
    42:  {'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}

# Map: TF prefix -> description
TF_GROUPS = {
    'm5':  {'desc': '5m (native)',          'target': 9},
    'm15': {'desc': '15m (resampled)',       'target': 9},
    'm30': {'desc': '30m (resampled)',       'target': 9},
    'h1':  {'desc': '1h (resampled)',        'target': 9},
    'h4':  {'desc': '4h (resampled)',        'target': 9},
}

CACHE_DIR = Path("data/ml_cache")


def load_cached_features(symbol: str) -> pd.DataFrame:
    """Load the cached multi-TF features."""
    cache_path = list(CACHE_DIR.glob(f"{symbol}_5m_120d_9c_*.parquet"))
    if not cache_path:
        print(f"  ❌ No cache for {symbol}")
        return None
    path = cache_path[0]
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} total features")
    return df


def get_tf_features(df: pd.DataFrame, tf_prefix: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract only features for a specific TF prefix + target column."""
    if 'target' not in df.columns:
        print(f"  ❌ No target column in dataframe")
        return None, None
    
    # Find features for this TF prefix
    tf_cols = [c for c in df.columns if c.startswith(f"{tf_prefix}_")]
    
    if not tf_cols:
        print(f"  ❌ No features found for prefix '{tf_prefix}_'")
        return None, None
    
    print(f"    {tf_prefix}: {len(tf_cols)} features")
    return df[tf_cols], df['target']


def select_features(X_train, y_train, X_val=None, top_k: int = 50):
    """Feature selection: remove low-var/nan, top K by mutual info."""
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

    if len(X_train.columns) < 3:
        return X_train, X_val, X_train.columns.tolist()

    mi = mutual_info_classif(X_train.fillna(0).clip(-10, 10), y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    top_k = min(top_k, len(mi_series))
    top_features = mi_series.head(top_k).index.tolist()

    X_train = X_train[top_features]
    if X_val is not None:
        X_val = X_val[top_features]
    return X_train, X_val, top_features


def train_tf_specific(symbol: str, tf_prefix: str, top_k: int = 50):
    """Train TF-specific ensemble for one symbol/TF combo."""
    desc = TF_GROUPS[tf_prefix]['desc']
    print(f"\n{'='*60}")
    print(f"Training {symbol} — {desc} features")
    print(f"{'='*60}")

    df = load_cached_features(symbol)
    if df is None:
        return None

    X, y = get_tf_features(df, tf_prefix)
    if X is None:
        return None

    # Split
    n = len(X)
    train_size = int(n * 0.7)
    val_size = int(n * 0.15)

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]

    # Feature selection
    X_train_fs, X_val_fs, selected_features = select_features(X_train, y_train, X_val, top_k=top_k)
    X_test_fs = X_test[selected_features]

    for df_ in [X_train_fs, X_val_fs, X_test_fs]:
        df_.fillna(0, inplace=True)
        df_.clip(-10, 10, inplace=True)

    models = []
    all_metrics = {}

    for seed in SEEDS:
        hp = HPARAMS[seed]
        np.random.seed(seed)
        n_features = len(selected_features)

        if n_features > 40:
            feat_mask = np.random.choice([True, False], size=n_features, p=[0.75, 0.25])
            model_features = [f for f, m in zip(selected_features, feat_mask) if m]
        else:
            model_features = selected_features

        # Class balance
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        scale_pos = n_neg / max(n_pos, 1) if n_pos < n_neg else 1.0

        params = {
            'n_estimators': 300,
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

        print(f"    Seed {seed:>3}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1_v:.3f} AUC={auc:.3f} (feats={len(model_features)})")
        all_metrics[seed] = {'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec), 'f1': float(f1_v), 'auc': float(auc)}
        models.append((str(seed), model, model_features))

    # Ensemble evaluation
    all_probs = []
    for seed, model, mf in models:
        X_test_m = X_test_fs[mf]
        probs = model.predict_proba(X_test_m)[:, 1]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)

    for thresh in [0.50, 0.55, 0.60]:
        ens_pred = (avg_probs >= thresh).astype(int)
        ens_acc = accuracy_score(y_test, ens_pred)
        ens_prec = precision_score(y_test, ens_pred, zero_division=0)
        ens_auc = roc_auc_score(y_test, avg_probs)
        print(f"    ENSEMBLE @{thresh:.2f}: Acc={ens_acc:.3f} Prec={ens_prec:.3f} AUC={ens_auc:.3f}")

    # Best threshold by F1
    best_f1 = 0
    best_thresh = 0.50
    for thresh in np.arange(0.35, 0.75, 0.01):
        ens_pred = (avg_probs >= thresh).astype(int)
        f1_val = f1_score(y_test, ens_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh

    # Save models
    model_paths = {}
    for seed, model, mf in models:
        path = MODEL_DIR / f"{symbol}_{tf_prefix}_xgb_ens_{seed}.json"
        model.save_model(str(path))
        model_paths[str(seed)] = str(path)

    meta = {
        'symbol': symbol,
        'tf_prefix': tf_prefix,
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

    meta_path = MODEL_DIR / f"{symbol}_{tf_prefix}_ensemble_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\n  ✅ {tf_prefix} ensemble saved: {len(SEEDS)} models + metadata")
    print(f"    Best thr={best_thresh:.2f}, AUC={meta['ensemble_metrics']['test_auc']:.3f}")
    return meta


def main():
    # TF groups to train (skip m5 — already have full model)
    tf_groups = ['m15', 'm30', 'h1', 'h4']
    
    all_results = {}
    
    for tf_prefix in tf_groups:
        print(f"\n{'#'*60}")
        print(f"# TRAINING: {TF_GROUPS[tf_prefix]['desc']} models")
        print(f"{'#'*60}")
        
        tf_results = {}
        for symbol in SYMBOLS:
            try:
                meta = train_tf_specific(symbol, tf_prefix)
                if meta:
                    tf_results[symbol] = meta['ensemble_metrics']
            except Exception as e:
                print(f"  ❌ {symbol} failed: {e}")
                import traceback
                traceback.print_exc()
        
        all_results[tf_prefix] = tf_results
        
        # Summary for this TF
        if tf_results:
            print(f"\n  {tf_prefix} SUMMARY:")
            print(f"  {'Symbol':<10} {'AUC':>6} {'Acc@55':>8} {'Prec@55':>8} {'BestThr':>8}")
            print("  " + "-" * 45)
            aucs = []
            for symbol, m in tf_results.items():
                print(f"  {symbol:<10} {m['test_auc']:>6.3f} {m['test_accuracy_at_55']:>6.3f} "
                      f"{m['test_precision_at_55']:>6.3f} {m['best_threshold']:>7.2f}")
                aucs.append(m['test_auc'])
            avg_auc = np.mean(aucs)
            print(f"  Avg AUC: {avg_auc:.3f}")
    
    # Final comparison table
    print(f"\n{'='*80}")
    print("TF-SPECIFIC MODEL COMPARISON (Avg AUC)")
    print(f"{'='*80}")
    print(f"  {'TF':>5} | " + " | ".join(f"{s[:4]:>6}" for s in SYMBOLS) + " | Avg")
    print("  " + "-" * (8 + 9 * len(SYMBOLS)))
    for tf_prefix, results in all_results.items():
        aucs = [results.get(s, {}).get('test_auc', 0) for s in SYMBOLS]
        avg = np.mean(aucs) if aucs else 0
        print(f"  {tf_prefix:>5} | " + " | ".join(f"{a:>6.3f}" for a in aucs) + f" | {avg:.3f}")

    # Save combined summary
    summary = {}
    for tf_prefix, results in all_results.items():
        summary[tf_prefix] = {
            symbol: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in m.items()}
            for symbol, m in results.items()
        }
    summary_path = MODEL_DIR / "tf_specific_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Combined summary saved: {summary_path}")


if __name__ == "__main__":
    main()
