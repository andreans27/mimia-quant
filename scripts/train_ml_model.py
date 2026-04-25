"""
Train XGBoost model on multi-timeframe features for crypto trading.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import calibration_curve

from src.strategies.ml_features import prepare_ml_dataset

# 8 pairs
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT"
]

MODEL_DIR = Path("data/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Feature selection: remove near-zero-variance and highly correlated ───
def select_features(X_train, y_train, X_val=None):
    """Remove low-quality features for robustness."""
    from sklearn.feature_selection import mutual_info_classif

    # 1. Remove constant/near-constant features
    variances = X_train.var()
    low_var_cols = variances[variances < 1e-8].index.tolist()
    if low_var_cols:
        print(f"    Removing {len(low_var_cols)} near-zero-variance features")
        X_train = X_train.drop(columns=low_var_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=low_var_cols, errors='ignore')

    # 2. Remove infinite/NaN columns
    nan_cols = X_train.columns[X_train.isna().any()].tolist()
    if nan_cols:
        print(f"    Removing {len(nan_cols)} NaN-containing columns")
        X_train = X_train.drop(columns=nan_cols)
        if X_val is not None:
            X_val = X_val.drop(columns=nan_cols, errors='ignore')

    # 3. Mutual information feature selection (keep top 80)
    mi = mutual_info_classif(X_train.fillna(0).clip(-10, 10), y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    top_features = mi_series.head(80).index.tolist()
    print(f"    Keeping top {len(top_features)} features by mutual information")
    
    X_train = X_train[top_features]
    if X_val is not None:
        X_val = X_val[top_features]

    return X_train, X_val, top_features


# ─── Train per symbol ───
def train_symbol(symbol: str, days: int = 120, use_hyperopt: bool = False):
    """Train an XGBoost model for a single symbol."""
    print(f"\n{'='*60}")
    print(f"Training model for {symbol}...")
    print(f"{'='*60}")
    
    # Prepare dataset
    X, y, idx = prepare_ml_dataset(symbol, days=days, target_candle=3)
    if X is None:
        print(f"  ❌ No data for {symbol}")
        return None
    
    print(f"  Dataset: {len(X)} samples, {len(X.columns)} features")
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X) * 0.15))
    
    # Get the split indices
    for train_idx, val_idx in tscv.split(X):
        pass  # Use the last split
    
    # Use last 15% as test, rest as train
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_val = X.iloc[train_size:train_size + val_size]
    y_val = y.iloc[train_size:train_size + val_size]
    X_test = X.iloc[train_size + val_size:]
    y_test = y.iloc[train_size + val_size:]
    
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class balance: {y_train.mean():.3f}")
    
    # Feature selection
    X_train, X_val, selected_features = select_features(X_train, y_train, X_val)
    X_test = X_test[selected_features]
    
    # Fill remaining NaNs
    for df_ in [X_train, X_val, X_test]:
        df_.fillna(0, inplace=True)
        df_.clip(-10, 10, inplace=True)
    
    # ─── Train model ───
    params = {
        'n_estimators': 300,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_weight': 5,
        'scale_pos_weight': (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'early_stopping_rounds': 30,
        'random_state': 42,
        'verbosity': 0,
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # ─── Evaluate ───
    for name, X_eval, y_eval in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        y_prob = model.predict_proba(X_eval)[:, 1]
        y_pred = (y_prob >= 0.55).astype(int)
        
        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        auc = roc_auc_score(y_eval, y_prob)
        
        print(f"  [{name:>5}] Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f} AUC={auc:.3f}")
    
    # ─── Feature importance ───
    importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 10 features:")
    for _, row in importance.head(10).iterrows():
        print(f"    {row['feature']:<25s} {row['importance']:.4f}")
    
    # ─── Save model ───
    model_path = MODEL_DIR / f"{symbol}_xgb.json"
    model.save_model(str(model_path))
    
    # Save metadata
    meta = {
        'symbol': symbol,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'features': selected_features,
        'feature_importance': importance.to_dict('records'),
        'metrics': {
            'test_accuracy': float(accuracy_score(y_test, (model.predict_proba(X_test)[:,1] >= 0.55).astype(int))),
            'test_precision': float(precision_score(y_test, (model.predict_proba(X_test)[:,1] >= 0.55).astype(int), zero_division=0)),
            'test_recall': float(recall_score(y_test, (model.predict_proba(X_test)[:,1] >= 0.55).astype(int), zero_division=0)),
            'test_f1': float(f1_score(y_test, (model.predict_proba(X_test)[:,1] >= 0.55).astype(int), zero_division=0)),
            'test_auc': float(roc_auc_score(y_test, model.predict_proba(X_test)[:,1])),
        },
        'training_date': datetime.now().isoformat(),
        'params': {k: str(v) if isinstance(v, float) else v for k, v in params.items()}
    }
    
    meta_path = MODEL_DIR / f"{symbol}_xgb_meta.json"
    with open(meta_path, 'w') as f:
        # Convert numpy types to native python
        json.dump(meta, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))
    
    print(f"\n  ✅ Model saved: {model_path}")
    print(f"  ✅ Metadata saved: {meta_path}")
    
    return model, meta


def main():
    results = {}
    
    for symbol in SYMBOLS:
        try:
            result = train_symbol(symbol, days=120)
            if result:
                model, meta = result
                results[symbol] = meta['metrics']
        except Exception as e:
            print(f"\n  ❌ {symbol} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'AUC':>6} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("-" * 46)
    for symbol, metrics in results.items():
        print(f"{symbol:<10} {metrics['test_auc']:>6.3f} {metrics['test_accuracy']:>6.3f} "
              f"{metrics['test_precision']:>6.3f} {metrics['test_recall']:>6.3f} {metrics['test_f1']:>6.3f}")
    
    avg_auc = np.mean([m['test_auc'] for m in results.values()])
    print(f"\n  Average Test AUC: {avg_auc:.3f}")
    
    # Save global summary
    summary_path = MODEL_DIR / "training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({"results": results, "avg_auc": float(avg_auc)}, f, indent=2, default=lambda x: float(x))
    print(f"\n  Summary saved: {summary_path}")

if __name__ == "__main__":
    main()
