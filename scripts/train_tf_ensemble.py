"""
Train XGBoost ensemble (5 models) at a specific native timeframe, no resampling.
Usage: python scripts/train_tf_ensemble.py <symbol> <interval> <target_candle> <days>

Examples:
  python scripts/train_tf_ensemble.py BTCUSDT 15m 3 120
  python scripts/train_tf_ensemble.py ETHUSDT 30m 2 120
  python scripts/train_tf_ensemble.py SOLUSDT 1h 1 120
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

from src.strategies.ml_features import compute_technical_features
from src.utils.binance_client import BinanceRESTClient

MODEL_DIR = Path("data/ml_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 101, 202, 303, 404]

HPARAMS = {
    42:  {'max_depth': 6,  'subsample': 0.80, 'colsample_bytree': 0.8, 'learning_rate': 0.05, 'min_child_weight': 5},
    101: {'max_depth': 5,  'subsample': 0.85, 'colsample_bytree': 0.7, 'learning_rate': 0.06, 'min_child_weight': 3},
    202: {'max_depth': 7,  'subsample': 0.75, 'colsample_bytree': 0.9, 'learning_rate': 0.04, 'min_child_weight': 7},
    303: {'max_depth': 4,  'subsample': 0.90, 'colsample_bytree': 0.6, 'learning_rate': 0.07, 'min_child_weight': 4},
    404: {'max_depth': 8,  'subsample': 0.70, 'colsample_bytree': 1.0, 'learning_rate': 0.03, 'min_child_weight': 6},
}


def _fetch_all_klines(client, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all klines with pagination."""
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
               'taker_buy_ignore', 'ignore']
    all_bars = []
    current_start = start_ms
    max_per_request = 1000

    while current_start < end_ms:
        try:
            batch = client.get_klines(
                symbol=symbol, interval=interval,
                limit=max_per_request,
                start_time=current_start,
                end_time=end_ms
            )
            if not batch or len(batch) == 0:
                break
            all_bars.extend(batch)
            last = batch[-1]
            if isinstance(last, (list, tuple)):
                current_start = int(last[0]) + 1
            else:
                current_start += 3600000
        except Exception as e:
            print(f"    ⚠️  API error: {e}")
            break

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars, columns=columns)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def fetch_native_data(symbol: str, interval: str, days: int = 120) -> pd.DataFrame:
    """Fetch OHLCV data at native interval from Binance."""
    print(f"  Fetching {days} days of {interval} data for {symbol}...")
    client = BinanceRESTClient(testnet=True)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)

    df = _fetch_all_klines(client, symbol, interval, start_ms, end_ms)
    if df is None or len(df) < 200:
        print(f"  ⚠️ Insufficient {interval} data for {symbol}")
        return None
    print(f"    Loaded {len(df)} bars")
    return df


def prepare_native_dataset(symbol: str, interval: str, days: int = 120,
                           target_candle: int = 3) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """Fetch native TF data, compute features, create target.

    Args:
        symbol: Trading pair
        interval: Native interval (15m, 30m, 1h)
        days: Historical data to fetch
        target_candle: How many candles ahead to predict (3 for 15m = 45min)

    Returns:
        (X, y, index)
    """
    df = fetch_native_data(symbol, interval, days)
    if df is None:
        return None, None, None

    # Compute features on native OHLCV
    print(f"  Computing technical features on {interval} data...")
    featured = compute_technical_features(df, prefix=f"{interval}_", drop_raw=False)
    featured = compute_derived_features(featured, prefix=interval)

    # Drop NaN rows from indicator lookback
    featured = featured.dropna()
    if len(featured) < 200:
        print(f"  ⚠️ Too few rows after NaN drop: {len(featured)}")
        return None, None, None

    # Create target: price direction % target_candle bars ahead
    future_close = featured['close'].shift(-target_candle)
    target = ((future_close - featured['close']) / featured['close'] > 0.001).astype(int)

    # Remove rows where target is unknown (end of series)
    valid = target.notna()
    featured = featured[valid]
    target = target[valid]

    # Exclude raw OHLCV columns, keep only derived features
    ohlcv_cols = {'open', 'high', 'low', 'close', 'volume'}
    feature_cols = [c for c in featured.columns if c not in ohlcv_cols]

    if len(feature_cols) < 20:
        print(f"  ⚠️ Too few feature columns: {len(feature_cols)}")
        return None, None, None

    print(f"  Dataset: {len(featured)} rows, {len(feature_cols)} features")
    return featured[feature_cols], target, featured.index


def compute_derived_features(df: pd.DataFrame, ohlcv: pd.DataFrame = None, prefix: str = "") -> pd.DataFrame:
    """Additional derived features beyond technical indicators."""
    result = df.copy()
    if ohlcv is not None:
        close = ohlcv['close'].astype(float)
        high = ohlcv['high'].astype(float)
        low = ohlcv['low'].astype(float)
        volume = ohlcv['volume'].astype(float)
    else:
        close = result['close'].astype(float)
        high = result['high'].astype(float)
        low = result['low'].astype(float)
        volume = result['volume'].astype(float)

    p = prefix if prefix else ''

    # Price location within recent range
    for period in [5, 10, 20, 40]:
        hh = high.rolling(period).max()
        ll = low.rolling(period).min()
        result[f"{p}pct_loc_{period}"] = (close - ll) / (hh - ll).replace(0, np.nan)
        result[f"{p}range_{period}"] = (hh - ll) / close.replace(0, np.nan)

    # Volume features
    result[f"{p}vol_ma_5"] = volume / volume.rolling(5).mean().replace(0, np.nan)
    result[f"{p}vol_ma_10"] = volume / volume.rolling(10).mean().replace(0, np.nan)
    result[f"{p}vwap"] = (volume * (high + low + close) / 3).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)

    # Moving average crossovers
    for fast, slow in [(5, 20), (10, 30), (20, 50)]:
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()
        result[f"{p}ma_{fast}_{slow}_ratio"] = ma_fast / ma_slow.replace(0, np.nan)

    # Rate of change
    for period in [3, 5, 10]:
        result[f"{p}roc_{period}"] = close.pct_change(period)

    return result


def select_features(X_train, y_train, X_val=None, top_k: int = 80):
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

    mi = mutual_info_classif(X_train.fillna(0).clip(-10, 10), y_train, random_state=42)
    mi_series = pd.Series(mi, index=X_train.columns).sort_values(ascending=False)
    top_features = mi_series.head(top_k).index.tolist()

    X_train = X_train[top_features]
    if X_val is not None:
        X_val = X_val[top_features]

    return X_train, X_val, top_features


def train_tf_ensemble(symbol: str, interval: str, target_candle: int = 3, days: int = 120,
                      top_features: int = 80):
    """Train 5 XGBoost models at a specific timeframe."""
    print(f"\n{'='*60}")
    print(f"Training {interval} ENSEMBLE for {symbol} (target={target_candle}bars)...")
    print(f"{'='*60}")

    X, y, idx = prepare_native_dataset(symbol, interval, days, target_candle)
    if X is None:
        print(f"  ❌ No data for {symbol}")
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

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Feature selection
    X_train_fs, X_val_fs, selected_features = select_features(X_train, y_train, X_val, top_k=top_features)
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

        if n_features > 60:
            feat_mask = np.random.choice([True, False], size=n_features, p=[0.75, 0.25])
            model_features = [f for f, m in zip(selected_features, feat_mask) if m]
        else:
            model_features = selected_features

        # Class balance
        pos_mask = (y_train == 1).values
        neg_mask = (y_train == 0).values
        n_pos = pos_mask.sum()
        n_neg = neg_mask.sum()
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

        print(f"    Model {seed}: Acc={acc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1_v:.3f} AUC={auc:.3f} (feats={len(model_features)})")
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
        ens_pred = (avg_probs >= thresh).astype(int)
        ens_acc = accuracy_score(y_test, ens_pred)
        ens_prec = precision_score(y_test, ens_pred, zero_division=0)
        ens_rec = recall_score(y_test, ens_pred, zero_division=0)
        ens_auc = roc_auc_score(y_test, avg_probs)
        print(f"    ENSEMBLE @{thresh:.2f}: Acc={ens_acc:.3f} Prec={ens_prec:.3f} Rec={ens_rec:.3f} AUC={ens_auc:.3f}")

    # Best threshold by F1
    best_f1 = 0
    best_thresh = 0.55
    for thresh in np.arange(0.40, 0.80, 0.01):
        ens_pred = (avg_probs >= thresh).astype(int)
        f1_val = f1_score(y_test, ens_pred, zero_division=0)
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh
    print(f"    Best threshold: {best_thresh:.2f} (F1={best_f1:.3f})")

    # Save models with TF-specific naming
    model_paths = {}
    for seed, model, mf in models:
        path = MODEL_DIR / f"{symbol}_{interval}_xgb_ens_{seed}.json"
        model.save_model(str(path))
        model_paths[str(seed)] = str(path)

    meta = {
        'symbol': symbol,
        'interval': interval,
        'target_candle': target_candle,
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

    meta_path = MODEL_DIR / f"{symbol}_{interval}_ensemble_meta.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    print(f"\n  ✅ {interval} ensemble saved: {len(SEEDS)} models + metadata")
    return meta


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train TF-specific ensemble')
    parser.add_argument('--symbols', nargs='+', default=[
        'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
        'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT'
    ])
    parser.add_argument('--interval', default='15m', choices=['15m', '30m', '1h'])
    parser.add_argument('--target_candle', type=int, default=3)
    parser.add_argument('--days', type=int, default=120)
    parser.add_argument('--top_features', type=int, default=80)
    args = parser.parse_args()

    results = {}
    for symbol in args.symbols:
        try:
            meta = train_tf_ensemble(symbol, args.interval, args.target_candle,
                                     args.days, args.top_features)
            if meta:
                results[symbol] = meta['ensemble_metrics']
        except Exception as e:
            print(f"\n  ❌ {symbol} failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"{args.interval} ENSEMBLE TRAINING SUMMARY")
    print(f"{'='*60}")
    if results:
        print(f"{'Symbol':<10} {'AUC':>6} {'Acc@55':>8} {'Prec@55':>8} {'BestThr':>8}")
        print("-" * 45)
        for symbol, m in results.items():
            print(f"{symbol:<10} {m['test_auc']:>6.3f} {m['test_accuracy_at_55']:>6.3f} "
                  f"{m['test_precision_at_55']:>6.3f} {m['best_threshold']:>7.2f}")

        avg_auc = np.mean([m['test_auc'] for m in results.values()])
        avg_acc = np.mean([m['test_accuracy_at_55'] for m in results.values()])
        print(f"\n  Average Test AUC: {avg_auc:.3f}")
        print(f"  Average Test Acc: {avg_acc:.3f}")

        summary_path = MODEL_DIR / f"{args.interval}_ensemble_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                "interval": args.interval,
                "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                               for kk, vv in v.items()} for k, v in results.items()},
                "avg_auc": float(avg_auc), "avg_acc": float(avg_acc)
            }, f, indent=2)
        print(f"\n  Summary saved: {summary_path}")
    else:
        print("  No results to report.")


if __name__ == "__main__":
    main()
