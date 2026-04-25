"""
OOS Validation v2 — Heavy Regularization + TF Group Filtering
================================================================
- Skip m15, m30 models (AUC < 0.65 historically)
- Only vote: full + h1 + h4
- XGBoost params: high regularization, shallow trees, less features
- Feature subsampling per seed (50-75%)
- Support for weighted voting by TF group AUC

Usage: python scripts/oos_validation_v2.py [--threshold 0.60]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
import requests
import random
warnings.filterwarnings('ignore')

import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
           'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT']

# Only use strong TF groups — skip m15 (AUC 0.563) and m30 (AUC 0.632)
TF_GROUPS_V2 = ['full', 'h1', 'h4']
SEEDS = [42, 101, 202, 303, 404]

# Weights for TF groups based on training AUC
TF_WEIGHTS = {'full': 1.0, 'h1': 1.2, 'h4': 1.0}

COMMISSION = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0


def fetch_5m_ohlcv_range(symbol, start_time, end_time):
    """Fetch 5m OHLCV from Binance public API."""
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    limit = 1000
    all_bars = []
    last_ts = start_ms
    while last_ts < end_ms:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol, 'interval': '5m', 'limit': limit,
            'startTime': last_ts, 'endTime': end_ms
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        all_bars.extend(batch)
        last_ts = batch[-1][0] + 1
        if len(batch) < limit:
            break
    if len(all_bars) < 100:
        return None
    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    return df[['open', 'high', 'low', 'close', 'volume']]


def train_oos_models(feat_df, split_date):
    """Train models on train split with heavy regularization."""
    train_df = feat_df[feat_df.index < split_date].copy()
    test_df = feat_df[feat_df.index >= split_date].copy()

    feature_sets = {}
    feature_sets['full'] = [c for c in feat_df.columns if c != 'target']
    for tf in ['m15', 'm30', 'h1', 'h4']:
        prefix = f"{tf}_"
        tf_feat = [c for c in feat_df.columns if c.startswith(prefix) and c != 'target']
        feature_sets[tf] = tf_feat

    print(f"    Train: {len(train_df)} | Test: {len(test_df)}")

    models_group = {}
    for tf in TF_GROUPS_V2:
        features = feature_sets[tf]
        if len(features) < 10:
            continue

        # Use full feature set for X matrix but subsample per seed
        y_train = train_df['target']
        valid = np.isfinite(train_df[features].fillna(0)).all(axis=1) & np.isfinite(y_train)
        y_train = y_train[valid]
        train_clean = train_df.loc[valid]
        full_X = train_clean[features].fillna(0).clip(-10, 10)

        if len(train_clean) < 1000:
            print(f"      ⚠️ {tf}: only {len(train_clean)} rows")
            continue

        tf_models = []
        for seed in SEEDS:
            # Feature subsampling (50-75%) for each seed
            random.seed(seed * 7)
            n_feat = max(15, int(len(features) * random.uniform(0.5, 0.75)))
            sampled = random.sample(list(features), n_feat)

            X_train = train_clean[sampled].fillna(0).clip(-10, 10)

            model = xgb.XGBClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.04,
                subsample=0.6, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=3.0,
                min_child_weight=7, gamma=1.0,
                random_state=seed, verbosity=0,
                use_label_encoder=False, eval_metric='logloss',
                early_stopping_rounds=20
            )
            eval_size = min(3000, len(X_train) // 5)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train[:eval_size], y_train[:eval_size])],
                verbose=False
            )
            tf_models.append((str(seed), model, sampled))

        models_group[tf] = tf_models
        print(f"      {tf}: {len(tf_models)} seeds, {n_feat} feats each")

    return models_group, test_df


def run_oos(symbol, threshold=0.60, hold=9, cool=3,
            warmup=200, initial_capital=5000.0):
    """Run OOS with regularized models."""
    print(f"\n{'='*70}")
    print(f"  OOSv2 | {symbol} | thresh={threshold} hold={hold}")
    print(f"{'='*70}")

    end = datetime.now()
    start = end - timedelta(days=135)
    df_ohlcv = fetch_5m_ohlcv_range(symbol, start, end)
    if df_ohlcv is None or len(df_ohlcv) < 2000:
        print(f"  ❌ OHLCV: {len(df_ohlcv) if df_ohlcv is not None else 0}")
        return None
    print(f"  OHLCV: {len(df_ohlcv)} bars")

    cache = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    if not cache:
        return None
    feat_df = pd.read_parquet(max(cache, key=lambda p: p.stat().st_mtime))
    print(f"  Cache: {len(feat_df)} rows")

    split_date = feat_df.index[int(len(feat_df) * 0.80)]

    print(f"  Training {TF_GROUPS_V2} on train split ({split_date})...")
    models_group, test_feat = train_oos_models(feat_df, split_date)

    if len(models_group) < 2:
        print("  ❌ Not enough trained groups")
        return None

    # --- Predict OOS ---
    group_probs = {}
    for tf in TF_GROUPS_V2:
        if tf not in models_group:
            continue
        all_probs = []
        for seed, model, sampled in models_group[tf]:
            available = [c for c in sampled if c in test_feat.columns]
            if len(available) < 5:
                continue
            X = test_feat[available].fillna(0).clip(-10, 10)
            probs = model.predict_proba(X)[:, 1]
            all_probs.append(probs)
        if all_probs:
            avg = np.mean(all_probs, axis=0)
            group_probs[tf] = avg

    if len(group_probs) < 2:
        return None

    print(f"  Groups: {list(group_probs.keys())}")

    # Weighted voting
    prob_stack = np.column_stack([
        group_probs[tf] * TF_WEIGHTS.get(tf, 1.0)
        for tf in group_probs
    ])
    weights = np.array([TF_WEIGHTS.get(tf, 1.0) for tf in group_probs])
    avg_probs = np.sum(prob_stack, axis=1) / np.sum(weights)

    # Align with OHLCV
    ohlcv_start = split_date - timedelta(hours=12)
    df_bt = df_ohlcv[df_ohlcv.index >= ohlcv_start].join(
        pd.DataFrame({'proba': avg_probs}, index=test_feat.index), how='inner')
    df_bt = df_bt.iloc[warmup:]
    print(f"  Test: {len(df_bt)} bars | {df_bt.index[0].strftime('%Y-%m-%d')} → {df_bt.index[-1].strftime('%Y-%m-%d')}")

    # --- Simulate ---
    capital = initial_capital
    trades = []
    equity_curve = [capital]
    timestamps = [df_bt.index[0]]
    cooldown = 0
    hold_remaining = 0
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    long_pnl = 0.0
    short_pnl = 0.0

    for idx in range(len(df_bt)):
        row = df_bt.iloc[idx]
        price = float(row['close'])
        ts = df_bt.index[idx]

        if cooldown > 0:
            cooldown -= 1

        # Exit when hold expires
        if position != 0:
            hold_remaining -= 1
            if hold_remaining <= 0:
                exit_price = price * (1 - SLIPPAGE) if position == 1 else price * (1 + SLIPPAGE)
                if position == 1:
                    raw_pnl = entry_qty * (exit_price - entry_price)
                else:
                    raw_pnl = entry_qty * (entry_price - exit_price)

                comm = (entry_qty * entry_price + entry_qty * exit_price) * COMMISSION
                pnl_net = raw_pnl - comm
                capital += raw_pnl

                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net

                hold_dur = hold_counter - hold_remaining
                trades.append({
                    'direction': 'long' if position == 1 else 'short',
                    'pnl_net': pnl_net,
                    'hold_bars': hold_dur,
                })
                position = 0
                cooldown = cool

        # Entry
        if position == 0 and cooldown <= 0:
            prob_val = float(df_bt.iloc[idx]['proba'])
            if price <= 0 or np.isnan(prob_val):
                continue
            direction = 0
            if prob_val >= threshold:
                direction = 1
                entry_price = price * (1 + SLIPPAGE)
            elif prob_val <= (1 - threshold):
                direction = -1
                entry_price = price * (1 - SLIPPAGE)
            if direction != 0:
                entry_qty = (capital * 0.15) / entry_price
                position = direction
                hold_remaining = hold
                hold_counter = hold
                entry_time = ts

        equity_curve.append(capital)
        timestamps.append(ts)

    # --- Metrics ---
    if len(trades) < 10:
        print(f"  ❌ Only {len(trades)} trades")
        return None

    equity = np.array(equity_curve)
    total_return_pct = (capital - initial_capital) / initial_capital * 100
    peak = np.maximum.accumulate(equity)
    max_dd_pct = float(np.max((peak - equity) / peak)) * 100
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    win_rate = wins / len(trades) * 100
    gross_profit = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gross_loss = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    months = max(total_days / 30.44, 0.5)
    avg_monthly_return = total_return_pct / months

    rets = np.diff(equity) / equity[:-1]
    if len(rets) > 0 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)
        downside = rets[rets < 0]
        sortino = np.mean(rets) / np.std(downside) * np.sqrt(365 * 24 * 12) if len(downside) > 0 and np.std(downside) > 0 else 0
    else:
        sharpe = sortino = 0

    # Monthly consistency
    df_eq = pd.DataFrame({'equity': equity}, index=timestamps)
    df_eq['month'] = df_eq.index.to_period('M')
    monthly_rets = df_eq.groupby('month')['equity'].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
    profitable = sum(1 for r in monthly_rets if r > 0)
    monthly_consistency = profitable / len(monthly_rets) * 100 if len(monthly_rets) > 0 else 0

    result = {
        'symbol': symbol,
        'total_trades': len(trades),
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'avg_monthly_return_pct': avg_monthly_return,
        'max_drawdown_pct': max_dd_pct,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'monthly_consistency_pct': monthly_consistency,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'num_tf_groups': len(group_probs),
        'test_bars': len(df_bt),
        'date_range': f"{df_bt.index[0].strftime('%Y-%m-%d')} → {df_bt.index[-1].strftime('%Y-%m-%d')}",
        'threshold': threshold,
    }
    return result


def print_result(r):
    fade = " 🔥" if r['total_return_pct'] > 0 and r['win_rate_pct'] > 50 else ""
    print(f"\n  ▸ {r['symbol']} (thresh={r['threshold']:.2f}){fade}")
    print(f"    {r['date_range']} ({r['test_bars']} bars)")
    print(f"    Trades: {r['total_trades']} | WR: {r['win_rate_pct']:.1f}% | PF: {r['profit_factor']:.2f}")
    print(f"    Return: {r['total_return_pct']:.2f}% | Monthly: {r['avg_monthly_return_pct']:.2f}% | DD: {r['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe: {r['sharpe_ratio']:.2f} | Sortino: {r['sortino_ratio']:.2f} | Consistent: {r['monthly_consistency_pct']:.0f}%")
    print(f"    Long: ${r['long_pnl']:.2f} | Short: ${r['short_pnl']:.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', default=','.join(SYMBOLS))
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--hold', type=int, default=9)
    parser.add_argument('--cool', type=int, default=3)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    print(f"\n{'='*70}")
    print(f"  OOS VALIDATION v2 — Light Models | {TF_GROUPS_V2}")
    print(f"  XGBoost: depth=3, lr=0.04, reg_alpha=1.0, reg_lambda=3.0")
    print(f"  Feature subsampling: 50-75% per seed")
    print(f"  Weighted voting: {TF_WEIGHTS}")
    print(f"{'='*70}")

    all_results = []
    for sym in symbols:
        r = run_oos(sym, threshold=args.threshold, hold=args.hold, cool=args.cool)
        if r:
            print_result(r)
            all_results.append(r)

    if all_results:
        print(f"\n{'='*70}")
        print(f"  OOSv2 SUMMARY")
        print(f"{'='*70}")
        for r in all_results:
            monthly_str = f"M:{r['avg_monthly_return_pct']:.1f}%"
            status = "✅" if r['win_rate_pct'] > 50 and r['profit_factor'] > 1.2 else "⚠️"
            print(f"  {status} {r['symbol']}: "
                  f"WR {r['win_rate_pct']:.1f}% | "
                  f"PF {r['profit_factor']:.2f} | "
                  f"DD {r['max_drawdown_pct']:.2f}% | "
                  f"{monthly_str} | "
                  f"{r['total_trades']} trades")
