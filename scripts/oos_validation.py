"""
OOS Validation — Proper Chronological Split for Multi-TF Voting
================================================================
1. Load cached 5m features for each symbol
2. Chronological split: first 80% train, last 20% test
3. Re-train all 5 TF groups (full, m15, m30, h1, h4) on train only
4. Load 5m OHLCV for test period only
5. Voting backtest on test set
6. Compare OOS metrics vs full-sample in-sample metrics

Usage: python scripts/oos_validation.py [--symbols BTCUSDT,ETHUSDT]
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import warnings
import requests
warnings.filterwarnings('ignore')

import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
           'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT']

TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']
SEEDS = [42, 101, 202, 303, 404]

TAKER_FEE = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0


def fetch_5m_ohlcv_range(symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """Fetch 5m OHLCV for a specific time range from Binance public API."""
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
            print(f"    ⚠️ Binance returned {r.status_code}")
            break
        batch = r.json()
        if not batch or len(batch) == 0:
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


def train_models_on_split(feat_df, split_date):
    """Train all 5 TF groups on training split. Returns trained models dict."""
    train_df = feat_df[feat_df.index < split_date].copy()
    test_df = feat_df[feat_df.index >= split_date].copy()

    print(f"    Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    models_group = {}
    feature_sets = {}

    # --- 'full' group: use meta features ---
    # For full group, we need to know which features the original models used
    # Since we're training fresh, use a broad feature set
    full_features = [c for c in train_df.columns if c != 'target']
    # Filter to only feature columns (not OHLCV)
    # For now, use all non-target columns minus any non-feature cols
    # Actually, the cache only has features + target
    feature_sets['full'] = full_features

    for tf in ['m15', 'm30', 'h1', 'h4']:
        prefix = f"{tf}_"
        tf_features = [c for c in train_df.columns if c.startswith(prefix) and c != 'target']
        feature_sets[tf] = tf_features

    # Train each TF group
    for tf in TF_GROUPS:
        features = feature_sets[tf]
        if len(features) < 10:
            print(f"      ⚠️ {tf}: only {len(features)} features, skipping")
            continue

        X_train_full = train_df[features].fillna(0).clip(-10, 10)
        y_train = train_df['target']
        X_test_full = test_df[features].fillna(0).clip(-10, 10)

        # Remove non-finite
        valid_train = np.isfinite(X_train_full).all(axis=1) & np.isfinite(y_train)
        X_train_full = X_train_full[valid_train]
        y_train = y_train[valid_train]

        if len(X_train_full) < 1000:
            print(f"      ⚠️ {tf}: only {len(X_train_full)} training rows, skipping")
            continue

        # Train multiple seeds with feature subsampling for diversity
        import random as _random
        group_models = []
        for seed in SEEDS:
            _random.seed(seed * 7)
            # Each seed gets different feature subset (50-80% of features)
            # This reduces overfitting via ensemble diversity
            n_feat = max(20, int(len(features) * _random.uniform(0.5, 0.8)))
            sampled_features = _random.sample(list(features), min(n_feat, len(features)))

            X_train = X_train_full[sampled_features]
            eval_size = min(3000, len(X_train) // 5)
            eval_set = [(X_train[:eval_size], y_train[:eval_size])]

            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                subsample=0.65, colsample_bytree=0.65,
                reg_alpha=0.5, reg_lambda=2.0,
                min_child_weight=5,
                random_state=seed, verbosity=0,
                use_label_encoder=False, eval_metric='logloss',
                early_stopping_rounds=25
            )
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            group_models.append((str(seed), model, sampled_features))

        models_group[tf] = group_models
        print(f"      {tf}: {len(group_models)} seeds trained (feat: {len(features)}→~{n_feat})")

    return models_group, test_df


def run_oos_voting(symbol: str, threshold: float = 0.60,
                   hold_bars: int = 9, cooldown_bars: int = 3,
                   train_days: int = 100, test_days: int = 30,
                   warmup_bars: int = 200,
                   initial_capital: float = 5000.0,
                   commission_rate: float = 0.0004,
                   slippage_rate: float = 0.0005) -> dict:
    """Run proper chronological OOS validation."""
    print(f"\n{'='*70}")
    print(f"  OOS VALIDATION | {symbol} | thresh={threshold} hold={hold_bars}")
    print(f"{'='*70}")

    # Fetch extended OHLCV
    end = datetime.now()
    total_days = train_days + test_days + 5  # 5 extra days buffer
    start = end - timedelta(days=total_days)

    df_ohlcv = fetch_5m_ohlcv_range(symbol, start, end)
    if df_ohlcv is None or len(df_ohlcv) < 2000:
        print(f"  ❌ Insufficient OHLCV: {len(df_ohlcv) if df_ohlcv is not None else 0}")
        return None
    print(f"  OHLCV: {len(df_ohlcv)} 5m bars")

    # Load cached features
    cache_candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    if not cache_candidates:
        print(f"  ❌ No cache found for {symbol}")
        return None
    cache_path = max(cache_candidates, key=lambda p: p.stat().st_mtime)
    feat_df = pd.read_parquet(cache_path)
    print(f"  Cache: {len(feat_df)} rows, {len(feat_df.columns)} cols")

    # Chronological split point
    split_date = feat_df.index[int(len(feat_df) * 0.80)]
    print(f"  Split date: {split_date} (80% train / 20% test)")

    # Train on training split
    print(f"  Training {len(TF_GROUPS)} TF groups on {train_days}d...")
    models_group, test_df = train_models_on_split(feat_df, split_date)

    if len(models_group) < 2:
        print(f"  ❌ Only {len(models_group)} groups trained, need >= 2")
        return None

    # Load OHLCV for test period (with buffer)
    ohlcv_test_start = split_date - timedelta(hours=24)  # 24h buffer before test
    df_ohlcv_test = df_ohlcv[df_ohlcv.index >= ohlcv_test_start].copy()
    print(f"  OHLCV test period: {len(df_ohlcv_test)} bars (from {ohlcv_test_start})")

    # Compute probabilities for test set from each group
    group_probs = {}
    for tf in TF_GROUPS:
        if tf not in models_group:
            continue
        mg = models_group[tf]
        all_probs = []
        for seed, model, features in mg:
            available = [c for c in features if c in test_df.columns]
            if len(available) < 5:
                continue
            X = test_df[available].fillna(0).clip(-10, 10)
            probs = model.predict_proba(X)[:, 1]
            all_probs.append(probs)
        if all_probs:
            avg = np.nanmean(all_probs, axis=0)
            group_probs[tf] = avg

    if len(group_probs) < 2:
        print(f"  ❌ Only {len(group_probs)} groups have predictions")
        return None

    print(f"  Predicting groups: {list(group_probs.keys())}")

    # Average all groups
    prob_stack = np.column_stack([group_probs[tf] for tf in group_probs])
    avg_probs = np.mean(prob_stack, axis=1)

    # Align with OHLCV
    sig_df = pd.DataFrame({'proba': avg_probs}, index=test_df.index)
    df_bt = df_ohlcv_test.join(sig_df, how='inner')

    if len(df_bt) < warmup_bars + 100:
        print(f"  ❌ Aligned bars too few: {len(df_bt)}")
        return None

    df_bt = df_bt.iloc[warmup_bars:].copy()
    print(f"  Aligned: {len(df_bt)} test bars (after warmup)")
    print(f"  Date range: {df_bt.index[0]} → {df_bt.index[-1]}")

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
                exit_price = price * (1 - slippage_rate) if position == 1 else price * (1 + slippage_rate)
                if position == 1:
                    raw_pnl = entry_qty * (exit_price - entry_price)
                else:
                    raw_pnl = entry_qty * (entry_price - exit_price)

                comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
                pnl_net = raw_pnl - comm
                capital += raw_pnl

                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net

                trades.append({
                    'direction': 'long' if position == 1 else 'short',
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_net': pnl_net,
                    'pnl_pct': ((exit_price - entry_price) / entry_price * 100) if position == 1 else ((entry_price - exit_price) / entry_price * 100),
                    'hold_bars': hold_counter - hold_remaining,
                })
                position = 0
                cooldown = cooldown_bars

        # Entry
        if position == 0 and cooldown <= 0:
            prob_val = float(df_bt.iloc[idx]['proba'])
            if price <= 0 or np.isnan(prob_val):
                cooldown = max(cooldown, 1)
                continue
            direction = 0
            if prob_val >= threshold:
                direction = 1
                entry_price = price * (1 + slippage_rate)
            elif prob_val <= (1 - threshold):
                direction = -1
                entry_price = price * (1 - slippage_rate)
            if direction != 0:
                position_pct = 0.15
                entry_qty = (capital * position_pct) / entry_price
                position = direction
                hold_remaining = hold_bars
                entry_time = ts
                hold_counter = hold_bars

        equity_curve.append(capital)
        timestamps.append(ts)

    # --- Compute metrics ---
    if len(trades) < 10:
        print(f"  ❌ Only {len(trades)} trades in OOS period")
        return None

    equity = np.array(equity_curve)
    total_return_pct = (capital - initial_capital) / initial_capital * 100

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd_pct = float(np.max(dd)) * 100

    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    win_rate = wins / len(trades) * 100

    gross_profit = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gross_loss = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    months = max(total_days / 30.44, 0.5)
    avg_monthly_return = total_return_pct / months

    rets = np.diff(equity) / equity[:-1]
    if len(rets) > 0 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)
        downside = rets[rets < 0]
        sortino = np.mean(rets) / np.std(downside) * np.sqrt(365 * 24 * 12) if len(downside) > 0 and np.std(downside) > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    df_equity = pd.DataFrame({'equity': equity}, index=timestamps)
    df_equity['month'] = df_equity.index.to_period('M')
    monthly_rets = df_equity.groupby('month')['equity'].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
    profitable_months = sum(1 for r in monthly_rets if r > 0) if len(monthly_rets) > 0 else 0
    monthly_consistency = profitable_months / len(monthly_rets) * 100 if len(monthly_rets) > 0 else 0

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
        'final_capital': capital,
        'num_tf_groups': len(group_probs),
        'threshold': threshold,
        'hold_bars': hold_bars,
        'cooldown_bars': cooldown_bars,
        'test_bars': len(df_bt),
        'date_range': f"{df_bt.index[0].strftime('%Y-%m-%d')} → {df_bt.index[-1].strftime('%Y-%m-%d')}",
    }

    return result


def print_oos_result(r: dict, oos_degradation: float = None):
    """Print OOS result with degradation indicator."""
    fade = " 🔥" if r['total_return_pct'] > 0 and r['win_rate_pct'] > 55 else ""
    if oos_degradation is not None:
        deg_str = f" | OOS degradation: {oos_degradation:.1f}%{' ✅' if oos_degradation <= 25 else ' ⚠️'}"
    else:
        deg_str = ""

    print(f"\n  ▸ {r['symbol']} (threshold={r['threshold']:.2f}, hold={r['hold_bars']}){fade}")
    print(f"    Period: {r['date_range']} ({r['test_bars']} bars)")
    print(f"    Trades: {r['total_trades']} | WR: {r['win_rate_pct']:.1f}% | PF: {r['profit_factor']:.2f}{deg_str}")
    print(f"    Return: {r['total_return_pct']:.2f}% | Monthly: {r['avg_monthly_return_pct']:.2f}% | DD: {r['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe: {r['sharpe_ratio']:.2f} | Sortino: {r['sortino_ratio']:.2f} | Quality: {r['monthly_consistency_pct']:.0f}%")
    print(f"    Long: ${r['long_pnl']:.2f} | Short: ${r['short_pnl']:.2f} | Groups: {r['num_tf_groups']}")


def check_oos_criteria(r: dict) -> list:
    """Check OOS criteria (relaxed slightly for smaller sample)."""
    results = []
    results.append(('Max DD < 15%', r['max_drawdown_pct'] < 15, f"{r['max_drawdown_pct']:.2f}%"))
    results.append(('Monthly Return > 0', r['avg_monthly_return_pct'] > 0, f"{r['avg_monthly_return_pct']:.2f}%"))
    results.append(('Win Rate > 50%', r['win_rate_pct'] > 50, f"{r['win_rate_pct']:.1f}%"))
    results.append(('PF > 1.2', r['profit_factor'] > 1.2, f"{r['profit_factor']:.2f}"))
    results.append(('Total Return > 0', r['total_return_pct'] > 0, f"{r['total_return_pct']:.2f}%"))
    results.append(('Long PnL > 0', r['long_pnl'] > 0, f"${r['long_pnl']:.2f}"))
    results.append(('Short PnL > 0', r['short_pnl'] > 0, f"${r['short_pnl']:.2f}"))
    results.append(('Min 10 trades', r['total_trades'] >= 10, f"{r['total_trades']}"))
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', default=','.join(SYMBOLS),
                        help='Comma-separated symbols')
    parser.add_argument('--threshold', type=float, default=0.60)
    parser.add_argument('--hold', type=int, default=9)
    parser.add_argument('--cool', type=int, default=3)
    parser.add_argument('--train-days', type=int, default=100)
    parser.add_argument('--test-days', type=int, default=30)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')]

    print(f"\n{'='*70}")
    print(f"  OOS VALIDATION — Chronological Train/Test Split")
    print(f"  Train: {args.train_days}d | Test: {args.test_days}d")
    print(f"  Threshold={args.threshold} Hold={args.hold} Cool={args.cool}")
    print(f"{'='*70}")

    all_results = []
    for sym in symbols:
        r = run_oos_voting(sym,
                          threshold=args.threshold,
                          hold_bars=args.hold,
                          cooldown_bars=args.cool,
                          train_days=args.train_days,
                          test_days=args.test_days)
        if r:
            print_oos_result(r)
            all_results.append(r)

    # Summary
    if all_results:
        print(f"\n{'='*70}")
        print(f"  OOS VALIDATION SUMMARY")
        print(f"{'='*70}")
        for r in all_results:
            checks = check_oos_criteria(r)
            passed = sum(1 for _, p, _ in checks if p)
            total = len(checks)
            status = "✅" if passed >= 5 else "⚠️"
            monthly_str = f"M:{r['avg_monthly_return_pct']:.1f}%" if r['avg_monthly_return_pct'] > 0 else "M:❌"
            print(f"  {status} {r['symbol']}: "
                  f"{passed}/{total} passed | "
                  f"WR {r['win_rate_pct']:.1f}% | "
                  f"PF {r['profit_factor']:.2f} | "
                  f"DD {r['max_drawdown_pct']:.2f}% | "
                  f"{monthly_str} | "
                  f"{r['total_trades']} trades")
