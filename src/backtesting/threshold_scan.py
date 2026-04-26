"""
Threshold Scan — Find Optimal Entry Threshold Per Symbol
=========================================================
Efficient approach: train models ONCE per symbol, then evaluate
all thresholds (0.50-0.70, step 0.05) on the same predictions.

Usage: python scripts/threshold_scan.py 2>&1 | tee data/threshold_scan.log
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json, warnings, requests, random
warnings.filterwarnings('ignore')
import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
           'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT']
TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']
SEEDS = [42, 101, 202, 303, 404]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
COMMISSION = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0


def fetch_ohlcv(symbol, start_time, end_time):
    """5m OHLCV from Binance Futures API."""
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    all_bars, last_ts = [], start_ms
    while last_ts < end_ms:
        params = {'symbol': symbol, 'interval': '5m', 'limit': 1000,
                  'startTime': last_ts, 'endTime': end_ms}
        r = requests.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=30)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch:
            break
        all_bars.extend(batch)
        last_ts = batch[-1][0] + 1
        if len(batch) < 1000:
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
    return df.set_index('open_time')[['open', 'high', 'low', 'close', 'volume']]


def train_models(feat_df, split_date):
    """Train 5 TF groups on train split, return (models, test_preds_index)."""
    train = feat_df[feat_df.index < split_date].copy()
    test = feat_df[feat_df.index >= split_date].copy()

    feature_sets = {
        'full': [c for c in feat_df.columns if c != 'target']
    }
    for tf in ['m15', 'm30', 'h1', 'h4']:
        prefix = f"{tf}_"
        feature_sets[tf] = [c for c in feat_df.columns if c.startswith(prefix) and c != 'target']

    models_group = {}
    for tf in TF_GROUPS:
        features = feature_sets[tf]
        if len(features) < 10:
            continue
        y_train = train['target']
        valid = np.isfinite(train[features].fillna(0)).all(axis=1) & np.isfinite(y_train)
        train_clean = train.loc[valid]
        y_clean = y_train[valid]
        if len(train_clean) < 1000:
            continue
        tf_models = []
        for seed in SEEDS:
            random.seed(seed * 7)
            n_feat = max(15, int(len(features) * random.uniform(0.5, 0.75)))
            sampled = random.sample(list(features), n_feat)
            X = train_clean[sampled].fillna(0).clip(-10, 10)
            model = xgb.XGBClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.04,
                subsample=0.6, colsample_bytree=0.6,
                reg_alpha=1.0, reg_lambda=3.0, min_child_weight=7, gamma=1.0,
                random_state=seed, verbosity=0, use_label_encoder=False,
                eval_metric='logloss', early_stopping_rounds=20
            )
            model.fit(X, y_clean,
                      eval_set=[(X[:min(3000, len(X)//5)], y_clean[:min(3000, len(y_clean)//5)])],
                      verbose=False)
            tf_models.append((str(seed), model, sampled))
        if tf_models:
            models_group[tf] = tf_models
            print(f"      {tf}: {len(tf_models)} seeds ({n_feat} feats)")
    return models_group, test


def predict_oos(models_group, test_df):
    """Get ensemble probabilities for test set."""
    group_probs = {}
    for tf in TF_GROUPS:
        if tf not in models_group:
            continue
        all_probs = []
        for seed, model, sampled in models_group[tf]:
            avail = [c for c in sampled if c in test_df.columns]
            if len(avail) < 5:
                continue
            X = test_df[avail].fillna(0).clip(-10, 10)
            all_probs.append(model.predict_proba(X)[:, 1])
        if all_probs:
            group_probs[tf] = np.mean(all_probs, axis=0)
    if len(group_probs) < 2:
        return None
    prob_stack = np.column_stack([group_probs[tf] for tf in group_probs])
    return np.mean(prob_stack, axis=1)


def simulate(ohlcv, probs, threshold, hold=9, cool=3):
    """Simulate trades at given threshold. Returns metrics dict."""
    df = pd.DataFrame({'proba': probs}, index=ohlcv.index)
    df = df.join(ohlcv, how='inner')

    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    timestamps = [df.index[0]]
    cooldown, hold_rem = 0, 0
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    long_pnl = short_pnl = 0.0

    for idx in range(len(df)):
        row = df.iloc[idx]
        price = float(row['close'])
        ts = df.index[idx]

        if cooldown > 0:
            cooldown -= 1
        if position != 0:
            hold_rem -= 1
            if hold_rem <= 0:
                exit_px = price * (1 - SLIPPAGE) if position == 1 else price * (1 + SLIPPAGE)
                raw = entry_qty * (exit_px - entry_price) if position == 1 else entry_qty * (entry_price - exit_px)
                comm = (entry_qty * entry_price + entry_qty * exit_px) * COMMISSION
                pnl = raw - comm
                capital += raw
                if position == 1:
                    long_pnl += pnl
                else:
                    short_pnl += pnl
                trades.append({'direction': 'long' if position == 1 else 'short', 'pnl_net': pnl})
                position = 0
                cooldown = cool
        if position == 0 and cooldown <= 0:
            prob = float(df.iloc[idx]['proba'])
            if price <= 0 or np.isnan(prob):
                continue
            direction = 0
            if prob >= threshold:
                direction = 1
                entry_price = price * (1 + SLIPPAGE)
            elif prob <= (1 - threshold):
                direction = -1
                entry_price = price * (1 - SLIPPAGE)
            if direction != 0:
                entry_qty = (capital * 0.15) / entry_price
                position = direction
                hold_rem = hold
        equity_curve.append(capital)
        timestamps.append(ts)

    if len(trades) < 5:
        return None

    equity = np.array(equity_curve)
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max((peak - equity) / peak)) * 100
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    wr = wins / len(trades) * 100
    gp = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gl = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    pf = gp / gl if gl > 0 else float('inf')
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    months = max(total_days / 30.44, 0.5)
    monthly_ret = total_ret / months
    rets = np.diff(equity) / equity[:-1]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365*24*12) if len(rets) > 0 and np.std(rets) > 0 else 0
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(365*24*12) if len(downside) > 0 and np.std(downside) > 0 else 0

    df_eq = pd.DataFrame({'eq': equity}, index=timestamps)
    df_eq['month'] = df_eq.index.to_period('M')
    mrets = df_eq.groupby('month')['eq'].apply(lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100)
    consistency = sum(1 for r in mrets if r > 0) / len(mrets) * 100 if len(mrets) > 0 else 0

    return {
        'trades': len(trades),
        'wr': wr, 'pf': pf,
        'monthly': monthly_ret,
        'dd': max_dd,
        'sharpe': sharpe,
        'sortino': sortino,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl,
        'consistency': consistency,
        'total_ret': total_ret,
    }


def scan_symbol(symbol):
    """Full train + predict + sweep for one symbol."""
    print(f"\n{'='*65}")
    print(f"  {symbol}")
    print(f"{'='*65}")

    end = datetime.now()
    start = end - timedelta(days=135)
    ohlcv = fetch_ohlcv(symbol, start, end)
    if ohlcv is None or len(ohlcv) < 2000:
        print(f"  ❌ OHLCV: {len(ohlcv) if ohlcv is not None else 0}")
        return None

    cache = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    if not cache:
        print(f"  ❌ No cache")
        return None
    feat_df = pd.read_parquet(max(cache, key=lambda p: p.stat().st_mtime))
    split_date = feat_df.index[int(len(feat_df) * 0.80)]
    print(f"  Data: {len(ohlcv)} bars OHLCV, {len(feat_df)} cache rows")
    print(f"  Split: {split_date.date()}")

    print(f"  Training {len(TF_GROUPS)} TF groups...")
    models, test_feat = train_models(feat_df, split_date)
    if len(models) < 2:
        return None

    probs = predict_oos(models, test_feat)
    if probs is None:
        return None

    ohlcv_test = ohlcv[ohlcv.index >= split_date - timedelta(hours=12)]
    df_aligned = ohlcv_test.join(pd.DataFrame({'proba': probs}, index=test_feat.index), how='inner')
    df_aligned = df_aligned.iloc[200:]  # warmup
    ohlcv_t = df_aligned[['open', 'high', 'low', 'close', 'volume']]
    probs_t = df_aligned['proba'].values
    print(f"  Test period: {df_aligned.index[0].date()} → {df_aligned.index[-1].date()} ({len(df_aligned)} bars)")

    results = {}
    for t in THRESHOLDS:
        r = simulate(ohlcv_t, probs_t, threshold=t)
        if r:
            results[t] = r
            marker = " 🔥" if r['wr'] > 50 and r['pf'] > 1.2 else ""
            print(f"    thresh={t:.2f}: WR {r['wr']:.1f}% | PF {r['pf']:.2f} | "
                  f"M:{r['monthly']:.1f}% | DD {r['dd']:.2f}% | {r['trades']} trades{marker}")

    return results


def best_threshold(results):
    """Pick best threshold: max Sharpe, or fallback to PF."""
    if not results:
        return None, None
    best_t = max(results, key=lambda t: results[t]['sharpe'])
    return best_t, results[best_t]


if __name__ == '__main__':
    print(f"\n{'='*65}")
    print(f"  THRESHOLD SCAN — 8 Symbols × {len(THRESHOLDS)} Thresholds")
    print(f"  Method: train models ONCE, sweep thresholds on same predictions")
    print(f"{'='*65}")

    all_best = {}
    for sym in SYMBOLS:
        results = scan_symbol(sym)
        if results:
            best_t, best_r = best_threshold(results)
            all_best[sym] = {'threshold': best_t, 'metrics': best_r, 'all': results}

    # Summary
    print(f"\n{'='*65}")
    print(f"  BEST THRESHOLD PER SYMBOL")
    print(f"{'='*65}")
    print(f"  {'Symbol':<12} {'Thresh':<8} {'WR':<8} {'PF':<8} {'Monthly':<10} {'DD':<8} {'Trades':<8}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
    for sym in SYMBOLS:
        if sym in all_best:
            b = all_best[sym]
            r = b['metrics']
            if r:
                print(f"  {sym:<12} {b['threshold']:<8.2f} "
                      f"{r['wr']:<8.1f} {r['pf']:<8.2f} "
                      f"{r['monthly']:<10.2f} {r['dd']:<8.2f} {r['trades']:<8d}")
        else:
            print(f"  {sym:<12} {'FAILED':<8}")

    # Save
    out = {
        'best_per_symbol': {
            sym: {'threshold': v['threshold'], 'metrics': v['metrics']}
            for sym, v in all_best.items()
        }
    }
    with open('data/threshold_scan_results.json', 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: data/threshold_scan_results.json")
