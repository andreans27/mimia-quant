"""
Multi-TF Voting Backtest - average probabilities from 5 model groups:
  Original 5m (all features) + m15 + m30 + h1 + h4 TF-specific

Architecture:
  1. Fetch 5m OHLCV from Binance (public API, no auth needed)
  2. Load cached multi-TF features (has 372 feature columns + target)
  3. Load all 5 model groups (25 models total: 5 seeds × 5 groups)
  4. For each group: predict proba on each 5m bar (using that TF's feature prefix)
  5. Vote: average probs across all groups → signal on threshold cross
  6. Hold N bars, cooldown M bars, compute PnL with fees

Usage: python scripts/ml_voting_backtest.py sweep <thresholds> <hold> <cool>
       python scripts/ml_voting_backtest.py single <symbol> <threshold> <hold> <cool>
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
warnings.filterwarnings('ignore')

import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")

# Symbol mapping: Futures symbol → Spot symbol for OHLCV data fetch.
# Some tokens use "1000" prefix on Futures but not on Spot (e.g. 1000PEPEUSDT = PEPEUSDT).
SPOT_SYMBOL_MAP = {
    '1000PEPEUSDT': 'PEPEUSDT',
}

SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT',
           'DOGEUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT']

TAKER_FEE = 0.0004   # 0.04%
SLIPPAGE = 0.0005    # 0.05%
INITIAL_CAPITAL = 5000.0


def fetch_5m_ohlcv(symbol: str, days: int = 130) -> pd.DataFrame:
    """Fetch 5m OHLCV from Binance public API."""
    # Map Futures symbol → Spot symbol if needed (e.g. 1000PEPEUSDT → PEPEUSDT)
    spot_symbol = SPOT_SYMBOL_MAP.get(symbol, symbol)
    end = datetime.now()
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    limit = 1000
    all_bars = []
    last_ts = start_ms
    while last_ts < end_ms:
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            'symbol': spot_symbol,
            'interval': '5m',
            'limit': limit,
            'startTime': last_ts,
            'endTime': end_ms
        }
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            break
        batch = r.json()
        if not batch or len(batch) == 0:
            break
        all_bars.extend(batch)
        last_ts = batch[-1][0] + 1
        if len(batch) < limit:
            break

    if len(all_bars) < 1000:
        print(f"    ❌ Insufficient data: {len(all_bars)} bars")
        return None

    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def load_tf_models(symbol: str, tf_group: str, seeds=None):
    """Load a group of ensemble models for a specific TF prefix.

    tf_group: 'full' (original 5m models, use all features from meta)
              'm15', 'm30', 'h1', 'h4' (TF-specific models)
    """
    if seeds is None:
        seeds = [42, 101, 202, 303, 404]

    models = []
    feature_cols = None

    if tf_group == 'full':
        # Original 5m full models: use meta's feature list
        meta_path = MODEL_DIR / f"{symbol}_ensemble_meta.json"
        if not meta_path.exists():
            print(f"    ⚠️ No full ensemble meta for {symbol}")
            return None, None
        with open(meta_path) as f:
            meta = json.load(f)
        if 'features' in meta:
            feature_cols = meta['features']
        else:
            feature_cols = meta.get('full_feature_set', [])

        for seed in seeds:
            path = MODEL_DIR / f"{symbol}_xgb_ens_{seed}.json"
            if not path.exists():
                continue
            # Get per-seed feature selection
            mf = meta.get('model_features', {}).get(str(seed), meta.get('features', []))
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            models.append((str(seed), m, mf))

    else:
        # TF-specific models: feature prefix = tf_group
        prefix = f"{tf_group}_"
        for seed in seeds:
            path = MODEL_DIR / f"{symbol}_{tf_group}_xgb_ens_{seed}.json"
            if not path.exists():
                continue
            m = xgb.XGBClassifier()
            m.load_model(str(path))
            # Load feature names from the model
            model_features = m.get_booster().feature_names
            # Ensure they have the correct prefix
            if model_features and not model_features[0].startswith(prefix):
                # Prepend prefix if missing
                model_features = [f"{prefix}{f}" if not f.startswith(prefix) else f for f in model_features]
            models.append((str(seed), m, model_features))

    if len(models) < 2:
        print(f"    ⚠️ Only {len(models)} models for {symbol}/{tf_group}")
        return None, None

    return models, feature_cols


def run_voting_backtest(symbol: str, threshold: float = 0.60,
                        hold_bars: int = 9, cooldown_bars: int = 3,
                        days_data: int = 130, warmup_bars: int = 200,
                        initial_capital: float = 5000.0,
                        commission_rate: float = 0.0004,
                        slippage_rate: float = 0.0005) -> dict:
    """Run multi-TF voting backtest."""
    tf_groups = ['full', 'm15', 'm30', 'h1', 'h4']
    seeds = [42, 101, 202, 303, 404]

    # --- 1. Fetch 5m OHLCV ---
    print(f"\n  [{symbol}] Fetching 5m OHLCV ({days_data}d)...")
    df_ohlcv = fetch_5m_ohlcv(symbol, days_data)
    if df_ohlcv is None or len(df_ohlcv) < 1000:
        print(f"  [{symbol}] ❌ Failed to fetch OHLCV")
        return None
    print(f"    OHLCV: {len(df_ohlcv)} 5m bars")

    # --- 2. Load cached features ---
    tf_suffix = '_'.join(sorted(['15m', '30m', '1h', '4h']))
    cache_candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*_{tf_suffix}.parquet"))
    if not cache_candidates:
        cache_candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    if not cache_candidates:
        print(f"  [{symbol}] ❌ No cache found")
        return None
    cache_path = max(cache_candidates, key=lambda p: p.stat().st_mtime)
    feat_df = pd.read_parquet(cache_path)
    print(f"    Features: {len(feat_df)} rows, {len(feat_df.columns)} cols")

    # --- 3. Load all model groups ---
    group_probs = {}
    for tf in tf_groups:
        models, _ = load_tf_models(symbol, tf, seeds)
        if models is None:
            print(f"  [{symbol}] ⚠️ Skipping {tf}: models not available")
            continue

        all_probs = []
        for seed, m, mf in models:
            available = [c for c in mf if c in feat_df.columns]
            if len(available) < 5:
                print(f"    ⚠️ {tf}/{seed}: only {len(available)} features available")
                continue
            X = feat_df[available].fillna(0).clip(-10, 10)
            probs = m.predict_proba(X)[:, 1]
            all_probs.append(probs)

        if all_probs:
            avg = np.nanmean(all_probs, axis=0)
            group_probs[tf] = avg
            print(f"    {tf}: {len(all_probs)} seeds loaded")

    if len(group_probs) < 2:
        print(f"  [{symbol}] ❌ Only {len(group_probs)} groups available, need >= 2")
        return None

    print(f"  Loaded {len(group_probs)} groups: {list(group_probs.keys())}")
    n_total = len(feat_df)

    # --- 4. Average all group probabilities ---
    prob_stack = np.column_stack([group_probs[tf] for tf in group_probs])
    avg_probs = np.mean(prob_stack, axis=1)

    # --- 5. Align with OHLCV ---
    # The features index may differ from OHLCV; use inner join on index
    sig_df = pd.DataFrame({'proba': avg_probs}, index=feat_df.index)
    df_bt = df_ohlcv.join(sig_df, how='inner')

    if len(df_bt) < warmup_bars + hold_bars + 10:
        print(f"  [{symbol}] ❌ Aligned bars too few: {len(df_bt)}")
        return None

    # Skip warmup
    df_bt = df_bt.iloc[warmup_bars:].copy()

    print(f"    Aligned: {len(df_bt)} bars (after warmup)")

    # --- 6. Generate signals (long only for now) ---
    prob = df_bt['proba'].values
    signals = pd.Series(False, index=df_bt.index)
    for i in range(1, len(prob)):
        if prob[i] >= threshold and prob[i-1] < threshold:
            signals.iloc[i] = True

    # --- 7. Simulate trading ---
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
                # Exit
                exit_price = price * (1 - slippage_rate) if position == 1 else price * (1 + slippage_rate)
                if position == 1:
                    raw_pnl = entry_qty * (exit_price - entry_price)
                else:
                    raw_pnl = entry_qty * (entry_price - exit_price)

                comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
                pnl_net = raw_pnl - comm

                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if position == 1 else ((entry_price - exit_price) / entry_price * 100)
                capital += raw_pnl

                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net

                trades.append({
                    'symbol': symbol,
                    'direction': 'long' if position == 1 else 'short',
                    'entry_time': entry_time,
                    'exit_time': ts,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'qty': entry_qty,
                    'pnl_net': pnl_net,
                    'pnl_pct': pnl_pct,
                    'prob_entry': entry_prob,
                    'hold_bars_actual': hold_counter - hold_remaining,
                })

                position = 0
                entry_price = 0.0
                entry_qty = 0.0
                cooldown = cooldown_bars

        # Entry check
        if position == 0 and cooldown <= 0:
            prob_val = float(df_bt.iloc[idx]['proba'])
            if price <= 0 or np.isnan(prob_val):
                cooldown = max(cooldown, 1)
                continue
            direction = 0
            if prob_val >= threshold:
                direction = 1  # long
                entry_price = price * (1 + slippage_rate)
            elif prob_val <= (1 - threshold):
                direction = -1  # short
                entry_price = price * (1 - slippage_rate)
            if direction != 0:
                position_pct = 0.15  # 15% per trade
                entry_qty = (capital * position_pct) / entry_price
                position = direction
                hold_remaining = hold_bars
                entry_time = ts
                entry_prob = prob_val
                hold_counter = hold_bars

        equity_curve.append(capital)
        timestamps.append(ts)

    # --- 8. Compute metrics ---
    if len(trades) < 5:
        print(f"  [{symbol}] ❌ Only {len(trades)} trades")
        return None

    equity = np.array(equity_curve)
    rets = np.diff(equity) / equity[:-1]

    # Total return
    total_return_pct = (capital - initial_capital) / initial_capital * 100

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd_pct = float(np.max(dd)) * 100

    # Win rate
    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    win_rate = wins / len(trades) * 100

    # Profit factor
    gross_profit = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gross_loss = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    # Monthly return
    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    months = max(total_days / 30.44, 1)
    avg_monthly_return = total_return_pct / months

    # Sharpe / Sortino
    if len(rets) > 0 and np.std(rets) > 0:
        sharpe = np.mean(rets) / np.std(rets) * np.sqrt(365 * 24 * 12)  # 5m bars annualized
        downside = rets[rets < 0]
        sortino = np.mean(rets) / np.std(downside) * np.sqrt(365 * 24 * 12) if len(downside) > 0 and np.std(downside) > 0 else 0.0
    else:
        sharpe = 0.0
        sortino = 0.0

    # Monthly consistency
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
    }

    return result


def check_voting_criteria(r: dict) -> list:
    """Check 11 voting backtest criteria. Returns list of (name, pass, value)."""
    results = []
    # 1. Max DD < 15%
    results.append(('Max DD < 15%', r['max_drawdown_pct'] < 15, f"{r['max_drawdown_pct']:.2f}%"))
    # 2. Avg monthly return >= 8%
    results.append(('Avg Monthly >= 8%', r['avg_monthly_return_pct'] >= 8, f"{r['avg_monthly_return_pct']:.2f}%"))
    # 3. Win rate > 65%
    results.append(('Win Rate > 65%', r['win_rate_pct'] > 65, f"{r['win_rate_pct']:.2f}%"))
    # 4. Profit factor > 1.5
    results.append(('PF > 1.5', r['profit_factor'] > 1.5, f"{r['profit_factor']:.2f}"))
    # 5. Sharpe > 1.5
    results.append(('Sharpe > 1.5', r['sharpe_ratio'] > 1.5, f"{r['sharpe_ratio']:.2f}"))
    # 6. Sortino > 2.0
    results.append(('Sortino > 2.0', r['sortino_ratio'] > 2.0, f"{r['sortino_ratio']:.2f}"))
    # 7. Trade frequency >= 3/day
    results.append(('Min 3 trades/day', r['total_trades'] >= 3 * 60, f"{r['total_trades']} total"))
    # 8. Monthly consistency >= 70%
    results.append(('Monthly >= 70%', r['monthly_consistency_pct'] >= 70, f"{r['monthly_consistency_pct']:.0f}%"))
    # 9. Total return positive
    results.append(('Total Return > 0', r['total_return_pct'] > 0, f"{r['total_return_pct']:.2f}%"))
    # 10. Long side profitable (if applicable)
    results.append(('Long PnL > 0', r['long_pnl'] > 0, f"${r['long_pnl']:.2f}"))
    # 11. Short side profitable (if applicable)
    results.append(('Short PnL > 0', r['short_pnl'] > 0, f"${r['short_pnl']:.2f}"))
    return results


def print_voting_results(r: dict):
    """Print backtest results."""
    print(f"\n  ▸ {r['symbol']}  (threshold={r['threshold']:.2f}, hold={r['hold_bars']}, cool={r['cooldown_bars']})")
    print(f"    Trades: {r['total_trades']} | Win Rate: {r['win_rate_pct']:.1f}% | PF: {r['profit_factor']:.2f}")
    print(f"    Return: {r['total_return_pct']:.2f}% | Monthly: {r['avg_monthly_return_pct']:.2f}% | Max DD: {r['max_drawdown_pct']:.2f}%")
    print(f"    Sharpe: {r['sharpe_ratio']:.2f} | Sortino: {r['sortino_ratio']:.2f} | Monthly Quality: {r['monthly_consistency_pct']:.0f}%")
    print(f"    Long: ${r['long_pnl']:.2f} | Short: ${r['short_pnl']:.2f} | TF groups: {r['num_tf_groups']}")

    checks = check_voting_criteria(r)
    passed = sum(1 for _, p, _ in checks if p)
    total = len(checks)
    print(f"    Criteria: {passed}/{total} passed")
    for name, passed, val in checks:
        icon = "✅" if passed else "❌"
        print(f"      {icon} {name}: {val}")


def run_voting_sweep(thresholds_str: str, hold: int = 9, cool: int = 3):
    """Run voting backtest across thresholds and symbols."""
    thresholds = [float(t.strip()) for t in thresholds_str.split(',')]

    for thresh in thresholds:
        print(f"\n{'='*70}")
        print(f"  VOTING BACKTEST | threshold={thresh:.2f} hold={hold} cool={cool}")
        print(f"{'='*70}")

        all_results = []
        for sym in SYMBOLS:
            r = run_voting_backtest(
                sym, threshold=thresh, hold_bars=hold, cooldown_bars=cool,
                days_data=130, warmup_bars=200
            )
            if r:
                print_voting_results(r)
                all_results.append(r)

        if all_results:
            passed = sum(1 for r in all_results for _, p, _ in check_voting_criteria(r) if p)
            total = sum(len(check_voting_criteria(r)) for r in all_results)
            print(f"\n  Threshold {thresh:.2f}: {len([r for r in all_results if sum(1 for _, p, _ in check_voting_criteria(r) if p) >= 9])}/{len(all_results)} symbols passed (≥9/11)")

    # Final summary
    print(f"\n{'='*70}")
    print(f"VOTING BACKTEST SUMMARY")
    print(f"{'='*70}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['sweep', 'single'])
    parser.add_argument('thresholds', help='e.g. 0.50,0.55,0.60 or single 0.60')
    parser.add_argument('hold', type=int, default=9, nargs='?')
    parser.add_argument('cool', type=int, default=3, nargs='?')
    args = parser.parse_args()

    if args.mode == 'sweep':
        run_voting_sweep(args.thresholds, args.hold, args.cool)
    else:
        # Single symbol
        sym = args.thresholds  # Interpret first arg as symbol for 'single' mode
        thresh = float(args.hold) if args.hold else 0.60
        hold = args.cool if args.cool else 9
        cool = 3
        r = run_voting_backtest(sym, threshold=thresh, hold_bars=hold, cooldown_bars=cool)
        if r:
            print_voting_results(r)
