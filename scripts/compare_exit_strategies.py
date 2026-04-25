"""
Compare Exit Strategies — run multiple exit types on the same proba array.

Usage: python scripts/compare_exit_strategies.py [symbols...]

Exits tested:
  baseline          time-based only (hold=9, cooldown=3)
  sl_2pct           + 2% stop loss
  sl_3pct           + 3% stop loss
  tp_3pct           + 3% take profit
  tp_5pct           + 5% take profit
  sl_tp             + 2% SL + 3% TP
  signal_reversal   exit when proba crosses back below threshold
  trailing_1pct     + 1% trailing stop from peak
  trailing_2pct     + 2% trailing stop from peak
"""

import sys; sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json, warnings, time, argparse
import xgboost as xgb
warnings.filterwarnings('ignore')

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")

TAKER_FEE = 0.0004
SLIPPAGE = 0.0005
INITIAL_CAPITAL = 5000.0

SEEDS = [42, 101, 202, 303, 404]
TF_GROUPS = ['full', 'm15', 'm30', 'h1', 'h4']

# ─── Exit strategy presets ───────────────────────────────────────────────
EXIT_STRATEGIES = {
    'baseline':       {'type': 'baseline',       'sl_pct': 0,    'tp_pct': 0,    'trail_pct': 0,   'signal_exit': False},
    'sl_2pct':        {'type': 'stop_loss',      'sl_pct': 2.0,  'tp_pct': 0,    'trail_pct': 0,   'signal_exit': False},
    'sl_3pct':        {'type': 'stop_loss',      'sl_pct': 3.0,  'tp_pct': 0,    'trail_pct': 0,   'signal_exit': False},
    'tp_3pct':        {'type': 'take_profit',    'sl_pct': 0,    'tp_pct': 3.0,  'trail_pct': 0,   'signal_exit': False},
    'tp_5pct':        {'type': 'take_profit',    'sl_pct': 0,    'tp_pct': 5.0,  'trail_pct': 0,   'signal_exit': False},
    'sl_tp':          {'type': 'sl_tp',          'sl_pct': 2.0,  'tp_pct': 3.0,  'trail_pct': 0,   'signal_exit': False},
    'signal_reversal':{'type': 'signal_reversal','sl_pct': 0,    'tp_pct': 0,    'trail_pct': 0,   'signal_exit': True},
    'trailing_1pct':  {'type': 'trailing_stop',  'sl_pct': 0,    'tp_pct': 0,    'trail_pct': 1.0, 'signal_exit': False},
    'trailing_2pct':  {'type': 'trailing_stop',  'sl_pct': 0,    'tp_pct': 0,    'trail_pct': 2.0, 'signal_exit': False},
}


def load_models(symbol):
    """Load all model groups for a symbol. Returns {tf: [(seed, model, features)]}"""
    groups = {}
    for tf in TF_GROUPS:
        models = []
        if tf == 'full':
            meta_path = MODEL_DIR / f'{symbol}_ensemble_meta.json'
            if not meta_path.exists():
                meta_path = MODEL_DIR / f'{symbol}_xgb_meta.json'
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            for seed in SEEDS:
                path = MODEL_DIR / f'{symbol}_xgb_ens_{seed}.json'
                if not path.exists():
                    continue
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                mf = meta.get('model_features', {}).get(str(seed), meta.get('features', []))
                models.append((str(seed), m, mf))
        else:
            for seed in SEEDS:
                path = MODEL_DIR / f'{symbol}_{tf}_xgb_ens_{seed}.json'
                if not path.exists():
                    continue
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                mf = m.get_booster().feature_names
                prefix = f'{tf}_'
                if mf and not mf[0].startswith(prefix):
                    mf = [f'{prefix}{f}' if not f.startswith(prefix) else f for f in mf]
                models.append((str(seed), m, mf))
        if len(models) >= 2:
            groups[tf] = models
    return groups


def compute_proba(symbol, groups, feat_df):
    """Compute ensemble probability for every bar. Returns numpy array."""
    n = len(feat_df)
    all_probs = []
    for tf, models in groups.items():
        tf_probs = []
        for seed, m, mf in models:
            available = [c for c in mf if c in feat_df.columns]
            X = feat_df[available].fillna(0).clip(-10, 10)
            probs = m.predict_proba(X)[:, 1]
            tf_probs.append(probs)
        if tf_probs:
            all_probs.append(np.nanmean(tf_probs, axis=0))
    if len(all_probs) < 2:
        return None
    return np.nanmean(all_probs, axis=0)


def run_backtest(symbol, df, threshold, hold_bars, cooldown_bars, exit_config):
    """Run a single backtest with configurable exit strategy.

    df: pre-aligned DataFrame with 'close' and 'proba' columns, already trimmed of warmup.
    exit_config keys:
      sl_pct:     stop loss % (0 = disabled)
      tp_pct:     take profit % (0 = disabled)
      trail_pct:  trailing stop % (0 = disabled)
      signal_exit: exit when proba crosses back below threshold
    """
    prob = df['proba'].values
    n = len(df)
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    timestamps = [df.index[0]]
    cooldown = 0
    hold_remaining = 0
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    entry_idx = 0
    peak_price = 0.0
    long_pnl = 0.0
    short_pnl = 0.0

    sl_pct = exit_config.get('sl_pct', 0)
    tp_pct = exit_config.get('tp_pct', 0)
    trail_pct = exit_config.get('trail_pct', 0)
    signal_exit = exit_config.get('signal_exit', False)

    for idx in range(n):
        row = df.iloc[idx]
        price = float(row['close'])
        ts = df.index[idx]
        prob_val = float(row['proba'])

        if cooldown > 0:
            cooldown -= 1

        # ── Exit logic ──────────────────────────────────────────────
        exit_reason = None
        if position != 0:
            hold_remaining -= 1
            exit_this_bar = False

            # 1. Time-based exit
            if hold_remaining <= 0:
                exit_reason = 'hold_expiry'
                exit_this_bar = True

            # 2. Stop loss
            if not exit_this_bar and sl_pct > 0:
                if position == 1:
                    pnl_pct = (price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - price) / entry_price * 100
                if pnl_pct <= -sl_pct:
                    exit_reason = f'stop_loss_{sl_pct}%'
                    exit_this_bar = True

            # 3. Take profit
            if not exit_this_bar and tp_pct > 0:
                if position == 1:
                    pnl_pct = (price - entry_price) / entry_price * 100
                else:
                    pnl_pct = (entry_price - price) / entry_price * 100
                if pnl_pct >= tp_pct:
                    exit_reason = f'take_profit_{tp_pct}%'
                    exit_this_bar = True

            # 4. Trailing stop
            if not exit_this_bar and trail_pct > 0:
                if position == 1:
                    peak_price = max(peak_price, price)
                    trail_pnl = (peak_price - entry_price) / entry_price * 100
                    # Exit if retraced trail_pct % from peak
                    if trail_pnl > trail_pct:
                        retrace = (peak_price - price) / peak_price * 100
                        if retrace >= trail_pct:
                            exit_reason = f'trailing_{trail_pct}%'
                            exit_this_bar = True
                else:
                    peak_price = min(peak_price, price)
                    trail_pnl = (entry_price - peak_price) / entry_price * 100
                    if trail_pnl > trail_pct:
                        retrace = (price - peak_price) / peak_price * 100
                        if retrace >= trail_pct:
                            exit_reason = f'trailing_{trail_pct}%'
                            exit_this_bar = True

            # 5. Signal reversal exit
            if not exit_this_bar and signal_exit:
                if position == 1 and prob_val < threshold:
                    exit_reason = 'signal_reversal'
                    exit_this_bar = True
                elif position == -1 and prob_val > (1 - threshold):
                    exit_reason = 'signal_reversal'
                    exit_this_bar = True

            if exit_this_bar:
                # Execute exit
                exit_price = price * (1 - SLIPPAGE) if position == 1 else price * (1 + SLIPPAGE)
                if position == 1:
                    raw_pnl = entry_qty * (exit_price - entry_price)
                else:
                    raw_pnl = entry_qty * (entry_price - exit_price)

                comm = (entry_qty * entry_price + entry_qty * exit_price) * TAKER_FEE
                pnl_net = raw_pnl - comm
                pnl_pct = ((exit_price - entry_price) / entry_price * 100) if position == 1 else ((entry_price - exit_price) / entry_price * 100)
                capital += raw_pnl

                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net

                trades.append({
                    'direction': 'long' if position == 1 else 'short',
                    'exit_reason': exit_reason,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_net': pnl_net,
                    'pnl_pct': pnl_pct,
                    'prob_entry': float(row['proba']),
                    'hold_bars': hold_bars + 1 + cooldown - hold_remaining,
                })

                position = 0
                entry_price = 0.0
                entry_qty = 0.0
                peak_price = 0.0
                cooldown = cooldown_bars

        # ── Entry logic (level-based, inline with backtest) ────────
        if position == 0 and cooldown <= 0:
            direction = 0
            if prob_val >= threshold:
                direction = 1  # long
                entry_price = price * (1 + SLIPPAGE)
            elif prob_val <= (1 - threshold):
                direction = -1  # short
                entry_price = price * (1 - SLIPPAGE)

            if direction != 0:
                position_pct = 0.15
                entry_qty = (capital * position_pct) / entry_price
                position = direction
                hold_remaining = hold_bars
                entry_idx = idx
                peak_price = entry_price  # init peak for trailing

        equity_curve.append(capital)
        timestamps.append(ts)

    # ── Compute metrics ────────────────────────────────────────────
    if len(trades) < 5:
        return None

    equity = np.array(equity_curve)
    rets = np.diff(equity) / equity[:-1]

    total_return_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    peak_eq = np.maximum.accumulate(equity)
    dd = (peak_eq - equity) / peak_eq
    max_dd_pct = float(np.max(dd)) * 100

    wins = sum(1 for t in trades if t['pnl_net'] > 0)
    win_rate = wins / len(trades) * 100

    gross_profit = sum(t['pnl_net'] for t in trades if t['pnl_net'] > 0)
    gross_loss = abs(sum(t['pnl_net'] for t in trades if t['pnl_net'] <= 0))
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    total_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    months = max(total_days / 30.44, 1)
    avg_monthly_return = total_return_pct / months

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

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        er = t['exit_reason']
        exit_reasons[er] = exit_reasons.get(er, 0) + 1

    return {
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
        'exit_reasons': exit_reasons,
    }


def run_all_strategies(symbol, threshold=0.60, hold_bars=9, cooldown_bars=3, days_data=130, warmup_bars=200):
    """Run all exit strategies on one symbol."""
    print(f"\n{'='*60}")
    print(f"  {symbol}")
    print(f"{'='*60}")
    print(f"  Loading data...", end=' ')

    # Fetch OHLCV
    from ml_voting_backtest import fetch_5m_ohlcv
    ohlcv = fetch_5m_ohlcv(symbol, days_data)
    if ohlcv is None or len(ohlcv) < 1000:
        print(f"❌ No OHLCV data")
        return None
    print(f"{len(ohlcv)} bars")

    # Load features
    tf_suffix = '_'.join(sorted(['15m', '30m', '1h', '4h']))
    candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*_{tf_suffix}.parquet"))
    if not candidates:
        candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    if not candidates:
        print(f"  ❌ No feature cache")
        return None
    cache_path = max(candidates, key=lambda p: p.stat().st_mtime)
    feat_df = pd.read_parquet(cache_path)
    print(f"  Features: {len(feat_df)} rows")

    # Load models
    groups = load_models(symbol)
    if len(groups) < 2:
        print(f"  ❌ Not enough model groups ({len(groups)})")
        return None
    print(f"  Models: {len(groups)} TF groups")

    # Compute proba once
    print(f"  Computing proba...", end=' ')
    t0 = time.time()
    proba = compute_proba(symbol, groups, feat_df)
    if proba is None:
        print(f"❌ Proba computation failed")
        return None
    print(f"{len(proba)} values ({time.time()-t0:.1f}s)")

    # Align indices
    sig_df = pd.DataFrame({'proba': proba}, index=feat_df.index)
    ohlcv_aligned = ohlcv.join(sig_df, how='inner').iloc[warmup_bars:].copy()
    aligned_proba = ohlcv_aligned['proba'].values
    print(f"  Aligned: {len(ohlcv_aligned)} bars")

    # Run each strategy
    results = {}
    for name, config in EXIT_STRATEGIES.items():
        print(f"  ▶ {name}...", end=' ')
        t0 = time.time()
        r = run_backtest(symbol, ohlcv_aligned, threshold, hold_bars, cooldown_bars, config)
        if r:
            results[name] = r
            exit_counts = ', '.join(f"{k}:{v}" for k, v in sorted(r['exit_reasons'].items()))
            print(f"WR={r['win_rate_pct']:.1f}% PF={r['profit_factor']:.2f} Mo={r['avg_monthly_return_pct']:.1f}% DD={r['max_drawdown_pct']:.2f}% Trades={r['total_trades']} [{exit_counts}] ({time.time()-t0:.1f}s)")
        else:
            print(f"❌ (< 5 trades)")

    return results


def print_comparison_table(all_results):
    """Print side-by-side comparison for all symbols."""
    strategies = list(EXIT_STRATEGIES.keys())
    symbols = list(all_results.keys())

    print(f"\n{'='*110}")
    print(f"  EXIT STRATEGY COMPARISON")
    print(f"{'='*110}")

    # Header
    header = f"{'Strategy':<18} {'WR':>6} {'PF':>6} {'Mo%':>6} {'DD%':>6} {'Trades':>7} {'Sharpe':>7} {'Return%':>8} {'PnL$':>9}"
    print(f"  {header}")

    aggs = {s: {'wr': [], 'pf': [], 'mo': [], 'dd': [], 'n': [], 'sharpe': [], 'ret': [], 'pnl': []} for s in strategies}

    for sym in symbols:
        print(f"  {'─'*105}")
        print(f"  │ {sym:<16}")
        for s in strategies:
            r = all_results[sym].get(s)
            if r:
                wr = r['win_rate_pct']
                pf = r['profit_factor']
                mo = r['avg_monthly_return_pct']
                dd = r['max_drawdown_pct']
                n = r['total_trades']
                sh = r['sharpe_ratio']
                ret = r['total_return_pct']
                pnl = r['final_capital'] - INITIAL_CAPITAL
                line = f"  │ {s:<18} {wr:>5.1f}% {pf:>6.2f} {mo:>5.1f}% {dd:>5.2f}% {n:>7} {sh:>7.2f} {ret:>7.2f}% ${pnl:>7.2f}"
                print(line)
                aggs[s]['wr'].append(wr)
                aggs[s]['pf'].append(pf)
                aggs[s]['mo'].append(mo)
                aggs[s]['dd'].append(dd)
                aggs[s]['n'].append(n)
                aggs[s]['sharpe'].append(sh)
                aggs[s]['ret'].append(ret)
                aggs[s]['pnl'].append(pnl)
            else:
                print(f"  │ {s:<18} {'N/A':>6}")

    # Average row
    print(f"  {'─'*105}")
    print(f"  │ {'AVERAGE':<16}")
    for s in strategies:
        d = aggs[s]
        if d['wr']:
            wr_avg = np.mean(d['wr'])
            pf_avg = np.mean(d['pf'])
            mo_avg = np.mean(d['mo'])
            dd_avg = np.mean(d['dd'])
            n_sum = sum(d['n'])
            sh_avg = np.mean(d['sharpe'])
            ret_avg = np.mean(d['ret'])
            pnl_sum = sum(d['pnl'])
            print(f"  │ {s:<18} {wr_avg:>5.1f}% {pf_avg:>6.2f} {mo_avg:>5.1f}% {dd_avg:>5.2f}% {n_sum:>7} {sh_avg:>7.2f} {ret_avg:>7.2f}% ${pnl_sum:>7.2f}")
        else:
            print(f"  │ {s:<18} {'N/A':>6}")

    print(f"  {'─'*105}")


# ─── Main ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare exit strategies')
    parser.add_argument('symbols', nargs='*', default=[],
                        help='Symbols to test (default: SOLUSDT BTCUSDT APTUSDT 1000PEPEUSDT)')
    args = parser.parse_args()

    symbols = args.symbols if args.symbols else ['SOLUSDT', 'BTCUSDT', 'APTUSDT', '1000PEPEUSDT']

    all_results = {}
    for sym in symbols:
        t0 = time.time()
        r = run_all_strategies(sym, threshold=0.60, hold_bars=9, cooldown_bars=3,
                               days_data=130, warmup_bars=200)
        if r:
            all_results[sym] = r
        print(f"\n  ⏱ Total for {sym}: {time.time()-t0:.0f}s")

    if all_results:
        print_comparison_table(all_results)

    print(f"\n  Done.")
