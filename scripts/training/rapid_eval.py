#!/usr/bin/env python3
"""
Rapid evaluation script: load cached features → train XGBoost → backtest → metrics.
Gives quick WR/PF/volatility read on any symbol without full 25-model training.
"""
import sys, os, json, warnings, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

import pandas as pd
import numpy as np
import xgboost as xgb

def load_features(symbol: str) -> pd.DataFrame:
    """Load cached feature parquet."""
    cache_dir = ROOT / 'data' / 'ml_cache'
    # Try 130d first, then 120d — handle naming like {symbol}_5m_130d_features.parquet
    for suffix in ['5m_130d_features.parquet', '5m_120d_9c_15m_1h_30m_4h.parquet']:
        path1 = cache_dir / f'{symbol}_{suffix}'
        path2 = cache_dir / f'{symbol[:-4]}_{suffix}'  # Try without USDT suffix
        if path1.exists():
            print(f"  Loading {path1.name} ({path1.stat().st_size / 1e6:.0f} MB)")
            df = pd.read_parquet(path1)
            return df
        if path2.exists():
            print(f"  Loading {path2.name} ({path2.stat().st_size / 1e6:.0f} MB)")
            df = pd.read_parquet(path2)
            return df
    # Fallback: glob for any parquet starting with symbol
    import glob
    matches = sorted(glob.glob(str(cache_dir / f'{symbol}*.parquet')))
    matches += sorted(glob.glob(str(cache_dir / f'{symbol[:-4]}*.parquet')))
    if matches:
        path = Path(matches[0])
        print(f"  Loading {path.name} ({path.stat().st_size / 1e6:.0f} MB) [fallback]")
        df = pd.read_parquet(path)
        return df
    raise FileNotFoundError(f"No feature cache for {symbol}")

def compute_targets(df: pd.DataFrame, horizon: int = 9) -> pd.Series:
    """Compute 1/0 targets: close > open after HOLD_BARS horizon."""
    close_col = 'close'
    open_col = 'open'
    if close_col not in df.columns or open_col not in df.columns:
        # Try multi-level columns
        for col in df.columns:
            if 'close' in str(col).lower():
                close_col = col
                break
        for col in df.columns:
            if 'open' in str(col).lower():
                open_col = col
                break
    
    future_close = df[close_col].shift(-horizon)
    target = (future_close > df[open_col]).astype(int)
    # Remove NaN from shift
    valid = target.notna()
    return target[valid].astype(int)

def filter_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only numeric feature columns, drop price/date cols."""
    exclude = {'open', 'high', 'low', 'close', 'volume', 'timestamp', 'date', 
               'time', 'datetime', 'target', 'symbol'}
    numeric_df = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in exclude if c in numeric_df.columns]
    features = numeric_df.drop(columns=cols_to_drop)
    # Drop zero-variance columns
    features = features.loc[:, features.std() > 0]
    return features

def main():
    parser = argparse.ArgumentParser(description='Rapid ML evaluation')
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--n_estimators', type=int, default=200)
    parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--horizon', type=int, default=9)
    args = parser.parse_args()
    
    symbol = args.symbol
    print(f"\n{'='*60}")
    print(f"RAPID EVAL: {symbol}")
    print(f"{'='*60}")
    
    # 1. Load features
    df = load_features(symbol)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.index[0] if hasattr(df.index,'dtype') else df.iloc[0].get('timestamp','?')} "
          f"→ {df.index[-1] if hasattr(df.index,'dtype') else df.iloc[-1].get('timestamp','?')}")
    
    # 2. Compute target
    clean_df = df.dropna(how='any', subset=['close', 'open']).copy()
    target = compute_targets(clean_df, args.horizon)
    
    # Align feature matrix
    features = filter_features(clean_df)
    
    # Find overlap (target has NaNs from shift)
    overlap_idx = target.index.intersection(features.index)
    X = features.loc[overlap_idx]
    y = target.loc[overlap_idx]
    
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Target balance: {y.mean():.3f} (class 1 ratio)")
    
    # 3. Train/test split
    split = int(len(X) * args.train_ratio)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Train model
    model = xgb.XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.6,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric='logloss',
        verbosity=0,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 5. Backtest on test set (simulate trading)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.6).astype(int)
    
    # Simple backtest metrics
    n_total = len(preds)
    n_signals = preds.sum()
    n_wins = ((preds == 1) & (y_test == 1)).sum()
    n_losses = ((preds == 1) & (y_test == 0)).sum()
    
    wr = n_wins / n_signals * 100 if n_signals > 0 else 0
    
    # Simulate trade returns
    # We need close prices for return calculation
    test_close = clean_df.loc[X_test.index, 'close'].values
    test_open = clean_df.loc[X_test.index, 'open'].values
    future_close = clean_df.loc[X_test.index, 'close'].shift(-args.horizon).values
    entry_price = test_open
    exit_price = np.where(preds == 1, future_close, np.nan)
    
    # P&L for actual trades (where preds=1)
    trade_returns = []
    trade_wins = 0
    trade_losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    
    for i in range(len(preds)):
        if preds[i] == 0:
            continue
        ret = (exit_price[i] - entry_price[i]) / entry_price[i] - 0.0008  # -0.08% fees
        trade_returns.append(ret)
        if ret > 0:
            trade_wins += 1
            gross_profit += ret
        else:
            trade_losses += 1
            gross_loss += abs(ret)
    
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    avg_ret = np.mean(trade_returns) * 100 if trade_returns else 0
    sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(365*24*12/args.horizon) if len(trade_returns) > 3 and np.std(trade_returns) > 0 else 0
    
    # Drawdown
    cum_ret = np.cumprod(1 + np.array(trade_returns)) if trade_returns else np.array([1])
    peak = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - peak) / peak
    max_dd = dd.min() * 100
    
    print(f"\n  {'─'*40}")
    print(f"  📊 BACKTEST RESULTS")
    print(f"  {'─'*40}")
    print(f"  Win Rate:          {wr:.1f}% ({n_wins}/{n_signals})")
    print(f"  Trade WR:          {trade_wins/(trade_wins+trade_losses)*100:.1f}% ({trade_wins}/{trade_wins+trade_losses})")
    print(f"  Profit Factor:     {pf:.2f}")
    print(f"  Total Trades:      {n_signals}")
    print(f"  Avg Return/Trade:  {avg_ret:.3f}%")
    print(f"  Sharpe (ann.):     {sharpe:.1f}")
    print(f"  Max DD:            {max_dd:.2f}%")
    
    # Volatility estimation
    close_prices = clean_df['close'].values
    returns_5m = np.diff(close_prices) / close_prices[:-1]
    daily_vol = np.std(returns_5m) * np.sqrt(288) * 100  # ~288 5m bars per day
    avg_range = (clean_df['high'] - clean_df['low']).mean() / clean_df['close'].mean() * 100
    
    print(f"\n  📊 VOLATILITY")
    print(f"  {'─'*40}")
    print(f"  Daily Vol (std):   {daily_vol:.1f}%")
    print(f"  Avg Candle Range:  {avg_range:.3f}%")
    
    # Summary verdict
    print(f"\n  {'─'*40}")
    verdict = "✅ PROMISING" if (trade_wins/(trade_wins+trade_losses)*100 >= 65 and pf >= 1.8) else \
              "⚠️  MARGINAL" if (trade_wins/(trade_wins+trade_losses)*100 >= 55) else \
              "❌ POOR"
    print(f"  VERDICT: {verdict}")
    print(f"  {'─'*40}\n")
    
    # Return summary dict
    return {
        'symbol': symbol,
        'wr': trade_wins/(trade_wins+trade_losses)*100,
        'pf': pf,
        'n_trades': n_signals,
        'avg_ret_pct': avg_ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'daily_vol': daily_vol,
        'avg_range': avg_range,
        'verdict': verdict,
    }

if __name__ == '__main__':
    result = main()
    # Save result
    out_dir = ROOT / 'data' / 'rapid_eval'
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'{result["symbol"]}.json'
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {out_path}")
