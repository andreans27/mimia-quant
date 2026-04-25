"""
Pure signal-based ML backtest — no SL/TP.
Entry on model signal → hold exactly `hold_bars` candles → exit.
This faithfully tests what the model was trained to predict (3-candle forward return).
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")


def load_model(symbol: str, confidence_threshold: float = 0.50) -> dict:
    """Load model + features, return dict with everything needed."""
    cache_path = CACHE_DIR / f"{symbol}_features_120d_3c.parquet"
    if not cache_path.exists():
        alt = CACHE_DIR / f"{symbol}_features_60d_3c.parquet"
        if alt.exists():
            cache_path = alt
        else:
            return None
    
    feat_df = pd.read_parquet(cache_path)
    
    model_path = MODEL_DIR / f"{symbol}_xgb.json"
    meta_path = MODEL_DIR / f"{symbol}_xgb_meta.json"
    if not model_path.exists() or not meta_path.exists():
        return None
    
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    return {
        "features": feat_df,
        "model": model,
        "meta": meta,
    }


def run_ml_signal_backtest(symbol: str,
                           confidence_threshold: float = 0.55,
                           hold_bars: int = 3,
                           cooldown_bars: int = 2,
                           initial_capital: float = 5000.0,
                           commission_rate: float = 0.0004,
                           slippage_rate: float = 0.0005,
                           max_position_pct: float = 0.10,
                           warmup_bars: int = 120) -> dict:
    """
    Pure signal-based backtest:
    - Predict on each 15m bar using pre-trained XGBoost
    - If proba >= threshold → enter long, hold for `hold_bars` candles
    - If proba <= 1-threshold → enter short, hold for `hold_bars` candles
    - Exit after exact hold period → record PnL
    - No SL, no TP — just model signal + fixed holding period
    """
    from src.strategies.backtester import BacktestTrade, TradeDirection
    from scripts.run_full_backtest import compute_metrics, check_criteria
    
    cache = load_model(symbol, confidence_threshold)
    if cache is None:
        print(f"    ❌ No model/data for {symbol}")
        return None
    
    feat_df = cache["features"]
    model = cache["model"]
    meta = cache["meta"]
    
    feature_list = meta.get('features', [])
    # Split into train/test as the original training did
    total = len(feat_df)
    split = int(total * 0.85)  # ~85/15 split used in training
    
    # Restrict to test portion for OOS backtest
    test_df = feat_df.iloc[split:].copy()
    if len(test_df) < warmup_bars + hold_bars + 10:
        test_df = feat_df.iloc[max(0, split - warmup_bars):].copy()  # include some warmup
    
    # Predict probabilities
    X = test_df[feature_list].fillna(0).clip(-10, 10)
    probas = model.predict_proba(X)[:, 1]
    
    # Build backtest DataFrame
    df_bt = pd.DataFrame(index=test_df.index)
    df_bt['close'] = None  # We'll get close from... 
    # Need OHLCV data. The features df has raw OHLCV stripped.
    # We need to fetch the 15m data separately.
    
    # STALE — this approach needs restructuring
    # Let me first fetch 15m data
    return None


def run_pure_signal_backtest_15m(symbol: str,
                                 confidence_threshold: float = 0.55,
                                 hold_bars: int = 3,
                                 cooldown_bars: int = 2,
                                 initial_capital: float = 5000.0,
                                 commission_rate: float = 0.0004,
                                 slippage_rate: float = 0.0005,
                                 max_position_pct: float = 0.10,
                                 days_data: int = 90,
                                 warmup_bars: int = 120) -> dict:
    """
    Pure signal ML backtest on 15m data:
    1. Fetch 15m OHLCV
    2. Load model + cached features (for predictions)
    3. Align predictions to OHLCV by index
    4. When model signal fires → enter at close, hold for exactly hold_bars → exit
    5. No stop loss, no take profit
    """
    from src.strategies.backtester import BacktestTrade, TradeDirection
    from scripts.run_full_backtest import compute_metrics, check_criteria
    from src.utils.binance_client import BinanceRESTClient
    
    # --- 1. Fetch 15m OHLCV ---
    client = BinanceRESTClient(testnet=True)
    end = datetime.now()
    start = end - timedelta(days=days_data)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    limit = 1000
    all_bars = []
    last_ts = start_ms
    while last_ts < end_ms:
        raw = client.get_klines(symbol, "15m", limit=limit,
                                start_time=last_ts, end_time=end_ms)
        if raw is None or len(raw) == 0:
            break
        all_bars.extend(raw)
        last_ts = raw[-1][0] + 1
        if len(raw) < limit:
            break
    
    df = pd.DataFrame(all_bars, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']]
    
    if len(df) < warmup_bars + hold_bars + 10:
        print(f"    ❌ Insufficient 15m data: {len(df)} bars")
        return None
    
    # --- 2. Load model + predictions ---
    cache = load_model(symbol, confidence_threshold)
    if cache is None:
        return None
    
    feat_df = cache["features"]
    model = cache["model"]
    meta = cache["meta"]
    
    feature_list = meta.get('features', [])
    
    # Align: use only rows where both OHLCV and features exist
    # feat_df has 15m timestamps, so does df
    aligned_df = df.join(feat_df[feature_list], how='inner')
    
    if len(aligned_df) < warmup_bars + hold_bars + 10:
        print(f"    ❌ Too few aligned bars: {len(aligned_df)}")
        return None
    
    # --- 3. Predict ---
    X = aligned_df[feature_list].fillna(0).clip(-10, 10)
    probas = model.predict_proba(X)[:, 1]
    aligned_df['proba'] = probas
    aligned_df['signal'] = 0
    aligned_df.loc[aligned_df['proba'] >= confidence_threshold, 'signal'] = 1
    aligned_df.loc[aligned_df['proba'] <= (1 - confidence_threshold), 'signal'] = -1
    
    # --- 4. Simulate ---
    aligned_df = aligned_df.iloc[warmup_bars:].copy()
    
    capital = initial_capital
    trades = []
    equity_curve = [capital]
    cooldown = 0
    hold_remaining = 0
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    entry_idx = 0
    long_pnl = 0.0
    short_pnl = 0.0
    
    for idx in range(len(aligned_df)):
        row = aligned_df.iloc[idx]
        price = float(row['close'])
        ts = aligned_df.index[idx]
        
        if cooldown > 0:
            cooldown -= 1
        
        if position != 0:
            hold_remaining -= 1
            # Exit when hold period expires
            if hold_remaining <= 0:
                exit_price = price * (1 - slippage_rate) if position == 1 else price * (1 + slippage_rate)
                if position == 1:
                    raw_pnl = entry_qty * (exit_price - entry_price)
                else:
                    raw_pnl = entry_qty * (entry_price - exit_price)
                
                comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
                pnl_net = raw_pnl - comm
                pnl_pct = raw_pnl / (entry_qty * entry_price) * 100 if entry_qty > 0 else 0
                
                capital += raw_pnl
                
                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net
                
                trades.append(BacktestTrade(
                    entry_time=aligned_df.index[entry_idx],
                    exit_time=ts,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=entry_qty,
                    side=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                    pnl=pnl_net,
                    pnl_pct=pnl_pct,
                    commission=comm,
                    slippage=price * slippage_rate * entry_qty,
                    strategy_name="ml_signal",
                    signal_id=f"exit_{idx}",
                ))
                position = 0
                cooldown = cooldown_bars
        
        # Entry logic
        if position == 0 and cooldown == 0:
            sig = int(row.get('signal', 0))
            if sig == 1:  # Enter long
                pos_val = capital * max_position_pct
                entry_price = price * (1 + slippage_rate)
                entry_qty = pos_val / entry_price
                position = 1
                entry_idx = idx
                hold_remaining = hold_bars
                cooldown = cooldown_bars
            elif sig == -1:  # Enter short
                pos_val = capital * max_position_pct
                entry_price = price * (1 - slippage_rate)
                entry_qty = pos_val / entry_price
                position = -1
                entry_idx = idx
                hold_remaining = hold_bars
                cooldown = cooldown_bars
        
        # Equity tracking
        equity_curve.append(capital)
    
    if not trades:
        print(f"    ❌ No trades for {symbol}")
        return None
    
    # Compute metrics
    start_date = aligned_df.index[0]
    end_date = aligned_df.index[-1]
    computed = compute_metrics(trades, equity_curve, start_date, end_date)
    computed['long_pnl'] = long_pnl
    computed['short_pnl'] = short_pnl
    passed, criteria = check_criteria(computed)
    
    return {
        "symbol": symbol,
        "strategy": "ml_signal",
        "metrics": computed,
        "passed": passed,
        "criteria": criteria,
        "trade_count": len(trades),
        "trades": trades,
        "equity_curve": equity_curve,
        "start_date": start_date,
        "end_date": end_date,
    }


if __name__ == "__main__":
    # Test single symbol
    result = run_pure_signal_backtest_15m("BTCUSDT", confidence_threshold=0.55, hold_bars=3)
    if result:
        m = result['metrics']
        print(f"Symbol: {result['symbol']}")
        print(f"  Trades: {result['trade_count']}")
        print(f"  Win Rate: {m['win_rate']:.1f}%")
        print(f"  Profit Factor: {m['profit_factor']:.2f}")
        print(f"  Max DD: {m['max_drawdown']:.2f}%")
        print(f"  Sharpe: {m['sharpe']:.2f}")
        print(f"  Total PnL: ${m['total_pnl']:.2f}")
        print(f"  Long PnL: ${m['long_pnl']:.2f}")
        print(f"  Short PnL: ${m['short_pnl']:.2f}")
        print(f"  Passed: {'✅' if result['passed'] else '❌'}")
    else:
        print("❌ No results")
