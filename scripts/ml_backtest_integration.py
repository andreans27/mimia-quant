"""
ML backtest — runs directly on 15m data using pre-trained XGBoost model predictions.
Uses the cached feature DataFrame's index for alignment.
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


def load_signals_15m(symbol: str, confidence_threshold: float = 0.52) -> pd.DataFrame:
    """
    Load cached features + model → pre-compute 15m predictions.
    Returns DataFrame with index=15m_timestamp, columns=['proba','signal_long','signal_short'].
    """
    cache_path = CACHE_DIR / f"{symbol}_features_120d_3c.parquet"
    if not cache_path.exists():
        alt = CACHE_DIR / f"{symbol}_features_60d_3c.parquet"
        if alt.exists():
            cache_path = alt
        else:
            print(f"    No cached features for {symbol}")
            return None
    
    feat_df = pd.read_parquet(cache_path)
    
    model_path = MODEL_DIR / f"{symbol}_xgb.json"
    meta_path = MODEL_DIR / f"{symbol}_xgb_meta.json"
    if not model_path.exists() or not meta_path.exists():
        print(f"    No trained model for {symbol}")
        return None
    
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    
    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    
    feature_list = meta.get('features', [])
    X = feat_df[feature_list].fillna(0).clip(-10, 10)
    probas = model.predict_proba(X)[:, 1]
    
    sig = pd.DataFrame(index=feat_df.index)
    sig['proba'] = probas
    sig['signal_long'] = (probas >= confidence_threshold).astype(int)
    sig['signal_short'] = (probas <= (1 - confidence_threshold)).astype(int)
    sig['signal'] = 0
    sig.loc[sig['signal_long'] == 1, 'signal'] = 1
    sig.loc[sig['signal_short'] == 1, 'signal'] = -1
    
    return sig


def fetch_15m_ohlcv(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch 15m OHLCV data for backtesting."""
    from src.utils.binance_client import BinanceRESTClient
    
    client = BinanceRESTClient(testnet=True)
    end = datetime.now()
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    interval = "15m"
    limit = 1000
    
    all_bars = []
    last_ts = start_ms
    
    while last_ts < end_ms:
        raw = client.get_klines(symbol, interval, limit=limit,
                                start_time=last_ts,
                                end_time=end_ms)
        if raw is None or len(raw) == 0:
            break
        all_bars.extend(raw)
        last_ts = raw[-1][0] + 1
        if len(raw) < limit:
            break
    
    df = pd.DataFrame(all_bars, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base',
        'taker_buy_quote','ignore'
    ])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']]
    return df


def run_ml_backtest_15m(symbol: str,
                         confidence_threshold: float = 0.52,
                         cooldown_candles: int = 4,
                         stop_loss_pct: float = 1.2,
                         take_profit_pct: float = 2.0,
                         initial_capital: float = 5000.0,
                         commission_rate: float = 0.0004,
                         slippage_rate: float = 0.0005,
                         max_position_pct: float = 0.08,
                         warmup_bars: int = 100) -> dict:
    """
    Run ML backtest on 15m data using pre-trained model predictions.
    Returns metrics dict or None.
    """
    from src.strategies.backtester import BacktestTrade, TradeDirection
    from scripts.run_full_backtest import compute_metrics, check_criteria
    
    # 1. Fetch 15m OHLCV data
    df_15m = fetch_15m_ohlcv(symbol, days=90)
    if df_15m is None or len(df_15m) < warmup_bars + 10:
        print(f"    ❌ Not enough 15m data for {symbol}")
        return None
    
    # 2. Load pre-computed model predictions on the same 15m timestamps
    signals = load_signals_15m(symbol, confidence_threshold)
    if signals is None:
        return None
    
    # 3. Align: signals.index is 15m, df_15m.index is 15m → merge on index
    df_bt = df_15m.join(signals[['proba','signal_long','signal_short','signal']], how='left')
    
    # Remove rows before the first signal (model warmup)
    first_signal_idx = signals.index[0]
    df_bt = df_bt[df_bt.index >= first_signal_idx].copy()
    
    if len(df_bt) < warmup_bars + 10:
        print(f"    ❌ Too few aligned bars for {symbol}: {len(df_bt)}")
        return None
    
    # Skip warmup
    df_bt = df_bt.iloc[warmup_bars:].copy()
    
    # Forward-fill signal NaNs (bar with no precomputed signal = no trade)
    for col in ['signal_long','signal_short','signal']:
        df_bt[col] = df_bt[col].fillna(0).astype(int)
    df_bt['proba'] = df_bt['proba'].fillna(0.5)
    
    # 4. Run simulation
    capital = initial_capital
    position = 0
    entry_price = 0.0
    entry_qty = 0.0
    entry_idx = 0
    trades = []
    equity_curve = [capital]
    cooldown = 0
    
    for idx in range(1, len(df_bt)):
        row = df_bt.iloc[idx]
        price = float(row['close'])
        ts = df_bt.index[idx]
        
        if cooldown > 0:
            cooldown -= 1
        
        # Open position management
        if position != 0:
            if position == 1:
                raw_pnl = entry_qty * (price - entry_price)
            else:
                raw_pnl = entry_qty * (entry_price - price)
            
            pnl_pct = raw_pnl / (entry_qty * entry_price) * 100 if entry_qty > 0 else 0
            
            # Stop-loss hit
            if pnl_pct <= -stop_loss_pct:
                exit_price = price * (1 - slippage_rate) if position == 1 else price * (1 + slippage_rate)
                exit_pnl = entry_qty * (exit_price - entry_price) if position == 1 else entry_qty * (entry_price - exit_price)
                comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
                pnl_net = exit_pnl - comm
                capital += exit_pnl
                
                trades.append(BacktestTrade(
                    entry_time=df_bt.index[entry_idx], exit_time=ts,
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=entry_qty,
                    side=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                    pnl=pnl_net, pnl_pct=pnl_pct,
                    commission=comm,
                    slippage=price * slippage_rate * entry_qty,
                    strategy_name="ml_backtest", signal_id=f"sl_{idx}"))
                position = 0
                cooldown = cooldown_candles
                continue
            
            # Take-profit hit
            if pnl_pct >= take_profit_pct:
                exit_price = price * (1 - slippage_rate) if position == 1 else price * (1 + slippage_rate)
                exit_pnl = entry_qty * (exit_price - entry_price) if position == 1 else entry_qty * (entry_price - exit_price)
                comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
                pnl_net = exit_pnl - comm
                capital += exit_pnl
                
                trades.append(BacktestTrade(
                    entry_time=df_bt.index[entry_idx], exit_time=ts,
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=entry_qty,
                    side=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                    pnl=pnl_net, pnl_pct=pnl_pct,
                    commission=comm,
                    slippage=price * slippage_rate * entry_qty,
                    strategy_name="ml_backtest", signal_id=f"tp_{idx}"))
                position = 0
                cooldown = cooldown_candles
                continue
        
        # Entry logic
        if position == 0 and cooldown == 0:
            sig = int(row.get('signal', 0))
            if sig == 1:  # Long
                pos_val = capital * max_position_pct
                entry_price = price * (1 + slippage_rate)
                entry_qty = pos_val / entry_price
                position = 1
                entry_idx = idx
                cooldown = cooldown_candles
            elif sig == -1:  # Short
                pos_val = capital * max_position_pct
                entry_price = price * (1 - slippage_rate)
                entry_qty = pos_val / entry_price
                position = -1
                entry_idx = idx
                cooldown = cooldown_candles
        
        # Equity tracking
        equity = capital
        if position == 1:
            equity = capital + max(0, entry_qty * (price - entry_price))
        elif position == -1:
            equity = capital + max(0, entry_qty * (entry_price - price))
        equity_curve.append(equity)
    
    # Close any remaining position
    if position != 0:
        last_price = float(df_bt.iloc[-1]['close'])
        exit_price = last_price * (1 - slippage_rate) if position == 1 else last_price * (1 + slippage_rate)
        exit_pnl = entry_qty * (exit_price - entry_price) if position == 1 else entry_qty * (entry_price - exit_price)
        comm = (entry_qty * entry_price + entry_qty * exit_price) * commission_rate
        pnl_net = exit_pnl - comm
        capital += exit_pnl
        pnl_pct = exit_pnl / (entry_qty * entry_price) * 100 if entry_qty > 0 else 0
        
        trades.append(BacktestTrade(
            entry_time=df_bt.index[entry_idx], exit_time=df_bt.index[-1],
            entry_price=entry_price, exit_price=exit_price,
            quantity=entry_qty,
            side=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
            pnl=pnl_net, pnl_pct=pnl_pct,
            commission=comm,
            slippage=last_price * slippage_rate * entry_qty,
            strategy_name="ml_backtest", signal_id="close_eod"))
        position = 0
    
    if not trades:
        return None
    
    # Compute metrics
    start_date = df_bt.index[0]
    end_date = df_bt.index[-1]
    computed = compute_metrics(trades, equity_curve, start_date, end_date)
    passed, criteria = check_criteria(computed)
    
    return {
        "symbol": symbol, "strategy": "ml_backtest",
        "metrics": computed, "passed": passed, "criteria": criteria,
        "trade_count": len(trades),
        "equity_curve": equity_curve,
        "start_date": start_date, "end_date": end_date,
    }
