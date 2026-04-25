"""
Ensemble ML backtest — 5m base, 5 models voting, pure signal (hold N bars).
Loads cached features from the 5m multi-TF pipeline, aligns with OHLCV data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb

CACHE_DIR = Path("data/ml_cache")
MODEL_DIR = Path("data/ml_models")


def load_ensemble(symbol: str, intervals=None) -> dict:
    """Load the 5-model ensemble + features from 5m cache."""
    if intervals is None:
        intervals = ['15m', '30m', '1h', '4h']
    
    meta_path = MODEL_DIR / f"{symbol}_ensemble_meta.json"
    if not meta_path.exists():
        print(f"    ❌ No ensemble meta for {symbol}")
        return None
    
    with open(meta_path) as f:
        meta = json.load(f)
    
    # Try 5m cache paths
    tf_suffix = '_'.join(sorted(intervals))
    cache_candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*_{tf_suffix}.parquet"))
    
    if not cache_candidates:
        # Fallback: look for any 5m cache for this symbol
        cache_candidates = list(CACHE_DIR.glob(f"{symbol}_5m_*.parquet"))
    
    if not cache_candidates:
        print(f"    ❌ No 5m cache found for {symbol}")
        return None
    
    # Use most recent cache
    cache_path = max(cache_candidates, key=lambda p: p.stat().st_mtime)
    
    feat_df = pd.read_parquet(cache_path)
    print(f"    Cache: {len(feat_df)} rows @5m")
    
    # Load all 5 models
    models = []
    for seed_str in [str(s) for s in meta.get('seeds', [42, 101, 202, 303, 404])]:
        path = MODEL_DIR / f"{symbol}_xgb_ens_{seed_str}.json"
        if not path.exists():
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(path))
        # Get features this model was trained on
        mf = meta.get('model_features', {}).get(seed_str, meta.get('features', []))
        models.append((seed_str, m, mf))
    
    if len(models) < 3:
        print(f"    ❌ Only {len(models)} models loaded")
        return None
    
    return {
        "models": models,
        "features": feat_df,
        "meta": meta,
        "full_feature_set": meta.get('features', []),
    }


def load_ensemble_signals(symbol: str, confidence_threshold: float = 0.55,
                          intervals=None) -> pd.DataFrame:
    """Load 5-model ensemble → average probabilities → signal DataFrame."""
    if intervals is None:
        intervals = ['15m', '30m', '1h', '4h']
    
    ens = load_ensemble(symbol, intervals)
    if ens is None:
        return None
    
    feat_df = ens['features']
    models = ens['models']
    full_features = ens['full_feature_set']
    
    # Compute probabilities from each model
    all_probs = []
    for seed, model, model_features in models:
        if model_features is None or len(model_features) == 0:
            model_features = full_features
        try:
            # Intersect with available columns
            available = [c for c in model_features if c in feat_df.columns]
            if len(available) < 10:
                continue
            X = feat_df[available].fillna(0).clip(-10, 10)
            probs = model.predict_proba(X)[:, 1]
            all_probs.append(probs)
        except Exception as e:
            print(f"    ⚠️ Model {seed} failed: {e}")
    
    if not all_probs:
        return None
    
    # Average probabilities
    avg_probs = np.nanmean(all_probs, axis=0)
    
    sig = pd.DataFrame(index=feat_df.index)
    sig['proba'] = avg_probs
    sig['signal_long'] = (avg_probs >= confidence_threshold).astype(int)
    sig['signal_short'] = (avg_probs <= (1 - confidence_threshold)).astype(int)
    sig['signal'] = 0
    sig.loc[sig['signal_long'] == 1, 'signal'] = 1
    sig.loc[sig['signal_short'] == 1, 'signal'] = -1
    
    return sig


def run_ensemble_backtest(symbol: str,
                          confidence_threshold: float = 0.55,
                          hold_bars: int = 9,  # 9 × 5m = 45 min
                          cooldown_bars: int = 3,
                          initial_capital: float = 5000.0,
                          commission_rate: float = 0.0004,
                          slippage_rate: float = 0.0005,
                          max_position_pct: float = 0.10,
                          days_data: int = 90,
                          warmup_bars: int = 200) -> dict:
    """Run ensemble backtest on 5m data with pure signal approach."""
    from src.strategies.backtester import BacktestTrade, TradeDirection
    from scripts.backtesting.run_full_backtest import compute_metrics, check_criteria
    from src.utils.binance_client import BinanceRESTClient
    
    # --- 1. Fetch 5m OHLCV ---
    client = BinanceRESTClient(testnet=True)
    end = datetime.now()
    start = end - timedelta(days=days_data)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    
    limit = 1000
    all_bars = []
    last_ts = start_ms
    while last_ts < end_ms:
        raw = client.get_klines(symbol, "5m", limit=limit,
                                start_time=last_ts, end_time=end_ms)
        if raw is None or len(raw) == 0:
            break
        all_bars.extend(raw)
        last_ts = raw[-1][0] + 1
        if len(raw) < limit:
            break
    
    if len(all_bars) < 1000:
        print(f"    ❌ Insufficient 5m data: {len(all_bars)} bars")
        return None
    
    df = pd.DataFrame(all_bars, columns=[
        'open_time','open','high','low','close','volume',
        'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
    ])
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[['open','high','low','close','volume']]
    
    print(f"    OHLCV: {len(df)} 5m bars")
    
    if len(df) < warmup_bars + hold_bars + 10:
        print(f"    ❌ Too few 5m bars: {len(df)}")
        return None
    
    # --- 2. Load ensemble predictions ---
    signals = load_ensemble_signals(symbol, confidence_threshold)
    if signals is None:
        print(f"    ❌ No ensemble signals for {symbol}")
        return None
    
    # --- 3. Align signals to OHLCV by index ---
    # The cache starts later than raw OHLCV; join on common index
    df_bt = df.join(signals[['proba','signal_long','signal_short','signal']], how='inner')
    
    start_sig = signals.index[0]
    df_bt = df_bt[df_bt.index >= start_sig].copy()
    
    if len(df_bt) < warmup_bars + hold_bars + 10:
        print(f"    ❌ Too few aligned bars: {len(df_bt)}")
        return None
    
    # Skip warmup bars
    df_bt = df_bt.iloc[warmup_bars:].copy()
    
    # Fill signal NaNs
    for col in ['signal_long','signal_short','signal']:
        df_bt[col] = df_bt[col].fillna(0).astype(int)
    df_bt['proba'] = df_bt['proba'].fillna(0.5)
    
    # --- 4. Simulate ---
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
    
    for idx in range(len(df_bt)):
        row = df_bt.iloc[idx]
        price = float(row['close'])
        
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
                pnl_pct_val = ((exit_price - entry_price) / entry_price * 100) if position == 1 else ((entry_price - exit_price) / entry_price * 100)
                capital += raw_pnl
                
                if position == 1:
                    long_pnl += pnl_net
                else:
                    short_pnl += pnl_net
                
                trades.append(BacktestTrade(
                    entry_time=df_bt.index[entry_idx], exit_time=df_bt.index[idx],
                    entry_price=entry_price, exit_price=exit_price,
                    quantity=entry_qty,
                    side=TradeDirection.LONG if position == 1 else TradeDirection.SHORT,
                    pnl=pnl_net,
                    commission=comm,
                    slippage=price * slippage_rate * entry_qty,
                    strategy_name='ml_ensemble_5m', signal_id=f'exit_{idx}',
                    pnl_pct=pnl_pct_val))
                position = 0
                cooldown = cooldown_bars
        
        # Entry
        if position == 0 and cooldown == 0:
            sig = int(row.get('signal', 0))
            if sig == 1:
                pos_val = capital * max_position_pct
                entry_price = price * (1 + slippage_rate)
                entry_qty = pos_val / entry_price
                position = 1
                entry_idx = idx
                hold_remaining = hold_bars
                cooldown = cooldown_bars
            elif sig == -1:
                pos_val = capital * max_position_pct
                entry_price = price * (1 - slippage_rate)
                entry_qty = pos_val / entry_price
                position = -1
                entry_idx = idx
                hold_remaining = hold_bars
                cooldown = cooldown_bars
        
        equity_curve.append(capital)
    
    if not trades:
        print(f"    ❌ No trades for {symbol}")
        return None
    
    start_date = df_bt.index[0]
    end_date = df_bt.index[-1]
    computed = compute_metrics(trades, equity_curve, start_date, end_date)
    computed['long_pnl'] = long_pnl
    computed['short_pnl'] = short_pnl
    passed, criteria = check_criteria(computed)
    
    return {
        "symbol": symbol, "strategy": "ml_ensemble_5m",
        "metrics": computed, "passed": passed, "criteria": criteria,
        "trade_count": len(trades),
        "equity_curve": equity_curve,
        "start_date": start_date, "end_date": end_date,
    }
