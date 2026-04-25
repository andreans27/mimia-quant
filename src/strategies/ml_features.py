"""
Multi-timeframe feature engineering for ML-driven trading strategies.

Architecture:
  Fetch 5m klines → Resample to 15m, 30m, 1h, 4h → Compute technical features
  independently on EACH timeframe → Align all to 5m index → Cross-TF features.

Target: predict direction 9 5m-candles ahead (= 45 minutes).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json


def resample_to_timeframes(
    df_5m: pd.DataFrame,
    intervals: List[str] = None
) -> Dict[str, pd.DataFrame]:
    """Resample 5m OHLCV data to higher timeframes with proper OHLCV aggregation.
    
    Each resampled DataFrame has REAL candle data (open/high/low/close/volume),
    NOT forward-filled values. This is critical for correct indicator computation.
    
    Args:
        df_5m: 5-minute OHLCV DataFrame with DatetimeIndex
        intervals: Target intervals ['15m', '30m', '1h', '4h']
        
    Returns:
        Dict mapping interval name → OHLCV DataFrame
    """
    if intervals is None:
        intervals = ['15m', '30m', '1h', '4h']
    
    rule_map = {
        '5m': '5min',  # passthrough
        '10m': '10min',
        '15m': '15min',
        '30m': '30min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
    }
    
    result = {'5m': df_5m.copy()}
    
    for name in intervals:
        if name == '5m':
            continue
        rule = rule_map.get(name)
        if rule is None:
            continue
        
        # Proper OHLCV resample — NOT interpolation
        resampled = df_5m.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()
        
        # Ensure OHLC consistency: low <= open, close <= high
        resampled['low'] = resampled[['open', 'close', 'low']].min(axis=1)
        resampled['high'] = resampled[['open', 'close', 'high']].max(axis=1)
        
        result[name] = resampled
    
    return result


def compute_technical_features(df: pd.DataFrame, prefix: str = "", drop_raw: bool = True) -> pd.DataFrame:
    """
    Compute 30+ technical indicators on OHLCV data.
    Returns a DataFrame with the same index plus new feature columns.
    """
    result = df.copy()
    close = result["close"].astype(float)
    high = result["high"].astype(float)
    low = result["low"].astype(float)
    open_p = result["open"].astype(float)
    volume = result["volume"].astype(float)

    p = prefix  # shorthand

    # ── Returns ──
    for period in [1, 2, 3, 5, 8, 13, 21]:
        result[f"{p}ret_{period}"] = close.pct_change(period)
        result[f"{p}log_ret_{period}"] = np.log(close / close.shift(period))

    # ── Volatility ──
    result[f"{p}tr"] = np.maximum(
        high - low,
        np.maximum(
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        )
    )
    for period in [7, 14, 28, 56]:
        result[f"{p}atr_{period}"] = result[f"{p}tr"].rolling(period).mean()
    
    # Bollinger Bands (20,2)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    result[f"{p}bb_upper"] = sma20 + 2 * std20
    result[f"{p}bb_lower"] = sma20 - 2 * std20
    result[f"{p}bb_width"] = (result[f"{p}bb_upper"] - result[f"{p}bb_lower"]) / sma20.replace(0, np.nan)
    result[f"{p}bb_pct"] = (close - result[f"{p}bb_lower"]) / (result[f"{p}bb_upper"] - result[f"{p}bb_lower"]).replace(0, np.nan)

    # ── Momentum ──
    # RSI (14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result[f"{p}rsi_14"] = 100 - (100 / (1 + rs))
    
    # RSI (7)
    avg_gain7 = gain.rolling(7).mean()
    avg_loss7 = loss.rolling(7).mean()
    rs7 = avg_gain7 / avg_loss7.replace(0, np.nan)
    result[f"{p}rsi_7"] = 100 - (100 / (1 + rs7))

    # MACD (12,26,9)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    result[f"{p}macd"] = ema12 - ema26
    result[f"{p}macd_signal"] = result[f"{p}macd"].ewm(span=9).mean()
    result[f"{p}macd_hist"] = result[f"{p}macd"] - result[f"{p}macd_signal"]
    result[f"{p}macd_hist_pct"] = result[f"{p}macd_hist"] / close * 100

    # Stochastic (14,3)
    low14 = low.rolling(14).min()
    high14 = high.rolling(14).max()
    stoch_raw = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
    result[f"{p}stoch_k"] = stoch_raw
    result[f"{p}stoch_d"] = stoch_raw.rolling(3).mean()

    # ── Trend ──
    for period in [10, 20, 50, 100, 200]:
        result[f"{p}sma_{period}"] = close.rolling(period).mean()
        result[f"{p}ema_{period}"] = close.ewm(span=period).mean()
        result[f"{p}dist_sma_{period}"] = ((close - result[f"{p}sma_{period}"]) / result[f"{p}sma_{period}"].replace(0, np.nan)) * 100
        result[f"{p}dist_ema_{period}"] = ((close - result[f"{p}ema_{period}"]) / result[f"{p}ema_{period}"].replace(0, np.nan)) * 100

    # ADX (14)
    atr14 = result[f"{p}atr_14"]
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm_bool = (plus_dm > minus_dm) & (plus_dm > 0)
    minus_dm_bool = (minus_dm > plus_dm) & (minus_dm > 0)
    plus_di = 100 * (plus_dm.where(plus_dm_bool, 0).rolling(14).mean() / atr14.replace(0, np.nan))
    minus_di = 100 * (minus_dm.where(minus_dm_bool, 0).rolling(14).mean() / atr14.replace(0, np.nan))
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    result[f"{p}adx_14"] = dx.rolling(14).mean()
    result[f"{p}di_plus_14"] = plus_di
    result[f"{p}di_minus_14"] = minus_di

    # ── Volume ──
    result[f"{p}volume_sma_20"] = volume.rolling(20).mean()
    result[f"{p}volume_ratio"] = volume / result[f"{p}volume_sma_20"].replace(0, np.nan)
    result[f"{p}vwap"] = (volume * (high + low + close) / 3).rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    
    # OBV
    price_direction = np.sign(close - close.shift(1))
    obv = (price_direction * volume).cumsum()
    result[f"{p}obv"] = obv
    result[f"{p}obv_sma_20"] = obv.rolling(20).mean()
    result[f"{p}obv_ratio"] = obv / result[f"{p}obv_sma_20"].replace(0, np.nan)

    # ── Price action ──
    result[f"{p}hl_pct"] = (high - low) / close * 100
    result[f"{p}body_pct"] = abs(close - open_p) / (high - low).replace(0, np.nan) * 100
    result[f"{p}upper_wick"] = (high - np.maximum(close, open_p)) / (high - low).replace(0, np.nan) * 100
    result[f"{p}lower_wick"] = (np.minimum(close, open_p) - low) / (high - low).replace(0, np.nan) * 100
    result[f"{p}is_green"] = (close > open_p).astype(float)
    result[f"{p}is_red"] = (close < open_p).astype(float)

    # ── Candle patterns ──
    body = abs(close - open_p)
    result[f"{p}doji"] = (body < (high - low) * 0.1).astype(float)
    result[f"{p}marubozu"] = (body > (high - low) * 0.9).astype(float)

    prev_green = open_p.shift(1) < close.shift(1)
    result[f"{p}bullish_engulfing"] = (
        (close > open_p.shift(1)) & 
        (close.shift(1) < open_p.shift(1)) & 
        (close > close.shift(1)) &
        (close < open_p)
    ).astype(float)
    
    result[f"{p}bearish_engulfing"] = (
        (close < open_p.shift(1)) & 
        (close.shift(1) > open_p.shift(1)) & 
        (close < close.shift(1)) &
        (close > open_p)
    ).astype(float)

    # Drop raw OHLCV columns to avoid duplicates across timeframes (if requested)
    if drop_raw:
        for col in ["open", "high", "low", "close", "volume"]:
            if col in result.columns:
                del result[col]

    return result


def compute_5m_features_5tf(
    df_5m: pd.DataFrame,
    target_candle: int = 9,
    intervals: Optional[List[str]] = None,
    for_inference: bool = False
) -> pd.DataFrame:
    """
    Core feature computation: 5m → resample → compute on ALL TFs → combine.
    
    When for_inference=True: skips target column, keeps the most recent rows
    (normally dropped because target requires N-bar forward lookahead).
    
    5 timeframes used: 5m, 15m, 30m, 1h, 4h
    Target: whether close rises in `target_candle` 5m-bars ahead.
    
    Returns:
        DataFrame with ~300+ features aligned to 5m index + 'target' column.
    """
    if intervals is None:
        intervals = ['15m', '30m', '1h', '4h']
    
    # Step 1: Resample 5m to higher timeframes
    tf_data = resample_to_timeframes(df_5m, intervals)
    
    # Step 2: Compute technical features INDEPENDENTLY on each timeframe
    feats_5m = compute_technical_features(tf_data['5m'], prefix='m5_')
    print(f"    Features computed: 5m = {len(feats_5m.columns)}")
    
    other_feats = {}
    for name, df_tf in tf_data.items():
        if name == '5m':
            continue
        prefix_map = {'15m': 'm15_', '30m': 'm30_', '1h': 'h1_', '4h': 'h4_', '2h': 'h2_', '10m': 'm10_'}
        prefix = prefix_map.get(name, f'{name}_')
        other_feats[name] = compute_technical_features(df_tf, prefix=prefix)
        print(f"    Features computed: {name} = {len(other_feats[name].columns)}")
    
    # Step 3: Align all to 5m index via forward-fill
    idx_5m = feats_5m.index
    
    # Resample limits: max bars until next candle of that TF closes
    fill_limits = {'15m': 3, '30m': 6, '1h': 12, '4h': 48, '2h': 24, '10m': 2}
    feat_list = [feats_5m]
    
    for name, df_feat in other_feats.items():
        limit = fill_limits.get(name, 12)
        aligned = df_feat.reindex(idx_5m, method='ffill', limit=limit)
        feat_list.append(aligned)
    
    combined = pd.concat(feat_list, axis=1)
    
    # Step 4: Cross-timeframe features (aligned to 5m)
    close_5m = df_5m['close'].astype(float)
    volume_5m = df_5m['volume'].astype(float)
    
    # RSI divergence: 5m vs 1h
    rsi_1h_aligned = other_feats['1h']['h1_rsi_14'].reindex(idx_5m, method='ffill', limit=12)
    
    xtf = pd.DataFrame(index=idx_5m)
    
    # Trend regime: 1h SMA relationships
    for tf_name in ['15m', '30m', '1h', '4h']:
        if tf_name not in tf_data:
            continue
        tf_close = tf_data[tf_name]['close'].astype(float)
        prefix = {'15m': 'm15_', '30m': 'm30_', '1h': 'h1_', '4h': 'h4_'}[tf_name]
        
        for period in [20, 50]:
            sma = tf_close.rolling(period).mean()
            trend = pd.Series(0, index=tf_close.index)
            trend[sma > sma.shift(1)] = 1  # SMA rising
            trend[sma < sma.shift(1)] = -1  # SMA falling
            
            limit = fill_limits.get(tf_name, 12)
            aligned = trend.reindex(idx_5m, method='ffill', limit=limit)
            xtf[f'{prefix}trend_sma{period}'] = aligned
    
    # Volatility ratio: 5m ATR vs 1h ATR
    atr_5m_aligned = feats_5m['m5_atr_14']
    atr_1h_aligned = other_feats['1h']['h1_atr_14'].reindex(idx_5m, method='ffill', limit=12)
    xtf['vol_ratio_5m_vs_1h'] = atr_5m_aligned / atr_1h_aligned.replace(0, np.nan)
    xtf['vol_ratio_5m_vs_4h'] = atr_5m_aligned / other_feats['4h']['h4_atr_14'].reindex(idx_5m, method='ffill', limit=48).replace(0, np.nan)
    
    # RSI divergence
    rsi_15m_aligned = other_feats['15m']['m15_rsi_14'].reindex(idx_5m, method='ffill', limit=3)
    rsi_1h_aligned = other_feats['1h']['h1_rsi_14'].reindex(idx_5m, method='ffill', limit=12)
    rsi_4h_aligned = other_feats['4h']['h4_rsi_14'].reindex(idx_5m, method='ffill', limit=48)
    
    xtf['rsi_div_5m_vs_15m'] = feats_5m['m5_rsi_14'] - rsi_15m_aligned
    xtf['rsi_div_5m_vs_1h'] = feats_5m['m5_rsi_14'] - rsi_1h_aligned
    xtf['rsi_div_5m_vs_4h'] = feats_5m['m5_rsi_14'] - rsi_4h_aligned
    
    # Volume ratio: 5m volume vs 1h avg volume per 5m slice
    vol_1h_avg = df_5m['volume'].astype(float).rolling(12).sum().resample('1h').mean()
    vol_1h_avg_aligned = vol_1h_avg.reindex(idx_5m, method='ffill', limit=12)
    xtf['vol_5m_vs_1h_avg'] = volume_5m / vol_1h_avg_aligned.replace(0, np.nan)
    
    # Consolidation detection: is 5m range narrow vs 1h range?
    range_5m_20 = (tf_data['5m']['high'].astype(float).rolling(20).max() 
                   - tf_data['5m']['low'].astype(float).rolling(20).min())
    range_1h_20 = (df_5m['high'].astype(float).resample('1h').max().rolling(20).mean()
                   - df_5m['low'].astype(float).resample('1h').min().rolling(20).mean())
    range_1h_20_aligned = range_1h_20.reindex(idx_5m, method='ffill', limit=12)
    xtf['consolidation_ratio'] = range_5m_20 / range_1h_20_aligned.replace(0, np.nan)
    xtf['is_consolidating'] = (range_5m_20 < range_5m_20.rolling(40).mean() * 0.5).astype(float)
    
    # TF alignment: do all TFs agree on direction?
    close_15m = tf_data['15m']['close'].astype(float)
    close_30m = tf_data['30m']['close'].astype(float)
    close_1h = tf_data['1h']['close'].astype(float)
    close_4h = tf_data['4h']['close'].astype(float)
    
    for tf_name, tf_close in [('15m', close_15m), ('30m', close_30m), ('1h', close_1h), ('4h', close_4h)]:
        trend_bull = (tf_close > tf_close.rolling(20).mean()).astype(int)
        limit = fill_limits.get(tf_name, 12)
        xtf[f'{tf_name}_bullish'] = trend_bull.reindex(idx_5m, method='ffill', limit=limit)
    
    # Combined: how many TFs agree?
    xtf['tf_bullish_count'] = (
        xtf['15m_bullish'] + xtf['30m_bullish'] + xtf['1h_bullish'] + xtf['4h_bullish']
    )
    
    # ── Combine ALL features ──
    combined = pd.concat([combined, xtf], axis=1)
    
    # ── Target: 9 5m-candles ahead (= 45 minutes) ──
    if not for_inference:
        combined['target'] = (close_5m.shift(-target_candle) > close_5m).astype(float)
    
    # ── Drop NaN rows (need ~200 bars for warmup due to 5m lookback) ──
    if for_inference:
        # Keep ALL rows including the most recent ones (no forward-looking target needed)
        combined = combined.iloc[200:].copy()
        print(f"    Total features (inference): {len(combined.columns)}")
    else:
        combined = combined.iloc[200:-target_candle].copy()
        print(f"    Total features: {len([c for c in combined.columns if c != 'target'])} + target")
    print(f"    Rows: {len(combined)}")
    
    return combined


def prepare_ml_dataset(
    symbol: str,
    days: int = 120,
    cache_dir: str = "data/ml_cache",
    target_candle: int = 9,
    intervals: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Fetch 5m klines, compute features across 5 timeframes, return X, y.
    
    Architecture:
      5m klines → resample to 15m/30m/1h/4h → compute features independently
      on each TF → align all to 5m index → target = direction in 9 bars.
      
    Args:
        symbol: Trading pair e.g. 'BTCUSDT'
        days: How many days of historical data to fetch
        cache_dir: Cache directory
        target_candle: Number of 5m candles forward to predict (default 9 = 45min)
        intervals: Timeframes to resample and compute features on
    
    Returns:
        (X, y, index) or (None, None, None) on failure
    """
    if intervals is None:
        intervals = ['15m', '30m', '1h', '4h']
    
    from src.utils.binance_client import BinanceRESTClient
    from datetime import datetime, timedelta
    
    # Cache key: differentiate from old 15m-based cache
    tf_suffix = '_'.join(sorted(intervals))
    cache_path = Path(cache_dir) / f"{symbol}_5m_{days}d_{target_candle}c_{tf_suffix}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"  Loading cached features for {symbol} (5m base)...")
        df = pd.read_parquet(cache_path)
        feature_cols = [c for c in df.columns if c != "target"]
        print(f"    {len(df)} rows, {len(feature_cols)} features")
        return df[feature_cols], df["target"], df.index

    print(f"  Fetching {days} days of 5m data for {symbol}...")
    client = BinanceRESTClient(testnet=True)
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    start_ms = int(start_time.timestamp() * 1000)
    end_ms = int(end_time.timestamp() * 1000)
    
    # Fetch 5m klines
    df_5m = _fetch_all_klines(client, symbol, "5m", start_ms, end_ms)
    
    if df_5m is None or len(df_5m) < 1000:
        print(f"  ⚠️ Insufficient 5m data for {symbol}")
        return None, None, None
    
    print(f"    {symbol}: {len(df_5m)} bars of 5m data")
    
    # Compute features (this does resampling internally)
    print(f"  Computing features across {len(intervals)} timeframes for {symbol}...")
    combined = compute_5m_features_5tf(df_5m, target_candle=target_candle, intervals=intervals)
    
    # Save to cache
    combined.to_parquet(cache_path)
    print(f"  Saved to {cache_path}")
    
    feature_cols = [c for c in combined.columns if c != "target"]
    return combined[feature_cols], combined["target"], combined.index


def _fetch_all_klines(client, symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Fetch all klines with pagination for a given interval and time range."""
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_volume', 'trades', 'taker_buy_base',
               'taker_buy_quote', 'ignore']
    all_bars = []
    current_start = start_ms
    
    max_per_request = 1000
    interval_ms_map = {
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000
    }
    interval_ms = interval_ms_map.get(interval, 15 * 60 * 1000)
    
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
                last_time = last[0]
            else:
                try:
                    last_time = last.open_time if hasattr(last, 'open_time') else last['open_time']
                except:
                    break
            current_start = int(last_time) + 1
            if len(batch) < max_per_request:
                break
        except Exception as e:
            print(f"    ⚠️ API error fetching {interval}: {e}")
            break

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars, columns=columns)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['open', 'high', 'low', 'close', 'volume']]
