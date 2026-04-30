"""
Multi-timeframe feature engineering for ML-driven trading strategies.

Architecture:
  Fetch 5m klines → Resample to 15m, 30m, 1h, 4h → Compute technical features
  independently on EACH timeframe → Align all to 5m index → Cross-TF features.

Target: predict direction 9 5m-candles ahead (= 45 minutes).

OHLCV Data Cache:
  All processes (train, retrain, live) share one OHLCV cache at
  data/ohlcv_cache/{symbol}_5m.parquet. This ensures consistent data
  across training and inference.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import os
import time
import json
from datetime import datetime, timedelta

# ── Shared OHLCV Cache (single source of truth for ALL processes) ──────
OHLCV_CACHE_DIR = Path("data/ohlcv_cache")
OHLCV_FETCH_DAYS = 120  # Days of 5m data to fetch on first run

# ── File-based write lock to prevent race conditions ──
# Multiple daemon instances can write to the same parquet file simultaneously,
# causing corruption. Lock files are per-symbol and auto-expire after 30s.
CACHE_LOCK_TIMEOUT = 30  # seconds

# ── 1h OHLCV Cache (independent from 5m — NO look-ahead bias) ──
# 1h klines are fetched DIRECTLY from Binance (not resampled from 5m).
# Each 1h bar is a complete historical entity — safe for training & inference.
OHLCV_CACHE_1H_DAYS = 120


def ensure_ohlcv_1h(symbol: str, min_days: int = 120) -> Optional[pd.DataFrame]:
    """Fetch 1h OHLCV DIRECTLY from Binance (NOT resampled from 5m).
    Each 1h bar is a complete historical entity — eliminates look-ahead bias.
    Uses the SAME public Futures API as 5m cache.
    """
    cache_path = OHLCV_CACHE_DIR / f"{symbol}_1h.parquet"

    # Check cache
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        min_bars = min_days * 24
        if len(df) >= min_bars:
            return df
        print(f"  1h Cache has {len(df)} bars (need {min_bars}), refreshing...")

    # Fetch from Binance public Futures API
    import requests
    end = datetime.now()
    start = end - timedelta(days=min_days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    limit = 1000
    all_bars = []
    last_ts = start_ms
    url = "https://fapi.binance.com/fapi/v1/klines"

    while last_ts < end_ms:
        params = {
            'symbol': symbol,
            'interval': '1h',
            'limit': limit,
            'startTime': last_ts,
            'endTime': end_ms,
        }
        try:
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
        except Exception:
            break

    if len(all_bars) < 200:
        return None

    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_quote', 'quote_volume', 'trades']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    
    # Keep only OHLCV + taker columns
    keep = ['open', 'high', 'low', 'close', 'volume']
    available = [c for c in keep if c in df.columns]
    result = df[available]

    # Cache it
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    print(f"  💾 Cached {len(result)} 1h bars → {cache_path}")
    return result


def _acquire_cache_lock(symbol: str) -> bool:
    """Try to acquire exclusive write lock for a symbol's cache file.
    Returns True if lock acquired, False if already locked by another process.
    Locks auto-expire after CACHE_LOCK_TIMEOUT seconds (stale lock recovery)."""
    lock_path = OHLCV_CACHE_DIR / f".{symbol}_5m.lock"
    OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Try exclusive creation — fails if file exists
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return True
    except FileExistsError:
        # Check if lock is stale
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > CACHE_LOCK_TIMEOUT:
                lock_path.unlink(missing_ok=True)
                # Retry once
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(os.getpid()).encode())
                    os.close(fd)
                    print(f"    ⚠️ Stale lock recovered for {symbol}")
                    return True
                except FileExistsError:
                    return False
            return False
        except FileNotFoundError:
            return False
    except Exception:
        return False


def _release_cache_lock(symbol: str):
    """Release write lock for a symbol's cache file."""
    lock_path = OHLCV_CACHE_DIR / f".{symbol}_5m.lock"
    try:
        # Only delete if this process owns the lock
        if lock_path.exists():
            try:
                pid = int(lock_path.read_text().strip())
                if pid == os.getpid():
                    lock_path.unlink(missing_ok=True)
            except (ValueError, IOError):
                lock_path.unlink(missing_ok=True)
    except Exception:
        pass


def ensure_ohlcv_data(symbol: str, min_days: int = 120) -> Optional[pd.DataFrame]:
    """Single source of truth for 5m OHLCV data across ALL processes.
    Uses the SAME public Futures API as signals.py (fapi.binance.com),
    NOT testnet — so live trader, training, and retraining all see identical data.

    Args:
        symbol: Trading symbol (e.g. 'ETHUSDT')
        min_days: Minimum days of data to keep in cache

    Returns:
        DataFrame with DatetimeIndex and OHLCV columns, or None if unavailable
    """
    cache_path = OHLCV_CACHE_DIR / f"{symbol}_5m.parquet"

    # Check cache first — return immediately if sufficient
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        min_bars = min_days * 288
        if len(df) >= min_bars:
            return df
        print(f"  Cache has {len(df)} bars (need {min_bars}), refreshing...")

    # Fetch from Binance PUBLIC Futures API (same source as live trader)
    # NOT from testnet — critical for data consistency across all processes
    df = _fetch_5m_public_klines(symbol, days=min_days)

    # Save to shared cache (with write lock to prevent race conditions)
    if df is not None and len(df) >= 1000:
        OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        locked = _acquire_cache_lock(symbol)
        if locked:
            try:
                df.to_parquet(cache_path)
                print(f"  💾 Cached {len(df)} OHLCV bars → {cache_path}")
            finally:
                _release_cache_lock(symbol)
        else:
            # Lock held by another process — skip write entirely
            # Return fetched data without caching (other process will cache it)
            print(f"  ⏭ Cache write skipped for {symbol} (locked by another process)")

    return df


def _fetch_5m_public_klines(symbol: str, days: int = 120) -> Optional[pd.DataFrame]:
    """Fetch 5m OHLCV from Binance public Futures API (no auth needed).
    Uses fapi.binance.com — same endpoint as the live trader's SignalGenerator.
    This ensures ALL processes (live, training, retrain) use identical data.
    """
    import requests
    end = datetime.now()
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    limit = 1000
    all_bars = []
    last_ts = start_ms
    url = "https://fapi.binance.com/fapi/v1/klines"

    while last_ts < end_ms:
        params = {
            'symbol': symbol,
            'interval': '5m',
            'limit': limit,
            'startTime': last_ts,
            'endTime': end_ms,
        }
        try:
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
        except Exception:
            break

    if len(all_bars) < 1000:
        print(f"    ❌ Insufficient data: {len(all_bars)} bars")
        return None

    df = pd.DataFrame(all_bars, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('open_time', inplace=True)
    df = df[~df.index.duplicated(keep='last')].sort_index()
    # Keep ALL columns: OHLCV + taker_buy + quote_volume + trades (for scalping features)
    for col in ['taker_buy_quote', 'quote_volume', 'trades', 'taker_buy_base']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    keep = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_quote', 'quote_volume', 'trades']
    available = [c for c in keep if c in df.columns]
    return df[available]


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
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }
        # Include additional columns if available
        for extra_col in ['taker_buy_quote', 'quote_volume', 'trades']:
            if extra_col in df_5m.columns:
                agg_dict[extra_col] = 'sum'

        resampled = df_5m.resample(rule).agg(agg_dict).dropna()

        # Ensure OHLC consistency: low <= open, close <= high
        resampled['low'] = resampled[['open', 'close', 'low']].min(axis=1)
        resampled['high'] = resampled[['open', 'close', 'high']].max(axis=1)

        # CRITICAL FIX: Drop ALL higher-TF bars that may be incomplete.
        # Pandas resample creates a bar as soon as ANY data in that window exists.
        # At inference time, e.g. at 14:30, the 1h bar 14:00-15:00 contains only
        # 5m data from 14:00-14:30 — its close/volume is NOT final.
        # This causes features to change each time new 5m data arrives!
        # A bar at index t is complete iff the window [t, t+rule) is fully
        # contained within the available 5m data, i.e. t+rule <= last_ts.
        last_ts = df_5m.index[-1]
        bar_close = resampled.index + pd.Timedelta(rule)
        resampled = resampled[bar_close <= last_ts]

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
        for col in ["open", "high", "low", "close", "volume", "taker_buy_quote", "quote_volume", "trades"]:
            if col in result.columns:
                del result[col]

    # ── ENHANCED FEATURES v2 ─────────────────────────────────────
    # Chaikin Money Flow (CMF, 20)
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    cmf20 = mfv.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    result[f"{p}cmf_20"] = cmf20

    # Money Flow Index (MFI, 14)
    tp = (high + low + close) / 3
    mf = tp * volume
    mf_pos = mf.where(tp > tp.shift(1), 0).rolling(14).sum()
    mf_neg = mf.where(tp < tp.shift(1), 0).rolling(14).sum()
    mfr = mf_pos / mf_neg.replace(0, np.nan)
    result[f"{p}mfi_14"] = 100 - (100 / (1 + mfr))

    # Elder's Force Index (13)
    result[f"{p}force_idx_13"] = (close - close.shift(1)) * volume

    # Williams %R (14)
    high14 = high.rolling(14).max()
    low14 = low.rolling(14).min()
    result[f"{p}williams_r_14"] = -100 * (high14 - close) / (high14 - low14).replace(0, np.nan)

    # Keltner %B (20, 2 ATR)
    ema20 = close.ewm(span=20).mean()
    kc_upper = ema20 + 2 * result[f"{p}atr_14"]
    kc_lower = ema20 - 2 * result[f"{p}atr_14"]
    result[f"{p}kc_pct_b"] = (close - kc_lower) / (kc_upper - kc_lower).replace(0, np.nan)
    result[f"{p}kc_width"] = (kc_upper - kc_lower) / ema20.replace(0, np.nan)

    # Volatility Z-Score (current ATR vs 50-period ATR)
    atr14 = result[f"{p}atr_14"]
    result[f"{p}vol_zscore"] = (atr14 - atr14.rolling(50).mean()) / atr14.rolling(50).std().replace(0, np.nan)

    # Rolling Sharpe (20 bars)
    ret_1 = close.pct_change()
    result[f"{p}roll_sharpe_20"] = ret_1.rolling(20).mean() / ret_1.rolling(20).std().replace(0, np.nan)

    # Efficiency Ratio (10): directionality / noise
    price_move = abs(close - close.shift(10))
    noise = (abs(close - close.shift(1))).rolling(10).sum().replace(0, np.nan)
    result[f"{p}efficiency_ratio_10"] = price_move / noise

    # Rolling Correlation: close vs volume (20)
    result[f"{p}corr_close_vol_20"] = close.rolling(20).corr(volume)

    # Net Volume Intensity: (close > open → +volume, close < open → -volume)
    net_vol = volume.where(close > open_p, -volume).cumsum()
    result[f"{p}net_vol_intensity"] = net_vol
    result[f"{p}net_vol_sma_20"] = net_vol.rolling(20).mean()
    result[f"{p}net_vol_ratio"] = net_vol / net_vol.rolling(20).mean().replace(0, np.nan)

    # Acceleration: 2nd derivative of price
    result[f"{p}acceleration"] = close.diff().diff()

    # Price ROC with multiple periods
    for period in [5, 10, 20]:
        result[f"{p}roc_{period}"] = (close / close.shift(period) - 1) * 100

    return result


def compute_5m_features_5tf(
    df_5m: pd.DataFrame,
    target_candle: int = 9,
    target_threshold: float = 0.005,
    intervals: Optional[List[str]] = None,
    for_inference: bool = False,
    available_until: Optional[int] = None,
    market_data: Optional[Dict] = None,
    df_1h: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Simplified feature computation: 5m execution + 1h predictive features.
    
    Uses DIRECT 1h OHLCV from Binance (fetched independently from 5m data).
    This eliminates look-ahead bias — each 1h bar is a complete historical entity.
    Training, validation, and inference all use the SAME methodology.
    
    2 timeframes used: 5m (timing/alignment), 1h (prediction).
    
    Args:
        df_5m: 5m OHLCV DataFrame
        target_candle: Number of 5m candles forward to predict (default: 8 = 40 min)
        target_threshold: Return threshold for binary target (default: 0.003 = 0.3%)
        intervals: Timeframes to compute (default: ['1h'])
        for_inference: If True, skips target column
        available_until: If set, simulates live conditions with limited data
        market_data: Optional dict of market data (funding rate, OI, etc.)
        df_1h: Optional pre-fetched 1h OHLCV (NO resampling from 5m)
               If provided, used directly — eliminates look-ahead bias.
               If None (legacy), falls back to resample_to_timeframes (WARNING: look-ahead possible)
    """
    if intervals is None:
        intervals = ['1h']  # Only 1h — 5m/15m/30m/4h had zero MI

    # ── Per-bar mode: use data subset for 5m features ──
    per_bar = available_until is not None
    if per_bar:
        df_5m_full = df_5m.copy()  # 5m features use FULL data (trailing — safe)
    else:
        df_5m_full = df_5m

    # Step 1: Get 1h OHLCV — DIRECT from Binance (NO resampling from 5m)
    # This is the CRITICAL fix: 1h features must use independent 1h data
    # to eliminate look-ahead bias from 5m→1h resampling.
    if df_1h is None:
        # Legacy fallback: resample 5m to 1h (⚠️ WARNING: look-ahead possible!)
        tf_data = resample_to_timeframes(df_5m, intervals)
        tf_data['5m'] = df_5m_full
    else:
        # NEW: use independent 1h data — each bar is a complete historical entity
        tf_data = {'5m': df_5m_full, '1h': df_1h.copy()}

        # In per-bar mode: only use 1h bars that were COMPLETE at cutoff time
        if per_bar:
            cutoff_time = df_5m_full.index[available_until]
            # A 1h bar at index T is complete at T + 1h
            df_1h_filtered = df_1h[df_1h.index + timedelta(hours=1) <= cutoff_time].copy()
            if len(df_1h_filtered) > 0:
                tf_data['1h'] = df_1h_filtered
            else:
                # No complete 1h bars yet — empty 1h features
                tf_data['1h'] = pd.DataFrame(index=df_1h.index[:0])
        
        # For inference: also drop the incomplete current 1h bar
        if for_inference and not per_bar:
            cutoff_time = datetime.utcnow()
            df_1h_filtered = df_1h[df_1h.index + timedelta(hours=1) <= cutoff_time].copy()
            if len(df_1h_filtered) > 0:
                tf_data['1h'] = df_1h_filtered
            else:
                tf_data['1h'] = pd.DataFrame(index=df_1h.index[:0])

    # During inference, additional safeguard: drop remaining incomplete bars
    if for_inference:
        for name, df_tf in tf_data.items():
            if name == '5m':
                continue
            if len(df_tf) == 0:
                continue
            if df_1h is not None and name == '1h':
                continue  # already filtered above
            last_bar_close = df_tf.index[-1] + timedelta(hours=1)
            if last_bar_close > datetime.utcnow():
                tf_data[name] = df_tf.iloc[:-1]
                print(f"    ⚠️ Dropped incomplete {name} bar: {df_tf.index[-1]}")
    
    # Step 2: Compute technical features on 5m (timing) and 1h (prediction)
    feats_5m = compute_technical_features(tf_data['5m'], prefix='m5_')
    if not per_bar:
        print(f"    Features computed: 5m = {len(feats_5m.columns)}")
    
    other_feats = {}
    for name, df_tf in tf_data.items():
        if name == '5m':
            continue
        if len(df_tf) == 0:
            other_feats[name] = pd.DataFrame(index=feats_5m.index)
            continue
        prefix_map = {'1h': 'h1_'}
        prefix = prefix_map.get(name, f'{name}_')
        other_feats[name] = compute_technical_features(df_tf, prefix=prefix)
        if not per_bar:
            print(f"    Features computed: {name} = {len(other_feats[name].columns)}")
    
    # Step 3: Align 1h to 5m index via forward-fill
    idx_5m = feats_5m.index
    
    fill_limits = {'1h': 12}
    if for_inference:
        fill_limits['1h'] = 36  # Bridge gap from incomplete bar
    
    feat_list = [feats_5m]
    for name, df_feat in other_feats.items():
        limit = fill_limits.get(name, 12)
        if len(df_feat) == 0:
            # No 1h features — create empty aligned DataFrame
            aligned = pd.DataFrame(index=idx_5m, columns=[])
        else:
            aligned = df_feat.reindex(idx_5m, method='ffill', limit=limit)
        feat_list.append(aligned)
    
    combined = pd.concat(feat_list, axis=1)
    
    # ── Cross-timeframe features (use proper data cutoff for per-bar mode) ──
    if per_bar:
        df_5m_cross = df_5m_full.iloc[:available_until + 1].copy()
        df_5m_trunc = df_5m_cross
        idx_xtf = [df_5m_cross.index[-1]]
    elif for_inference:
        # Existing logic: align to last non-NaN row
        valid_mask = combined.notna().all(axis=1)
        if valid_mask.any():
            last_valid_idx = combined.index[valid_mask][-1]
            last_pos = combined.index.get_loc(last_valid_idx)
            df_5m_trunc = df_5m[df_5m.index <= last_valid_idx].copy()
            combined = combined.iloc[:last_pos + 1].copy()
            idx_xtf = combined.index
        else:
            df_5m_trunc = df_5m.copy()
            idx_xtf = idx_5m
        df_5m_cross = df_5m_trunc
    else:
        df_5m_trunc = df_5m.copy()
        idx_xtf = idx_5m
        df_5m_cross = df_5m
    
    close_5m = df_5m_trunc['close'].astype(float) if len(df_5m_trunc) > 0 else pd.Series(dtype=float)
    volume_5m = df_5m_trunc['volume'].astype(float) if len(df_5m_trunc) > 0 else pd.Series(dtype=float)
    
    xtf = pd.DataFrame(index=idx_xtf)
    
    # 1h Trend regime — use only available 1h data (per_bar supports it)
    has_1h_ohlcv = '1h' in tf_data and len(tf_data['1h']) > 0
    if has_1h_ohlcv:
        tf_close = tf_data['1h']['close'].astype(float)
        for period in [20, 50]:
            sma = tf_close.rolling(period).mean()
            trend = pd.Series(0, index=tf_close.index)
            trend[sma > sma.shift(1)] = 1
            trend[sma < sma.shift(1)] = -1
            aligned = trend.reindex(idx_xtf, method='ffill', limit=fill_limits.get('1h', 12))
            xtf[f'h1_trend_sma{period}'] = aligned
    
    has_1h_feats = '1h' in other_feats and len(other_feats['1h']) > 0
    if has_1h_feats:
        # Volatility ratio: 5m ATR vs 1h ATR
        atr_1h_aligned = other_feats['1h']['h1_atr_14'].reindex(idx_xtf, method='ffill', limit=12)
        xtf['vol_ratio_5m_vs_1h'] = feats_5m['m5_atr_14'].reindex(idx_xtf) / atr_1h_aligned.replace(0, np.nan)
        
        # RSI divergence 5m vs 1h
        rsi_1h_aligned = other_feats['1h']['h1_rsi_14'].reindex(idx_xtf, method='ffill', limit=12)
        xtf['rsi_div_5m_vs_1h'] = feats_5m['m5_rsi_14'].reindex(idx_xtf) - rsi_1h_aligned
        
        # Volume ratio
        vol_1h_avg = df_5m_cross['volume'].astype(float).rolling(12).sum().resample('1h').mean()
        vol_1h_avg_aligned = vol_1h_avg.reindex(idx_xtf, method='ffill', limit=12)
        xtf['vol_5m_vs_1h_avg'] = volume_5m / vol_1h_avg_aligned.replace(0, np.nan)
        
        # Consolidation detection
        range_5m_20 = (df_5m_cross['high'].astype(float).rolling(20).max()
                       - df_5m_cross['low'].astype(float).rolling(20).min())
        range_1h_20 = (df_5m_cross['high'].astype(float).resample('1h').max().rolling(20).mean()
                       - df_5m_cross['low'].astype(float).resample('1h').min().rolling(20).mean())
        range_1h_20_aligned = range_1h_20.reindex(idx_xtf, method='ffill', limit=12)
        xtf['consolidation_ratio'] = range_5m_20.reindex(idx_xtf) / range_1h_20_aligned.replace(0, np.nan)
        xtf['is_consolidating'] = (range_5m_20.reindex(idx_xtf) < range_5m_20.reindex(idx_xtf).rolling(40).mean() * 0.5).astype(float)
        
        # 1h bullish alignment
        close_1h = tf_data['1h']['close'].astype(float)
        trend_bull = (close_1h > close_1h.rolling(20).mean()).astype(int)
        xtf['1h_bullish'] = trend_bull.reindex(idx_xtf, method='ffill', limit=fill_limits.get('1h', 12))
    
    # ── Combine ALL features ──
    combined = pd.concat([combined, xtf], axis=1)

    # ── MICRO-STRUCTURE SCALPING FEATURES (from klines) ───────────
    # taker_buy_quote, quote_volume, trades are now preserved in OHLCV cache
    if 'taker_buy_quote' in df_5m_full.columns and 'quote_volume' in df_5m_full.columns:
        close_full = df_5m_full['close'].astype(float)
        taker_quote = df_5m_full['taker_buy_quote'].astype(float).fillna(0)
        quote_vol = df_5m_full['quote_volume'].astype(float).fillna(0)
        trades = df_5m_full['trades'].astype(float).fillna(0)
        volume_full = df_5m_full['volume'].astype(float).fillna(0)
        
        # ── 1. TAKER BUY RATIO (accurate — using quote_volume directly) ──
        total_quote = quote_vol  # direct from Binance — more accurate than close*volume
        taker_buy_ratio = taker_quote / total_quote.replace(0, np.nan)
        taker_buy_ratio = taker_buy_ratio.fillna(0.5).clip(0, 1)
        
        tbr = taker_buy_ratio.reindex(combined.index)
        combined['taker_buy_ratio'] = tbr
        for w in [6, 12, 24, 48, 96]:  # 30m, 1h, 2h, 4h, 8h at 5m
            combined[f'taker_buy_ratio_sma_{w}'] = tbr.rolling(w, min_periods=3).mean()
            combined[f'taker_buy_ratio_zscore_{w}'] = (
                (tbr - tbr.rolling(w, min_periods=3).mean())
                / tbr.rolling(w, min_periods=3).std().replace(0, np.nan)
            ).fillna(0)
        combined['taker_buy_extreme'] = (tbr > 0.75).astype(float)
        combined['taker_sell_extreme'] = (tbr < 0.25).astype(float)
        
        # ── 2. TRADE FREQUENCY (scalping: spike in trades = volatility) ──
        num_trades = trades.reindex(combined.index)
        combined['num_trades'] = num_trades
        for w in [6, 12, 24]:
            combined[f'trade_freq_zscore_{w}'] = (
                (num_trades - num_trades.rolling(w, min_periods=3).mean())
                / num_trades.rolling(w, min_periods=3).std().replace(0, np.nan)
            ).fillna(0)
        combined['trade_spike'] = (num_trades > num_trades.rolling(24).mean() * 2).astype(float)
        
        # ── 3. AVERAGE TRADE SIZE (micro-structure: whale trades vs retail) ──
        avg_size = quote_vol / trades.replace(0, np.nan)
        avg_size = avg_size.fillna(0)
        avg_size_aligned = avg_size.reindex(combined.index)
        combined['avg_trade_size'] = avg_size_aligned
        for w in [6, 12, 24]:
            combined[f'avg_trade_size_zscore_{w}'] = (
                (avg_size_aligned - avg_size_aligned.rolling(w, min_periods=3).mean())
                / avg_size_aligned.rolling(w, min_periods=3).std().replace(0, np.nan)
            ).fillna(0)
        
        # ── 4. MICRO MOMENTUM (candle body: close - open) ──
        open_full = df_5m_full['open'].astype(float)
        body = close_full - open_full
        body_aligned = body.reindex(combined.index)
        combined['micro_body'] = body_aligned / close_full.reindex(combined.index).replace(0, np.nan) * 100  # %
        combined['micro_body_abs'] = abs(body_aligned) / close_full.reindex(combined.index).replace(0, np.nan) * 100
        for w in [3, 6, 12]:  # 15m, 30m, 1h
            combined[f'micro_body_roll_{w}'] = body_aligned.rolling(w).sum()
            combined[f'micro_hl_range_{w}'] = (
                df_5m_full['high'].astype(float).rolling(w).max()
                - df_5m_full['low'].astype(float).rolling(w).min()
            ).reindex(combined.index) / close_full.reindex(combined.index).replace(0, np.nan) * 100
        
        # ── 5. QUOTE VOLUME FLOW ──
        qv = quote_vol.reindex(combined.index)
        combined['quote_volume'] = qv
        for w in [6, 12, 24]:
            combined[f'quote_vol_ratio_{w}'] = qv / qv.rolling(w, min_periods=3).mean().replace(0, np.nan)

    # ── MARKET DATA FEATURES (Funding Rate, OI, Top Trader) ────────────
    # These are fetched from cache and aligned to 5m index
    try:
        from src.strategies.market_data_cache import (
            ensure_all_market_data, align_to_5m, compute_market_features
        )

        if market_data is None:
            # Auto-fetch from cache (fast — parquet files)
            market_data = ensure_all_market_data(
                # We don't have symbol here, but that's fine — cache files exist
                # For training, market_data is passed in from prepare_ml_dataset
            )
            market_data = None  # can't auto-fetch without symbol

        if market_data is not None:
            idx_5m_aligned = combined.index

            # Funding Rate Features
            fr_df = market_data.get('funding_rate')
            if fr_df is not None and len(fr_df) > 0:
                fr_aligned = align_to_5m(fr_df, idx_5m_aligned)
                fr_feats = compute_market_features(fr_aligned, prefix='fr_')
                # Add to combined (only columns that don't already exist)
                for col in fr_feats.columns:
                    if col not in combined.columns:
                        combined[col] = fr_feats[col].values

            # Open Interest Features
            oi_df = market_data.get('open_interest')
            if oi_df is not None and len(oi_df) > 0:
                oi_aligned = align_to_5m(oi_df, idx_5m_aligned)
                oi_feats = compute_market_features(oi_aligned, prefix='oi_')
                for col in oi_feats.columns:
                    if col not in combined.columns:
                        combined[col] = oi_feats[col].values

                # OI-Price divergence (smart money signal)
                if 'close' in df_5m_full.columns and 'oi_sumOpenInterest' in oi_aligned.columns:
                    close_full = df_5m_full['close'].astype(float).reindex(idx_5m_aligned)
                    oi_val = pd.to_numeric(oi_aligned['sumOpenInterest'], errors='coerce').reindex(idx_5m_aligned)
                    # 24-period (2h) correlation between price and OI
                    oi_price_corr = close_full.rolling(24).corr(oi_val)
                    combined['oi_price_corr_24'] = oi_price_corr.fillna(0)
                    # Negative corr = divergence (bearish if price up, OI down)
                    combined['oi_price_divergence'] = (oi_price_corr < -0.5).astype(float)

            # Top Trader Account Ratio Features
            top_df = market_data.get('top_trader_account')
            if top_df is not None and len(top_df) > 0:
                top_aligned = align_to_5m(top_df, idx_5m_aligned)
                top_feats = compute_market_features(top_aligned, prefix='top_')
                for col in top_feats.columns:
                    if col not in combined.columns:
                        combined[col] = top_feats[col].values

            # Top Trader Position Ratio Features
            topp_df = market_data.get('top_trader_position')
            if topp_df is not None and len(topp_df) > 0:
                topp_aligned = align_to_5m(topp_df, idx_5m_aligned)
                topp_feats = compute_market_features(topp_aligned, prefix='topp_')
                for col in topp_feats.columns:
                    if col not in combined.columns:
                        combined[col] = topp_feats[col].values

    except ImportError:
        pass  # market_data_cache not available — skip these features
    except Exception as e:
        print(f"    ⚠️ Market data features error: {e}")
        pass  # non-fatal — continue without market features

    # ── Time-based features (computed once at 5m level) ──
    if for_inference or not for_inference:
        idx_hour = combined.index.hour
        idx_minute = combined.index.minute
        # Session encoding: Asia 0-8, Euro 8-16, US 16-24
        session = pd.Series(0, index=combined.index)
        session[(idx_hour >= 0) & (idx_hour < 8)] = -1    # Asia
        session[(idx_hour >= 8) & (idx_hour < 16)] = 0    # Euro
        session[(idx_hour >= 16)] = 1                     # US
        combined['session'] = session
        # Sin/cos encoding of time
        hour_angle = 2 * np.pi * (idx_hour + idx_minute / 60) / 24
        combined['hour_sin'] = np.sin(hour_angle)
        combined['hour_cos'] = np.cos(hour_angle)
        # Day of week
        combined['day_of_week'] = combined.index.dayofweek / 6.0  # normalize 0-1
    
    # ── Target: dual — LONG predicts UP > threshold, SHORT predicts DOWN > threshold ──
    if not for_inference:
        ret_nbar = close_5m.shift(-(target_candle + 1)) / close_5m.shift(-1) - 1
        combined['target_long'] = (ret_nbar > target_threshold).astype(float)
        combined['target_short'] = (ret_nbar < -target_threshold).astype(float)
        combined['target'] = combined['target_long']  # default/backward compat
    
    # ── Drop NaN rows ──
    if per_bar:
        # Per-bar mode: return single row (no truncation needed)
        combined = combined.loc[[df_5m_trunc.index[-1]]].copy()
        pass  # single row, no NaN rows to worry about
    elif for_inference:
        combined = combined.iloc[200:].copy()
        now = datetime.utcnow()
        last_idx = combined.index[-1]
        incomplete_close = last_idx + timedelta(minutes=5)
        if incomplete_close > now:
            combined = combined.iloc[:-1]
            print(f"    ⚠️ Dropped incomplete 5m bar: {last_idx} (close at {incomplete_close.strftime('%H:%M')}, now={now.strftime('%H:%M')})")
        print(f"    Total features (inference): {len(combined.columns)}")
    else:
        combined = combined.iloc[200:-(target_candle + 1)].copy()
        print(f"    Total features: {len([c for c in combined.columns if c != 'target'])} + target")
    if not per_bar:
        print(f"    Rows: {len(combined)}")
    
    return combined


def prepare_ml_dataset(
    symbol: str,
    days: int = 120,
    cache_dir: str = "data/ml_cache",
    target_candle: int = 8,
    target_threshold: float = 0.003,
    intervals: Optional[List[str]] = None,
    side: str = 'long'
) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
    """
    Fetch 5m klines, compute features across timeframes, return X, y.
    
    Args:
        symbol: Trading pair e.g. 'BTCUSDT'
        days: How many days of historical data to fetch
        cache_dir: Cache directory
        target_candle: Number of 5m candles forward to predict (default: 8 = 40 min)
        target_threshold: Return threshold for binary target (default: 0.003 = 0.3%)
        intervals: Timeframes to resample and compute features on
        side: 'long' (UP > threshold) or 'short' (DOWN > threshold)
    
    Returns:
        (X, y, index) or (None, None, None) on failure
    """
    if intervals is None:
        intervals = ['1h']

    # Cache key: differentiate from old 15m-based cache
    tf_suffix = '_'.join(sorted(intervals))
    cache_path = Path(cache_dir) / f"{symbol}_5m_{days}d_{target_candle}c_{tf_suffix}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"  Loading cached features for {symbol} (5m base)...")
        df = pd.read_parquet(cache_path)
        feature_cols = [c for c in df.columns if c not in ("target", "target_long", "target_short")]
        target_col = f'target_{side}'
        if target_col not in df.columns:
            target_col = 'target'  # fallback for old caches
        print(f"    {len(df)} rows, {len(feature_cols)} features, target='{target_col}'")
        return df[feature_cols], df[target_col], df.index

    print(f"  Loading OHLCV data for {symbol} from shared cache...")
    df_5m = ensure_ohlcv_data(symbol, min_days=days)

    if df_5m is None or len(df_5m) < 1000:
        print(f"  ⚠️ Insufficient 5m data for {symbol}")
        return None, None, None

    print(f"    {symbol}: {len(df_5m)} bars of 5m data (shared cache)")
    print(f"  Computing features across {len(intervals)} timeframes for {symbol}...")

    # Fetch market data (funding rate, OI, top trader) — cached, fast
    market_data = None
    try:
        from src.strategies.market_data_cache import ensure_all_market_data
        print(f"  Fetching market data (funding rate, OI, top trader)...")
        market_data = ensure_all_market_data(symbol)
        if market_data:
            for k, v in market_data.items():
                print(f"    {k}: {len(v) if v is not None else 0} records")
    except Exception as e:
        print(f"  ⚠️ Market data fetch error: {e}")

    # Fetch 1h OHLCV DIRECTLY — NO resampling from 5m (eliminates look-ahead bias)
    df_1h = None
    if '1h' in intervals:
        try:
            print(f"  Fetching 1h OHLCV directly from Binance (no look-ahead)...")
            df_1h = ensure_ohlcv_1h(symbol, min_days=max(days // 24, 30))
            if df_1h is not None:
                print(f"    {symbol}: {len(df_1h)} bars of 1h data")
        except Exception as e:
            print(f"  ⚠️ 1h data fetch error: {e}")

    combined = compute_5m_features_5tf(df_5m, target_candle=target_candle,
                                       target_threshold=target_threshold,
                                       intervals=intervals,
                                       market_data=market_data,
                                       df_1h=df_1h)

    # Save to cache
    combined.to_parquet(cache_path)
    print(f"  Saved to {cache_path}")

    exclude = {"target", "target_long", "target_short"}
    feature_cols = [c for c in combined.columns if c not in exclude]
    target_col = f'target_{side}'
    return combined[feature_cols], combined[target_col], combined.index


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
                except Exception:
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
    for col in ['taker_buy_quote', 'quote_volume', 'trades', 'taker_buy_base']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    keep = ['open', 'high', 'low', 'close', 'volume', 'taker_buy_quote', 'quote_volume', 'trades']
    available = [c for c in keep if c in df.columns]
    return df[available]
