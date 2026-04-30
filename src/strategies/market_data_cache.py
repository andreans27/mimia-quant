"""
Market Data Cache for Mimia Quant Trading System.
===================================================
Caches and serves NON-OHLCV market data from Binance Futures:
- Funding Rate History (fapi/v1/fundingRate, 330 days, 8h intervals)
- Open Interest History (futures/data/openInterestHist, ~21 days, 1h)
- Top Trader Long/Short Ratio (futures/data/topLongShortAccountRatio, ~21 days, 1h)
- Top Trader Position Ratio (futures/data/topLongShortPositionRatio, ~21 days, 1h)

All data is cached to parquet files for incremental updates, similar to OHLCV cache.

USAGE:
    mdc = MarketDataCache(symbol="WIFUSDT")
    funding_df = mdc.ensure_funding_rate()     # returns DataFrame with DatetimeIndex
    oi_df = mdc.ensure_open_interest()          # returns DataFrame with DatetimeIndex
    taker_ratio_df = mdc.ensure_taker_ratio(period="1h")
    top_trader_df = mdc.ensure_top_trader(period="1h")

Each returns a DataFrame with DatetimeIndex. Use align_to_5m() to forward-fill to 5m index.
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
import requests

# ── Config ──
MARKET_CACHE_DIR = Path("data/market_cache")
MARKET_CACHE_DIR.mkdir(parents=True, exist_ok=True)
API_BASE = "https://fapi.binance.com"


def _fetch_with_pagination(url: str, params: Dict, data_key: str = None,
                           limit: int = 1000, max_records: int = 2000) -> List[Dict]:
    """Fetch paginated data from Binance Futures API.
    Uses timestamp-based pagination.
    """
    all_data = []
    current_start = params.get('startTime')
    if not current_start:
        # Get one batch to find the earliest timestamp
        r = requests.get(url, params=params, timeout=30)
        if r.status_code != 200:
            return []
        data = r.json()
        if not data:
            return []
        all_data.extend(data)
        if len(data) < limit:
            return all_data
        current_start = int(data[-1].get('timestamp', data[-1].get('fundingTime', 0))) + 1

    while len(all_data) < max_records:
        p = dict(params)
        p['startTime'] = current_start
        p['limit'] = limit
        try:
            r = requests.get(url, params=p, timeout=30)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            if len(batch) < limit:
                break
            current_start = int(batch[-1].get('timestamp', batch[-1].get('fundingTime', 0))) + 1
        except Exception:
            break

    return all_data


def _convert_timestamp_col(df: pd.DataFrame, ts_col: str) -> pd.DataFrame:
    """Convert timestamp column to DatetimeIndex."""
    if ts_col in df.columns:
        df['ts'] = pd.to_datetime(df[ts_col], unit='ms')
        df.set_index('ts', inplace=True)
    return df


def _num(s: Any) -> float:
    """Convert string or number to float safely."""
    try:
        return float(str(s))
    except (ValueError, TypeError):
        return 0.0


# ── Cache File Helpers ──

def _cache_path(symbol: str, data_type: str) -> Path:
    """Get cache file path, handling prefix normalization."""
    return MARKET_CACHE_DIR / f"{symbol}_{data_type}.parquet"


def _write_cache(df: pd.DataFrame, path: Path):
    """Thread-safe cache write with lock."""
    lock_path = path.with_suffix(".lock")
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        try:
            df.to_parquet(path)
        finally:
            lock_path.unlink(missing_ok=True)
    except FileExistsError:
        pass  # another process holds the lock; skip write


def _read_cache(path: Path) -> Optional[pd.DataFrame]:
    """Read cached parquet if available."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


# ── Individual Data Fetch Functions ──

def fetch_funding_rate(symbol: str, limit: int = 1000) -> Optional[pd.DataFrame]:
    """Fetch funding rate history from Binance Futures.
    Returns DataFrame with DatetimeIndex, columns: ['fundingRate', 'markPrice'].
    Fetches latest records first, then paginates backward for full history.
    Max ~330 days (~3000 records at 8h intervals).
    """
    url = f"{API_BASE}/fapi/v1/fundingRate"
    all_data = []

    # Step 1: Fetch latest records (no startTime → returns most recent)
    params = {'symbol': symbol, 'limit': limit}
    try:
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 200:
            batch = r.json()
            all_data.extend(batch)
    except Exception:
        pass

    if not all_data:
        return None

    # Step 2: Paginate backward from earliest record
    earliest_ts = min(int(d['fundingTime']) for d in all_data)
    while len(all_data) < 3000:
        params = {'symbol': symbol, 'limit': limit, 'endTime': earliest_ts - 1}
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code != 200:
                break
            batch = r.json()
            if not batch:
                break
            all_data.extend(batch)
            new_earliest = min(int(d['fundingTime']) for d in batch)
            if new_earliest >= earliest_ts:
                break  # no progress
            earliest_ts = new_earliest
            if len(batch) < limit:
                break
        except Exception:
            break

    df = pd.DataFrame(all_data)
    df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')
    _convert_timestamp_col(df, 'fundingTime')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['fundingRate', 'markPrice']]


def fetch_open_interest_hist(symbol: str, period: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch open interest history.
    Returns DataFrame with DatetimeIndex, columns: ['sumOpenInterest', 'sumOpenInterestValue'].
    Max ~21 days of 1h data (500 records).
    """
    url = f"{API_BASE}/futures/data/openInterestHist"
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None

    df = pd.DataFrame(data)
    df['sumOpenInterest'] = pd.to_numeric(df['sumOpenInterest'], errors='coerce')
    df['sumOpenInterestValue'] = pd.to_numeric(df['sumOpenInterestValue'], errors='coerce')
    _convert_timestamp_col(df, 'timestamp')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['sumOpenInterest', 'sumOpenInterestValue']]


def fetch_taker_ratio(symbol: str, period: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch taker long/short ratio.
    Returns DataFrame with DatetimeIndex, columns: ['buySellRatio', 'buyVol', 'sellVol'].
    Max ~20 days at 1h.
    """
    url = f"{API_BASE}/futures/data/takerlongshortRatio"
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None

    df = pd.DataFrame(data)
    df['buySellRatio'] = pd.to_numeric(df['buySellRatio'], errors='coerce')
    df['buyVol'] = pd.to_numeric(df['buyVol'], errors='coerce')
    df['sellVol'] = pd.to_numeric(df['sellVol'], errors='coerce')
    _convert_timestamp_col(df, 'timestamp')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['buySellRatio', 'buyVol', 'sellVol']]


def fetch_top_trader_account_ratio(symbol: str, period: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch top trader long/short account ratio.
    Returns DataFrame with DatetimeIndex, columns: ['longShortRatio', 'longAccount', 'shortAccount'].
    Max ~21 days at 1h.
    """
    url = f"{API_BASE}/futures/data/topLongShortAccountRatio"
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None

    df = pd.DataFrame(data)
    df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
    df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
    df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')
    _convert_timestamp_col(df, 'timestamp')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['longShortRatio', 'longAccount', 'shortAccount']]


def fetch_top_trader_position_ratio(symbol: str, period: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    """Fetch top trader long/short position ratio.
    Returns DataFrame with DatetimeIndex, columns: ['longShortRatio', 'longAccount', 'shortAccount'].
    """
    url = f"{API_BASE}/futures/data/topLongShortPositionRatio"
    params = {'symbol': symbol, 'period': period, 'limit': limit}
    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    data = r.json()
    if not data:
        return None

    df = pd.DataFrame(data)
    df['longShortRatio'] = pd.to_numeric(df['longShortRatio'], errors='coerce')
    df['longAccount'] = pd.to_numeric(df['longAccount'], errors='coerce')
    df['shortAccount'] = pd.to_numeric(df['shortAccount'], errors='coerce')
    _convert_timestamp_col(df, 'timestamp')
    df = df[~df.index.duplicated(keep='last')].sort_index()
    return df[['longShortRatio', 'longAccount', 'shortAccount']]


# ── Align to 5m Index ──

def align_to_5m(market_df: Optional[pd.DataFrame], idx_5m: pd.DatetimeIndex,
                method: str = 'ffill', limit: int = 24) -> pd.DataFrame:
    """Align market data (irregular or lower-frequency) to a 5m DatetimeIndex.

    Args:
        market_df: Source DataFrame with DatetimeIndex
        idx_5m: Target 5m DatetimeIndex
        method: 'ffill' (forward fill) or 'nearest'
        limit: Max periods to forward-fill

    Returns:
        DataFrame aligned to idx_5m, or empty DataFrame if market_df is None/empty
    """
    if market_df is None or len(market_df) == 0:
        return pd.DataFrame(index=idx_5m)

    # Ensure timezone compatibility between indices
    market_idx = market_df.index
    target_idx = idx_5m

    if hasattr(market_idx, 'tz') and market_idx.tz is not None:
        if hasattr(target_idx, 'tz') and target_idx.tz is None:
            target_idx = target_idx.tz_localize(market_idx.tz)
    if hasattr(target_idx, 'tz') and target_idx.tz is not None:
        if hasattr(market_idx, 'tz') and market_idx.tz is None:
            market_idx = market_idx.tz_localize(target_idx.tz)
            # Reindex the market_df with the localized index
            market_df = market_df.copy()
            market_df.index = market_idx

    aligned = market_df.reindex(target_idx, method=method, limit=limit)

    # For rows before the first market data point, backfill
    first_valid = aligned.first_valid_index()
    if first_valid is not None:
        first_pos = aligned.index.get_loc(first_valid)
        if first_pos > 0:
            # Backfill first rows from earliest market data
            aligned.iloc[:first_pos] = aligned.iloc[first_pos]

    return aligned


# ── Feature Computation from Market Data ──

def compute_market_features(market_df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Compute features from a single market data source.

    Args:
        market_df: Market data aligned to 5m index (from align_to_5m)
        prefix: Column prefix (e.g. 'fr_', 'oi_', 'top_')

    Returns:
        DataFrame with computed features (single or multi-column)
    """
    result = pd.DataFrame(index=market_df.index)
    p = prefix

    if len(market_df) == 0 or len(market_df.columns) == 0:
        return result

    # Process each numeric column
    for col in market_df.columns:
        s = pd.to_numeric(market_df[col], errors='coerce')

        # Raw value
        result[f"{p}{col}"] = s

        # Rate of change over various windows
        for w in [6, 12, 24, 72]:  # 30min, 1h, 2h, 6h at 5m
            result[f"{p}{col}_roc_{w}"] = s.pct_change(w).fillna(0)

        # Z-score over various windows
        for w in [72, 144, 288]:  # 6h, 12h, 24h at 5m
            mean_w = s.rolling(w, min_periods=10).mean()
            std_w = s.rolling(w, min_periods=10).std().replace(0, np.nan)
            result[f"{p}{col}_zscore_{w}"] = ((s - mean_w) / std_w).fillna(0)

        # Rolling min/max (extremes)
        for w in [72, 144]:
            result[f"{p}{col}_min_{w}"] = s.rolling(w, min_periods=10).min()
            result[f"{p}{col}_max_{w}"] = s.rolling(w, min_periods=10).max()
            # Position in range
            rng = (result[f"{p}{col}_max_{w}"] - result[f"{p}{col}_min_{w}"]).replace(0, np.nan)
            result[f"{p}{col}_pct_pos_{w}"] = ((s - result[f"{p}{col}_min_{w}"]) / rng).fillna(0.5)

    return result


# ── Convenience Function: Fetch ALL Market Data for a Symbol ──

def ensure_all_market_data(symbol: str, force_refresh: bool = False) -> Dict[str, Optional[pd.DataFrame]]:
    """Fetch ALL market data sources for a symbol, using cache.

    Args:
        symbol: Trading pair
        force_refresh: If True, skip cache and re-fetch all

    Returns:
        Dict with keys: 'funding_rate', 'open_interest', 'taker_ratio',
                        'top_trader_account', 'top_trader_position'
    """
    result = {}

    # Funding Rate (cached)
    if not force_refresh:
        cached = _read_cache(_cache_path(symbol, "funding_rate"))
        if cached is not None:
            result['funding_rate'] = cached
        else:
            df = fetch_funding_rate(symbol)
            if df is not None:
                _write_cache(df, _cache_path(symbol, "funding_rate"))
            result['funding_rate'] = df
    else:
        df = fetch_funding_rate(symbol)
        if df is not None:
            _write_cache(df, _cache_path(symbol, "funding_rate"))
        result['funding_rate'] = df

    # Open Interest
    if not force_refresh:
        cached = _read_cache(_cache_path(symbol, "open_interest"))
        if cached is not None:
            result['open_interest'] = cached
        else:
            df = fetch_open_interest_hist(symbol)
            if df is not None:
                _write_cache(df, _cache_path(symbol, "open_interest"))
            result['open_interest'] = df
    else:
        df = fetch_open_interest_hist(symbol)
        if df is not None:
            _write_cache(df, _cache_path(symbol, "open_interest"))
        result['open_interest'] = df

    # Taker Ratio
    if not force_refresh:
        cached = _read_cache(_cache_path(symbol, "taker_ratio"))
        if cached is not None:
            result['taker_ratio'] = cached
        else:
            df = fetch_taker_ratio(symbol)
            if df is not None:
                _write_cache(df, _cache_path(symbol, "taker_ratio"))
            result['taker_ratio'] = df
    else:
        df = fetch_taker_ratio(symbol)
        if df is not None:
            _write_cache(df, _cache_path(symbol, "taker_ratio"))
        result['taker_ratio'] = df

    # Top Trader Account Ratio
    if not force_refresh:
        cached = _read_cache(_cache_path(symbol, "top_trader_account"))
        if cached is not None:
            result['top_trader_account'] = cached
        else:
            df = fetch_top_trader_account_ratio(symbol)
            if df is not None:
                _write_cache(df, _cache_path(symbol, "top_trader_account"))
            result['top_trader_account'] = df
    else:
        df = fetch_top_trader_account_ratio(symbol)
        if df is not None:
            _write_cache(df, _cache_path(symbol, "top_trader_account"))
        result['top_trader_account'] = df

    # Top Trader Position Ratio
    if not force_refresh:
        cached = _read_cache(_cache_path(symbol, "top_trader_position"))
        if cached is not None:
            result['top_trader_position'] = cached
        else:
            df = fetch_top_trader_position_ratio(symbol)
            if df is not None:
                _write_cache(df, _cache_path(symbol, "top_trader_position"))
            result['top_trader_position'] = df
    else:
        df = fetch_top_trader_position_ratio(symbol)
        if df is not None:
            _write_cache(df, _cache_path(symbol, "top_trader_position"))
        result['top_trader_position'] = df

    return result
