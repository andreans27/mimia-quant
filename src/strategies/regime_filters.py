"""
Regime & Volatility Filters — compute masks to gate ML signal entries.

Filters:
  - atr_filter:  skip entry when ATR(14)/price is above P90 (extreme volatility)
  - trend_filter: long only when H1 price > SMA(50), short only when H1 price < SMA(50)
  - combined:    both filters must pass
"""

import numpy as np
import pandas as pd


def rolling_percentile(series: pd.Series, window: int = 500, pct: float = 90) -> pd.Series:
    """Rolling percentile threshold. Returns True where value <= P{pct}."""
    if len(series) < window:
        window = max(len(series) // 2, 50)
    rolling = series.rolling(window, min_periods=max(50, window // 4))
    threshold = rolling.quantile(pct / 100)
    return series <= threshold


def compute_atr_filter(feat_df: pd.DataFrame,
                       period: int = 14,
                       percentile_window: int = 500,
                       percentile: float = 90,
                       min_bars: int = 100) -> np.ndarray:
    """
    ATR Volatility Filter.
    Skip entry when ATR(14)/close_pct is above rolling P{percentile}.

    Uses 'm5_atr_{period}' column from feature cache.
    Returns boolean mask (True = allow entry).
    """
    n = len(feat_df)
    if n < min_bars:
        return np.ones(n, dtype=bool)

    atr_col = f'm5_atr_{period}'
    if atr_col not in feat_df.columns:
        # fallback: compute ATR from OHLC
        close = feat_df['close'].values
        high = feat_df['high'].values
        low = feat_df['low'].values if 'low' in feat_df.columns else close
        tr = np.maximum(high - low,
                        np.abs(high - np.roll(close, 1)))
        tr[0] = tr[1]
        atr = pd.Series(tr).rolling(period, min_periods=period).mean().values
    else:
        atr = feat_df[atr_col].values

    close = feat_df['close'].values
    atr_pct = atr / np.maximum(close, 1e-8) * 100  # ATR as % of price

    atr_series = pd.Series(atr_pct)
    mask = rolling_percentile(atr_series, window=percentile_window, pct=percentile)

    return mask.values


def compute_trend_filter(feat_df: pd.DataFrame,
                         sma_period: int = 50,
                         tf: str = 'h1',
                         min_bars: int = 100) -> np.ndarray:
    """
    Trend Filter.
    Long (prob >= threshold): only if price > SMA({sma_period}) on {tf} timeframe.
    Short (prob <= 1-threshold): only if price < SMA({sma_period}) on {tf}.

    Uses '{tf}_dist_sma_{sma_period}' column (> 0 = bullish, < 0 = bearish).
    Returns three states via -1/0/1:
      1  → allow longs only
      -1 → allow shorts only
      0  → allow both (when no trend info available)
    Actually, for simplicity we return boolean mask for ALL entries.
    We filter direction separately based on sign.
    """
    n = len(feat_df)
    if n < min_bars:
        return np.zeros(n, dtype=int)  # 0 = no restriction

    col = f'{tf}_dist_sma_{sma_period}'
    if col in feat_df.columns:
        dist = feat_df[col].values
    else:
        # fallback: use close price vs simple SMA
        close = feat_df['close'].values
        sma = pd.Series(close).rolling(sma_period, min_periods=sma_period).mean().values
        dist = (close - sma) / np.maximum(sma, 1e-8) * 100

    # Return direction filter: 1 = bullish (allow long), -1 = bearish (allow short), 0 = neutral
    direction_filter = np.where(dist > 0, 1, np.where(dist < 0, -1, 0))
    return direction_filter


def compute_combined_filter(feat_df: pd.DataFrame) -> np.ndarray:
    """
    Combined filter mask for entry.
    Returns boolean array: True = allow entry based on both ATR and trend.
    Note: For trend, we use direction-aware filtering later.
    """
    atr_mask = compute_atr_filter(feat_df)
    trend_dir = compute_trend_filter(feat_df)
    return atr_mask, trend_dir


# ─── Backtest integration ──────────────────────────────────────────

def make_entry_mask(feat_df: pd.DataFrame,
                    use_atr: bool = True,
                    use_trend: bool = True) -> tuple:
    """
    Generate entry mask + direction filter for backtest integration.

    Returns:
      entry_mask:  bool array — True where entry is allowed by ATR filter
      dir_filter:  int array  — 1 (bullish), -1 (bearish), 0 (neutral)
    """
    n = len(feat_df)
    entry_mask = np.ones(n, dtype=bool)
    dir_filter = np.zeros(n, dtype=int)

    if use_atr:
        entry_mask = compute_atr_filter(feat_df)

    if use_trend:
        dir_filter = compute_trend_filter(feat_df)

    return entry_mask, dir_filter
