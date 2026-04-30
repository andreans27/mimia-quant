#!/usr/bin/env python3
"""
Mimia Quant - Live Trading Signal Generator
=============================================
Multi-timeframe XGBoost ensemble signal generation for live trading.
Fetches live 5m OHLCV from Binance public API, computes features,
and produces ensemble probabilities.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import json
import warnings
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import requests
import xgboost as xgb

warnings.filterwarnings('ignore')

from src.trading.state import MODEL_DIR, SEEDS, TF_GROUPS, THRESHOLD, FETCH_DAYS, get_symbol_threshold
from src.strategies.ml_features import OHLCV_CACHE_DIR, OHLCV_FETCH_DAYS


class SignalGenerator:
    """Generates trading signals using dual XGBoost ensembles (long + short)."""

    def __init__(self, symbol: str, model_side: str = 'auto'):
        self.symbol = symbol
        self.model_side = model_side  # 'long', 'short', or 'auto' (both)
        self._cache: Dict[str, Any] = {}  # symbol -> loaded data
        self._calibrator: Dict[str, Optional[Dict]] = {}  # side -> calibrator (cached)

    def _load_calibrator(self, symbol: str, target_side: str = 'long') -> Optional[Dict]:
        """Load Platt scaling calibrator for a specific model side (cached)."""
        if target_side in self._calibrator:
            return self._calibrator[target_side]
        cal_path = MODEL_DIR / f"{symbol}_{target_side}_calibrator.json"
        if cal_path.exists():
            try:
                with open(cal_path) as f:
                    cal = json.load(f)
                self._calibrator[target_side] = cal
                return cal
            except Exception:
                return None
        return None

    def _apply_calibration(self, raw_proba: float, target_side: str = 'long') -> float:
        """Apply Platt scaling for a specific model side."""
        cal = self._load_calibrator(self.symbol, target_side)
        if cal is None:
            return raw_proba
        z = cal['coef'] * raw_proba + cal['intercept']
        calibrated = 1.0 / (1.0 + np.exp(-z))
        return float(np.clip(calibrated, 0.0, 1.0))

    def _fetch_5m_ohlcv(self, symbol: str, days: int = FETCH_DAYS) -> Optional[pd.DataFrame]:
        """Fetch 5m OHLCV from Binance public API (faster than testnet)."""
        end = datetime.now()
        start = end - timedelta(days=days)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        limit = 1000
        all_bars = []
        last_ts = start_ms
        while last_ts < end_ms:
            url = f"https://fapi.binance.com/fapi/v1/klines"
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
            return None

        df = pd.DataFrame(all_bars, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        for col in ['taker_buy_quote', 'quote_volume', 'trades', 'taker_buy_base']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_quote', 'quote_volume', 'trades']]

    # ── OHLCV Cache (Incremental Update) ─────────────────────────────────

    def _get_cached_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached OHLCV data from local parquet."""
        cache_path = OHLCV_CACHE_DIR / f"{symbol}_5m.parquet"
        if cache_path.exists():
            df = pd.read_parquet(cache_path)
            print(f"    💾 Cache: {len(df)} bars for {symbol}")
            return df
        return None

    def _save_ohlcv_cache(self, symbol: str, df: pd.DataFrame):
        """Save OHLCV data to local cache (with write lock)."""
        from src.strategies.ml_features import _acquire_cache_lock, _release_cache_lock
        OHLCV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = OHLCV_CACHE_DIR / f"{symbol}_5m.parquet"
        locked = _acquire_cache_lock(symbol)
        if locked:
            try:
                df.to_parquet(cache_path)
                print(f"    💾 Cached {len(df)} bars → {cache_path}")
            finally:
                _release_cache_lock(symbol)
        else:
            print(f"    ⏭ Cache write skipped for {symbol} (locked by another process)")

    def _fetch_ohlcv_range(self, symbol: str, start_ms: int) -> Optional[pd.DataFrame]:
        """Fetch 5m OHLCV from start_ms to now (incremental)."""
        end_ms = int(datetime.now().timestamp() * 1000)
        if start_ms >= end_ms:
            return None

        limit = 1000
        all_bars = []
        last_ts = start_ms
        while last_ts < end_ms:
            url = "https://fapi.binance.com/fapi/v1/klines"
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

        if not all_bars:
            return None

        df = pd.DataFrame(all_bars, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        for col in ['taker_buy_quote', 'quote_volume', 'trades', 'taker_buy_base']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume', 'taker_buy_quote', 'quote_volume', 'trades']]

    def _ensure_ohlcv_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get complete OHLCV data via incremental cache.

        First run: fetch OHLCV_FETCH_DAYS days and cache.
        Subsequent runs: fetch only new bars since last cache, append, re-cache.
        Always returns the full accumulated dataset for proper multi-TF features.
        """
        cached = self._get_cached_ohlcv(symbol)

        if cached is not None and len(cached) >= 1000:
            # Incremental: fetch only data newer than last cached bar
            latest_ts = int(cached.index[-1].timestamp() * 1000) + 1
            new_data = self._fetch_ohlcv_range(symbol, start_ms=latest_ts)

            if new_data is not None and len(new_data) > 0:
                print(f"    📡 Incremental: fetching {len(new_data)} new bars")
                # Deduplicate and merge
                combined = pd.concat([cached, new_data])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined.sort_index(inplace=True)
                self._save_ohlcv_cache(symbol, combined)
                return combined

            # No new data — use cache as-is
            return cached

        # First run (or cache too small): fetch full historical data
        print(f"    📡 Initial fetch: {OHLCV_FETCH_DAYS}d of 5m data for {symbol}...")
        df = self._fetch_5m_ohlcv(symbol, days=OHLCV_FETCH_DAYS)
        if df is not None and len(df) >= 1000:
            self._save_ohlcv_cache(symbol, df)
        return df

    def _load_models(self, symbol: str):
        """Load all models for a symbol. Cache models permanently; compute fresh features from live data."""
        if symbol in self._cache and 'groups' in self._cache[symbol]:
            # Models are cached — but features are stale (parquet), so we always recompute
            cached = self._cache[symbol]
            # Recompute fresh features
            fresh_features = self._compute_fresh_features(symbol)
            if fresh_features is not None:
                cached['features'] = fresh_features
                return cached
            # Fallback: try stale cache if fresh fails
            if 'features' in cached:
                print(f"    ⚠️ Fresh features failed, using stale for {symbol}")
                return cached
            return None

        # First load: compute fresh features + load models
        group_models = {}
        for tf in TF_GROUPS:
            # If model_side is specific ('long' or 'short'), only load that group
            if self.model_side != 'auto' and tf != self.model_side:
                continue
            models = self._load_tf_group(symbol, tf)
            if models:
                group_models[tf] = models

        if len(group_models) < 1:
            return None

        fresh_features = self._compute_fresh_features(symbol)
        if fresh_features is None:
            return None

        result = {
            'features': fresh_features,
            'groups': group_models,
        }
        self._cache[symbol] = result
        return result

    def _compute_fresh_features(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch live 5m data and compute all features for inference."""
        from src.strategies.ml_features import compute_5m_features_5tf

        # BUG FIX: Jangan strip 1000 prefix untuk Futures API!
        # fapi.binance.com menggunakan 1000PEPEUSDT sebagai symbol, BUKAN PEPEUSDT.
        # Spot API mapping tidak relevan karena kita pakai Futures API di semua path.
        # spot_symbol hanya untuk display/logging.
        spot_symbol = symbol
        if symbol.startswith("1000"):
            for prefix in ["1000", "10000", "100000"]:
                if symbol.startswith(prefix):
                    spot_symbol = symbol[len(prefix):]
                    break

        try:
            print(f"    📡 Loading OHLCV data for {symbol}...")
            # GUNAKAN symbol ASLI (1000PEPEUSDT) untuk Futures API!
            df_5m = self._ensure_ohlcv_data(symbol)
            if df_5m is None or len(df_5m) < 500:
                print(f"    ⚠️ Insufficient data for {symbol} (got {len(df_5m) if df_5m is not None else 0} rows)")
                return None

            print(f"    ✅ OHLCV: {len(df_5m)} bars (cached incremental)")
            print(f"    🔧 Computing features...")
            
            # Fetch 1h OHLCV directly (NO look-ahead from resampling)
            from src.strategies.ml_features import ensure_ohlcv_1h
            df_1h = ensure_ohlcv_1h(symbol, min_days=30)
            if df_1h is not None:
                print(f"    ✅ 1h: {len(df_1h)} bars (independent cache)")
            
            feat_df = compute_5m_features_5tf(df_5m, for_inference=True, df_1h=df_1h)

            if len(feat_df) == 0:
                print(f"    ⚠️ No feature rows for {symbol}")
                return None

            # Print latest timestamp for debugging
            latest = feat_df.index[-1]
            print(f"    ✅ {len(feat_df)} feature rows | Latest: {latest} | {len(feat_df.columns)} features")

            return feat_df

        except Exception as e:
            print(f"    ⚠️ Feature computation error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _load_tf_group(self, symbol: str, tf_group: str) -> Optional[List]:
        """Load models for one TF group. Supports 'full', 'long', 'short'."""
        models = []
        if tf_group in ('full', 'long', 'short'):
            # No prefixing for these groups
            meta = {}
            meta_path = MODEL_DIR / f"{symbol}_{tf_group}_meta.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)

            for seed in SEEDS:
                path = MODEL_DIR / f"{symbol}_{tf_group}_xgb_ens_{seed}.json"
                if tf_group == 'full':
                    # Fallback: old naming without prefix
                    if not path.exists():
                        path = MODEL_DIR / f"{symbol}_xgb_ens_{seed}.json"
                if not path.exists():
                    continue
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                # Extract feature names directly from the model
                mf = m.get_booster().feature_names
                if not mf:
                    mf = meta.get('model_features', {}).get(str(seed), meta.get('features', []))
                models.append((str(seed), m, mf))
        else:
            # Legacy: TF-specific models with prefix (deprecated)
            prefix = f"{tf_group}_"
            for seed in SEEDS:
                path = MODEL_DIR / f"{symbol}_{tf_group}_xgb_ens_{seed}.json"
                if not path.exists():
                    continue
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                model_features = m.get_booster().feature_names
                if model_features and not model_features[0].startswith(prefix):
                    model_features = [
                        f"{prefix}{f}" if not f.startswith(prefix) else f
                        for f in model_features
                    ]
                models.append((str(seed), m, model_features))

        return models if len(models) >= 2 else None

    def generate_signal(self, symbol: str) -> Optional[Dict]:
        """Generate a signal for a symbol using the latest data.

        Uses dual models: long ensemble (predicts UP > 0.5%) and short ensemble
        (predicts DOWN > 0.5%). Takes the higher-confidence signal.

        When model_side is 'long' or 'short' (calibration mode), computes only
        that side's proba.

        Returns:
            dict with keys: proba (winning side), signal (1=long, -1=short, 0=flat),
            long_proba, short_proba, or None if error
        """
        try:
            cached = self._load_models(symbol)
            if cached is None:
                return None

            feat_df = cached['features']
            groups = cached['groups']

            if len(feat_df) == 0:
                return None

            def compute_side_proba(side: str) -> float:
                """Compute ensemble proba for one side (long or short)."""
                if side not in groups:
                    return 0.5  # neutral if no models
                tf_models = groups[side]
                tf_probs = []
                for seed, m, mf in tf_models:
                    available = [c for c in mf if c in feat_df.columns]
                    if len(available) < 5:
                        continue
                    X = feat_df[available].fillna(0).clip(-10, 10).values
                    probs = m.predict_proba(X[-1:])[:, 1]
                    tf_probs.append(probs[0])
                if not tf_probs:
                    return 0.5
                raw = float(np.mean(tf_probs))
                return self._apply_calibration(raw, target_side=side)

            # Compute both side probas
            long_proba = compute_side_proba('long')
            short_proba = compute_side_proba('short')

            # Main signal: pick the higher confidence with per-symbol threshold
            threshold = get_symbol_threshold(symbol)
            if long_proba >= threshold and long_proba >= short_proba:
                # LONG signal — use long proba
                proba = long_proba
                signal = 1
            elif short_proba >= threshold:
                # SHORT signal — use short proba
                proba = short_proba
                signal = -1
            else:
                # No signal — use the higher of the two for monitoring
                proba = max(long_proba, short_proba)
                signal = 0

            return {
                'proba': proba,
                'signal': signal,
                'long_proba': long_proba,
                'short_proba': short_proba,
            }

        except Exception as e:
            print(f"    ⚠️ Signal error for {symbol}: {e}")
            return None
