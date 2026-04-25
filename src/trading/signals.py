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

from src.trading.state import MODEL_DIR, SEEDS, TF_GROUPS, THRESHOLD, FETCH_DAYS


class SignalGenerator:
    """Generates trading signals using the multi-TF XGBoost ensemble."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._cache: Dict[str, Any] = {}  # symbol -> loaded data

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
            url = f"https://api.binance.com/api/v3/klines"
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
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        return df[['open', 'high', 'low', 'close', 'volume']]

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
            models = self._load_tf_group(symbol, tf)
            if models:
                group_models[tf] = models

        if len(group_models) < 2:
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

        # Map 1000x symbols to spot symbols for OHLCV (Binance Spot API)
        spot_symbol = symbol
        if symbol.startswith("1000"):
            for prefix in ["1000", "10000", "100000"]:
                if symbol.startswith(prefix):
                    spot_symbol = symbol[len(prefix):]
                    break

        from datetime import datetime, timedelta

        try:
            print(f"    📡 Fetching live 5m data for {symbol} (Spot: {spot_symbol})...")
            df_5m = self._fetch_5m_ohlcv(spot_symbol, days=5)
            if df_5m is None or len(df_5m) < 500:
                print(f"    ⚠️ Insufficient data for {symbol} (got {len(df_5m) if df_5m is not None else 0} rows)")
                return None

            print(f"    🔧 Computing features...")
            feat_df = compute_5m_features_5tf(df_5m, for_inference=True)

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
        """Load models for one TF group."""
        models = []
        if tf_group == 'full':
            meta_path = MODEL_DIR / f"{symbol}_ensemble_meta.json"
            if not meta_path.exists():
                return None
            with open(meta_path) as f:
                meta = json.load(f)
            feature_cols = meta.get('features', meta.get('full_feature_set', []))

            for seed in SEEDS:
                path = MODEL_DIR / f"{symbol}_xgb_ens_{seed}.json"
                if not path.exists():
                    continue
                mf = meta.get('model_features', {}).get(str(seed), meta.get('features', []))
                m = xgb.XGBClassifier()
                m.load_model(str(path))
                models.append((str(seed), m, mf))
        else:
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

        Returns:
            dict with keys: proba, signal (1=long, -1=short, 0=flat),
            previous_proba, or None if error
        """
        # Handle 1000x symbols -> use spot symbol for OHLCV
        spot_symbol = symbol
        if symbol.startswith("1000"):
            # Map to spot symbol (e.g., 1000PEPEUSDT -> PEPEUSDT)
            for prefix in ["1000", "10000", "100000"]:
                if symbol.startswith(prefix):
                    remainder = symbol[len(prefix):]
                    spot_symbol = remainder
                    break

        try:
            cached = self._load_models(symbol)
            if cached is None:
                return None

            feat_df = cached['features']
            groups = cached['groups']

            # Get latest feature row
            latest_features = feat_df.iloc[-1:]
            if len(latest_features) == 0:
                return None

            # Compute probabilities from all groups
            group_probs = []
            for tf, models in groups.items():
                tf_probs = []
                for seed, m, mf in models:
                    available = [c for c in mf if c in feat_df.columns]
                    if len(available) < 5:
                        continue
                    X = feat_df[available].fillna(0).clip(-10, 10)
                    # Use last row
                    X_row = X.iloc[-1:].fillna(0)
                    probs = m.predict_proba(X_row)[:, 1]
                    tf_probs.append(probs[0])

                if tf_probs:
                    group_probs.append(np.mean(tf_probs))

            if len(group_probs) < 2:
                return None

            proba = float(np.mean(group_probs))

            # Get previous bar proba for cross detection
            prev_proba = None
            if len(feat_df) >= 2:
                prev_feat = feat_df.iloc[-2:-1]
                prev_group_probs = []
                for tf, models in groups.items():
                    tf_probs = []
                    for seed, m, mf in models:
                        available = [c for c in mf if c in feat_df.columns]
                        if len(available) < 5:
                            continue
                        X = feat_df[available].fillna(0).clip(-10, 10)
                        X_prev = X.iloc[-2:-1].fillna(0)
                        if len(X_prev) > 0:
                            probs = m.predict_proba(X_prev)[:, 1]
                            tf_probs.append(probs[0])
                    if tf_probs:
                        prev_group_probs.append(np.mean(tf_probs))
                if len(prev_group_probs) >= 2:
                    prev_proba = float(np.mean(prev_group_probs))

            # Determine signal: level-based (align with backtest)
            signal = 0  # flat
            if proba >= THRESHOLD:
                signal = 1  # LONG
            elif proba <= (1 - THRESHOLD):
                signal = -1  # SHORT

            return {
                'proba': proba,
                'signal': signal,
                'prev_proba': prev_proba,
            }

        except Exception as e:
            print(f"    ⚠️ Signal error for {symbol}: {e}")
            return None
