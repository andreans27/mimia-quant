"""
ML Strategy — uses trained XGBoost models to generate signals on 15m data.
Supports both long and short sides with confidence-based entry.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta

import xgboost as xgb

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.utils.binance_client import BinanceRESTClient
from src.strategies.ml_features import compute_technical_features


class MLStrategy(BaseStrategy):
    """
    Strategy that uses pre-trained XGBoost models on multi-timeframe features.
    - Fetches real-time 15m, 1h, 4h data
    - Computes 199+ features
    - Predicts probability of price increase in next 45m
    - Enters long when probability > threshold, short when probability < 1-threshold
    """
    
    def __init__(self, name: str = "ml_strategy", 
                 model_dir: str = "data/ml_models",
                 confidence_threshold: float = 0.55,
                 confidence_threshold_short: Optional[float] = None,
                 cooldown_candles: int = 4,
                 stop_loss_pct: float = 1.5,
                 take_profit_pct: float = 3.0,
                 max_risk_per_trade: float = 0.01):
        """
        Args:
            name: Strategy name
            model_dir: Directory with trained XGBoost models
            confidence_threshold: Minimum probability for LONG entry (0.5-1.0)
            confidence_threshold_short: Max probability for SHORT entry (0-0.5), 
                                        defaults to 1 - confidence_threshold
            cooldown_candles: Minimum 15m candles between signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_risk_per_trade: Max fraction of capital to risk
        """
        super().__init__(name=name)
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.confidence_threshold_short = confidence_threshold_short or (1 - confidence_threshold)
        self.cooldown_candles = cooldown_candles
        self._stop_loss_pct = stop_loss_pct
        self._take_profit_pct = take_profit_pct
        self.max_risk_per_trade = max_risk_per_trade
        
        # Load models per symbol
        self.models: Dict[str, xgb.Booster] = {}
        self.features: Dict[str, list] = {}
        self._load_models()
        
        # Cooldown tracker: symbol -> last signal time
        self._last_signal: Dict[str, int] = {}
        
        # Client for fetching data
        self._client = None
    
    def _load_models(self):
        """Load all trained models and their feature lists."""
        for model_path in self.model_dir.glob("*_xgb.json"):
            symbol = model_path.stem.replace("_xgb", "")
            meta_path = self.model_dir / f"{symbol}_xgb_meta.json"
            
            if not meta_path.exists():
                print(f"  ⚠️ No metadata for {symbol}, skipping")
                continue
            
            # Load model
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            
            # Load metadata for feature list
            import json
            with open(meta_path) as f:
                meta = json.load(f)
            
            self.models[symbol] = model
            self.features[symbol] = meta.get('features', [])
            print(f"  Loaded model for {symbol}: {len(self.features[symbol])} features")
    
    @property
    def client(self):
        if self._client is None:
            self._client = BinanceRESTClient(testnet=True)
        return self._client
    
    def _fetch_multi_tf_data(self, symbol: str) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """Fetch 15m, 1h, 4h data for feature computation."""
        from src.strategies.ml_features import _fetch_all_klines
        from datetime import datetime
        
        end = datetime.now()
        start = end - timedelta(hours=120)  # 5 days for indicator warmup
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        
        try:
            df_15m = _fetch_all_klines(self.client, symbol, "15m", start_ms, end_ms)
            df_1h = _fetch_all_klines(self.client, symbol, "1h", start_ms, end_ms)
            df_4h = _fetch_all_klines(self.client, symbol, "4h", start_ms, end_ms)
        except Exception as e:
            print(f"  ⚠️ Failed to fetch {symbol} data: {e}")
            return None
        
        if df_15m is None or df_1h is None or df_4h is None:
            return None
        
        return df_15m, df_1h, df_4h
    
    def _compute_features(self, symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute aligned multi-timeframe features for latest candle."""
        from src.strategies.ml_features import compute_multi_timeframe_features
        
        try:
            feat_df = compute_multi_timeframe_features(df_15m, df_1h, df_4h, target_candle=3)
            
            if feat_df is None or len(feat_df) == 0:
                return None
            
            # Get only the last row (latest candle)
            return feat_df
            
        except Exception as e:
            print(f"  ⚠️ Feature computation failed for {symbol}: {e}")
            return None
    
    def _predict(self, symbol: str, features_df: pd.DataFrame) -> Optional[Tuple[float, pd.Series]]:
        """Get model prediction for the latest candle."""
        if symbol not in self.models:
            return None
        
        model = self.models[symbol]
        feature_list = self.features[symbol]
        
        # Get latest row
        latest = features_df.iloc[-1:]
        
        # Ensure all required features exist
        missing = [f for f in feature_list if f not in latest.columns]
        if missing:
            print(f"  ⚠️ Missing {len(missing)} features for {symbol}")
            for col in missing:
                latest[col] = 0.0
        
        # Select and order features
        X = latest[feature_list].fillna(0).clip(-10, 10)
        
        # Predict
        proba = model.predict_proba(X)[0, 1]
        return proba, latest.iloc[0]
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on ML model predictions.
        
        Args:
            symbol: Trading pair
            data: OHLCV DataFrame (needed for interface compatibility)
            
        Returns:
            List of Signal objects
        """
        if symbol not in self.models:
            return []
        
        # Cooldown check
        current_candle = len(data) if data is not None else 0
        last_signal = self._last_signal.get(symbol, -self.cooldown_candles * 2)
        if current_candle - last_signal < self.cooldown_candles and last_signal > 0:
            return []
        
        # Fetch multi-timeframe data
        tf_data = self._fetch_multi_tf_data(symbol)
        if tf_data is None:
            return []
        
        df_15m, df_1h, df_4h = tf_data
        if len(df_15m) < 100:
            return []
        
        # Compute features
        feat_df = self._compute_features(symbol, df_15m, df_1h, df_4h)
        if feat_df is None or len(feat_df) < 1:
            return []
        
        # Get prediction
        result = self._predict(symbol, feat_df)
        if result is None:
            return []
        
        proba, latest_row = result
        current_price = float(df_15m["close"].iloc[-1])
        
        # Determine signal
        signals = []
        
        # LONG signal
        if proba >= self.confidence_threshold:
            strength = min(1.0, (proba - self.confidence_threshold) / 0.2)
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                entry_price=current_price,
                stop_loss=current_price * (1 - self._stop_loss_pct / 100),
                take_profit=current_price * (1 + self._take_profit_pct / 100),
                confidence=strength,
                reason=f"ML LONG (p={proba:.3f})"
            ))
            self._last_signal[symbol] = current_candle
            
        # SHORT signal
        elif proba <= self.confidence_threshold_short:
            strength = min(1.0, (self.confidence_threshold_short - proba) / 0.2)
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                entry_price=current_price,
                stop_loss=current_price * (1 + self._stop_loss_pct / 100),
                take_profit=current_price * (1 - self._take_profit_pct / 100),
                confidence=strength,
                reason=f"ML SHORT (p={proba:.3f})"
            ))
            self._last_signal[symbol] = current_candle
        
        return signals


# ─── Backtest-compatible version that uses cached features ───
class MLBacktestStrategy(BaseStrategy):
    """
    Strategy for backtesting ML models on historical 15m data.
    Uses pre-computed features from prepare_ml_dataset.
    Falls back to compute_technical_features if cached features not available.
    """
    
    def __init__(self, name: str = "ml_backtest",
                 model_dir: str = "data/ml_models",
                 confidence_threshold: float = 0.55,
                 confidence_threshold_short: Optional[float] = None,
                 cooldown_candles: int = 4,
                 stop_loss_pct: float = 1.5,
                 take_profit_pct: float = 2.0,
                 max_risk_per_trade: float = 0.01):
        super().__init__(name=name)
        self.model_dir = Path(model_dir)
        self.confidence_threshold = confidence_threshold
        self.confidence_threshold_short = confidence_threshold_short or (1 - confidence_threshold)
        self.cooldown_candles = cooldown_candles
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_risk_per_trade = max_risk_per_trade
        
        # Load models
        self.models: Dict[str, xgb.XGBClassifier] = {}
        self.features: Dict[str, list] = {}
        self._load_models()
        
        self._last_signal_bar: Dict[str, int] = {}
        self._cached_features: Dict[str, pd.DataFrame] = {}
    
    def _load_models(self):
        import json
        for model_path in self.model_dir.glob("*_xgb.json"):
            symbol = model_path.stem.replace("_xgb", "")
            meta_path = self.model_dir / f"{symbol}_xgb_meta.json"
            if not meta_path.exists():
                continue
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            with open(meta_path) as f:
                meta = json.load(f)
            self.models[symbol] = model
            self.features[symbol] = meta.get('features', [])
    
    def set_features(self, symbol: str, feat_df: pd.DataFrame):
        """Pre-load features for faster backtesting."""
        self._cached_features[symbol] = feat_df
    
    def generate_signals(self, symbol: str, data: pd.DataFrame) -> List[Signal]:
        if symbol not in self.models:
            return []
        
        # Determine current bar index
        bar_idx = len(data)
        last_signal = self._last_signal_bar.get(symbol, -self.cooldown_candles * 2)
        if bar_idx - last_signal < self.cooldown_candles and last_signal > 0:
            return []
        
        # Use cached features (pre-computed from prepare_ml_dataset)
        feat_df = self._cached_features.get(symbol)
        if feat_df is None:
            return []
        
        # Find the row corresponding to this bar
        # The data index should overlap with feat_df index
        if len(data) > 0 and data.index[-1] in feat_df.index:
            # Find position in feat_df
            idx_loc = feat_df.index.get_loc(data.index[-1])
            if isinstance(idx_loc, slice):
                return []
            row = feat_df.iloc[idx_loc:idx_loc+1]
        else:
            return []
        
        # Get features
        feature_list = self.features[symbol]
        missing = [f for f in feature_list if f not in row.columns]
        if missing:
            for col in missing:
                row[col] = 0.0
        
        X = row[feature_list].fillna(0).clip(-10, 10)
        
        # Predict
        proba = float(self.models[symbol].predict_proba(X)[0, 1])
        
        # Current price
        price = float(data["close"].iloc[-1])
        
        signals = []
        
        if proba >= self.confidence_threshold:
            strength = min(1.0, (proba - self.confidence_threshold) / 0.2)
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                entry_price=price,
                stop_loss=price * (1 - self.stop_loss_pct / 100),
                take_profit=price * (1 + self.take_profit_pct / 100),
                confidence=strength,
                reason=f"ML LONG (p={proba:.3f})"
            ))
            self._last_signal_bar[symbol] = bar_idx
            
        elif proba <= self.confidence_threshold_short:
            strength = min(1.0, (self.confidence_threshold_short - proba) / 0.2)
            signals.append(Signal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                entry_price=price,
                stop_loss=price * (1 + self.stop_loss_pct / 100),
                take_profit=price * (1 - self.take_profit_pct / 100),
                confidence=strength,
                reason=f"ML SHORT (p={proba:.3f})"
            ))
            self._last_signal_bar[symbol] = bar_idx
        
        return signals
