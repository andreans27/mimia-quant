"""
Multi-timeframe trading strategy for Mimia Quant.

Combines analysis across multiple timeframes to generate higher probability
trading signals. Uses trend alignment and momentum confluence.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import OrderSide, TimeFrame


class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-timeframe trading strategy.
    
    Analyzes price action and indicators across multiple timeframes
    to generate signals with higher confidence. Only trades when
    multiple timeframes are aligned.
    """
    
    def __init__(self, name: str = "multi_timeframe", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-timeframe strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration with keys:
                - primary_timeframe: Primary timeframe for signals (default: 1h)
                - secondary_timeframe: Secondary timeframe (default: 4h)
                - tertiary_timeframe: Tertiary timeframe (default: 1d)
                - alignment_threshold: Min timeframes that must align (default: 2)
                - trend_ma_period: MA period for trend detection (default: 50)
                - momentum_period: Period for momentum calculation (default: 14)
        """
        default_config = {
            "enabled": True,
            "primary_timeframe": "1h",
            "secondary_timeframe": "4h",
            "tertiary_timeframe": "1d",
            "alignment_threshold": 2,
            "trend_ma_period": 50,
            "momentum_period": 14,
            "cooldown_period_seconds": 600,
            "min_strength": 0.7,
            "position_size_pct": 0.12,
        }
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.primary_tf = TimeFrame(self.config["primary_timeframe"])
        self.secondary_tf = TimeFrame(self.config["secondary_timeframe"])
        self.tertiary_tf = TimeFrame(self.config["tertiary_timeframe"])
        self.alignment_threshold = self.config["alignment_threshold"]
        self.trend_ma_period = self.config["trend_ma_period"]
        self.momentum_period = self.config["momentum_period"]
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series.
            period: EMA period.
        
        Returns:
            EMA values as a pandas Series.
        """
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series.
            period: RSI period.
        
        Returns:
            RSI values as a pandas Series.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self, prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate price momentum.
        
        Args:
            prices: Price series.
            period: Momentum period.
        
        Returns:
            Momentum values as a pandas Series.
        """
        return prices.pct_change(periods=period)
    
    def analyze_timeframe(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single timeframe and return indicators.
        
        Args:
            data: DataFrame with OHLCV data.
        
        Returns:
            Dictionary of indicators for the timeframe.
        """
        closes = data["close"] if "close" in data.columns else data["Close"]
        highs = data["high"] if "high" in data.columns else data["High"]
        lows = data["low"] if "low" in data.columns else data["Low"]
        
        # Trend analysis
        ema_fast = self.calculate_ema(closes, 12)
        ema_slow = self.calculate_ema(closes, 26)
        ema_trend = self.calculate_ema(closes, self.trend_ma_period)
        
        # Momentum
        rsi = self.calculate_rsi(closes, self.momentum_period)
        momentum = self.calculate_momentum(closes, 10)
        
        current = {
            "price": float(closes.iloc[-1]),
            "ema_fast": float(ema_fast.iloc[-1]),
            "ema_slow": float(ema_slow.iloc[-1]),
            "ema_trend": float(ema_trend.iloc[-1]),
            "rsi": float(rsi.iloc[-1]),
            "momentum": float(momentum.iloc[-1]),
            "high_20": float(highs.tail(20).max()),
            "low_20": float(lows.tail(20).min()),
        }
        
        # Trend direction: 1 = bullish, -1 = bearish, 0 = neutral
        if current["price"] > current["ema_trend"]:
            current["trend"] = 1
        elif current["price"] < current["ema_trend"]:
            current["trend"] = -1
        else:
            current["trend"] = 0
        
        # RSI condition
        if current["rsi"] > 60:
            current["rsi_condition"] = "overbought"
        elif current["rsi"] < 40:
            current["rsi_condition"] = "oversold"
        else:
            current["rsi_condition"] = "neutral"
        
        # EMA alignment
        if current["ema_fast"] > current["ema_slow"]:
            current["ema_alignment"] = 1  # Bullish
        else:
            current["ema_alignment"] = -1  # Bearish
        
        # Price vs 20-period high/low
        if current["price"] > current["high_20"] * 0.98:
            current["near_high"] = True
        else:
            current["near_high"] = False
        
        if current["price"] < current["low_20"] * 1.02:
            current["near_low"] = True
        else:
            current["near_low"] = False
        
        return current
    
    def calculate_alignment_score(self, tf_analysis: Dict[str, Dict[str, Any]]) -> Tuple[int, float, str]:
        """
        Calculate alignment score across timeframes.
        
        Args:
            tf_analysis: Dictionary of timeframe analyses.
        
        Returns:
            Tuple of (alignment_count, strength, direction).
        """
        bullish_count = 0
        bearish_count = 0
        total_strength = 0.0
        direction = "neutral"
        
        for tf_name, analysis in tf_analysis.items():
            # Count trend alignments
            if analysis["trend"] == 1:
                bullish_count += 1
            elif analysis["trend"] == -1:
                bearish_count += 1
            
            # Count EMA alignments
            if analysis["ema_alignment"] == 1:
                bullish_count += 1
            elif analysis["ema_alignment"] == -1:
                bearish_count += 1
            
            # Count momentum alignments
            if analysis["momentum"] > 0:
                bullish_count += 1
            elif analysis["momentum"] < 0:
                bearish_count += 1
            
            # Add to total strength
            total_strength += abs(analysis["momentum"])
        
        # Normalize strength
        num_tfs = len(tf_analysis)
        if num_tfs > 0:
            avg_strength = total_strength / (num_tfs * 3)  # 3 indicators per tf
            strength = min(avg_strength * 50, 1.0)
        else:
            strength = 0.0
        
        # Determine direction
        if bullish_count > bearish_count + self.alignment_threshold:
            direction = "bullish"
        elif bearish_count > bullish_count + self.alignment_threshold:
            direction = "bearish"
        else:
            direction = "neutral"
        
        alignment_count = max(bullish_count, bearish_count)
        
        return alignment_count, strength, direction
    
    def analyze(self, symbol: str, data: Any) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        For multi-timeframe strategy, data should be a dict of DataFrames
        keyed by timeframe name, or a single DataFrame that will be used
        for all timeframes (for testing purposes).
        
        Args:
            symbol: Trading symbol.
            data: Market data - either dict of DataFrames or single DataFrame.
        
        Returns:
            Signal if generated, None otherwise.
        """
        if data is None:
            return None
        
        # Handle multi-timeframe data
        if isinstance(data, dict):
            tf_data = data
        elif isinstance(data, pd.DataFrame):
            # Use same data for all timeframes (simplified case)
            tf_data = {
                "primary": data,
                "secondary": data,
                "tertiary": data,
            }
        else:
            return None
        
        # Analyze each timeframe
        tf_analysis = {}
        
        min_required = self.trend_ma_period + 10
        
        for tf_name, tf_df in tf_data.items():
            if not isinstance(tf_df, pd.DataFrame):
                continue
            if "close" not in tf_df.columns and "Close" not in tf_df.columns:
                continue
            if len(tf_df) < min_required:
                continue
            
            tf_analysis[tf_name] = self.analyze_timeframe(tf_df)
        
        if len(tf_analysis) == 0:
            return None
        
        # Calculate alignment
        alignment_count, strength, direction = self.calculate_alignment_score(tf_analysis)
        
        # Need at least the threshold number of aligned indicators
        if alignment_count < self.alignment_threshold * 3:  # 3 indicators per tf
            return None
        
        # Determine signal side and strength
        if direction == "bullish":
            side = OrderSide.BUY
        elif direction == "bearish":
            side = OrderSide.SELL
        else:
            return None
        
        # Combine strength from alignment and timeframe analysis
        combined_strength = strength * alignment_count / 9  # Normalize
        
        if combined_strength < 0.3:
            return None
        
        # Get current price and indicators from primary timeframe
        primary_analysis = tf_analysis.get("primary", list(tf_analysis.values())[0])
        current_price = primary_analysis["price"]
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            side=side,
            strength=combined_strength,
            timestamp=datetime.now(),
            strategy_name=self.name,
            timeframe=self.primary_tf,
            indicators={
                "alignment_count": alignment_count,
                "direction": direction,
                "primary_trend": tf_analysis.get("primary", {}).get("trend", 0),
                "secondary_trend": tf_analysis.get("secondary", {}).get("trend", 0),
                "tertiary_trend": tf_analysis.get("tertiary", {}).get("trend", 0),
                "primary_rsi": primary_analysis.get("rsi", 50),
                "primary_momentum": primary_analysis.get("momentum", 0),
                "near_high": primary_analysis.get("near_high", False),
                "near_low": primary_analysis.get("near_low", False),
            },
            metadata={
                "price": current_price,
                "timeframe_analysis": {k: {kk: vv for kk, vv in v.items() 
                                          if kk not in ["high_20", "low_20"]} 
                                       for k, v in tf_analysis.items()},
            },
        )
        
        # Validate signal
        if not self.validate_signal(signal):
            return None
        
        return signal
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate position size for a signal.
        
        Multi-timeframe signals are considered higher confidence,
        so they may receive slightly larger position sizes.
        
        Args:
            signal: Trading signal.
            portfolio_value: Total portfolio value.
        
        Returns:
            Position size as a fraction of portfolio.
        """
        base_size = self.config.get("position_size_pct", 0.12)
        
        # Multi-timeframe alignment bonus
        alignment_count = signal.indicators.get("alignment_count", 0)
        alignment_bonus = min(alignment_count / 9, 0.5)  # Up to 50% bonus
        
        adjusted_size = base_size * signal.strength * (1 + alignment_bonus)
        
        # Cap at maximum position size
        max_size = 0.3  # 30% max for multi-timeframe trades
        return min(adjusted_size, max_size)
