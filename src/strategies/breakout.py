"""
Breakout trading strategy for Mimia Quant.

Uses support/resistance levels and price breakout detection to identify
and trade strong momentum moves when price breaks out of consolidation.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import OrderSide, TimeFrame


class BreakoutStrategy(BaseStrategy):
    """
    Breakout trading strategy.
    
    Identifies consolidation zones and generates signals when price
    breaks out with strong momentum. Uses Donchian channels and
    breakout confirmation indicators.
    """
    
    def __init__(self, name: str = "breakout", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the breakout strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration with keys:
                - lookback_period: Period for breakout detection (default: 20)
                - breakour_threshold_pct: Breakout threshold % (default: 0.5)
                - volume_ma_period: Volume moving average period (default: 20)
                - volume_confirmation: Require volume confirmation (default: True)
                - atr_period: ATR period for stop calculation (default: 14)
                - consolidation_threshold: Max range for consolidation % (default: 2.0)
        """
        default_config = {
            "enabled": True,
            "lookback_period": 20,
            "breakout_threshold_pct": 0.3,
            "volume_ma_period": 20,
            "volume_confirmation": True,
            "atr_period": 14,
            "consolidation_threshold": 2.0,
            "cooldown_period_seconds": 300,
            "min_strength": 0.15,
            "position_size_pct": 0.15,
        }
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.lookback_period = self.config["lookback_period"]
        self.breakout_threshold_pct = self.config["breakout_threshold_pct"] / 100.0
        self.volume_ma_period = self.config["volume_ma_period"]
        self.volume_confirmation = self.config["volume_confirmation"]
        self.atr_period = self.config["atr_period"]
        self.consolidation_threshold = self.config["consolidation_threshold"] / 100.0
    
    def calculate_donchian_channels(self, highs: pd.Series, 
                                    lows: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian channels.
        
        Args:
            highs: High price series.
            lows: Low price series.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band).
        """
        upper = highs.rolling(window=self.lookback_period, min_periods=1).max()
        lower = lows.rolling(window=self.lookback_period, min_periods=1).min()
        middle = (upper + lower) / 2
        
        return upper, middle, lower
    
    def calculate_atr(self, highs: pd.Series, lows: pd.Series, 
                      closes: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            highs: High price series.
            lows: Low price series.
            closes: Close price series.
            period: ATR period (uses config default if None).
        
        Returns:
            ATR values as a pandas Series.
        """
        period = period or self.atr_period
        
        high_low = highs - lows
        high_close = (highs - closes.shift(1)).abs()
        low_close = (lows - closes.shift(1)).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_volume_ratio(self, volumes: pd.Series) -> pd.Series:
        """
        Calculate volume ratio vs moving average.
        
        Args:
            volumes: Volume series.
        
        Returns:
            Volume ratio as a pandas Series.
        """
        volume_ma = volumes.rolling(window=self.volume_ma_period, min_periods=1).mean()
        volume_ratio = volumes / volume_ma
        return volume_ratio
    
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
    
    def detect_consolidation(self, highs: pd.Series, lows: pd.Series, 
                             closes: pd.Series) -> Tuple[bool, float]:
        """
        Detect if price is in a consolidation zone.
        
        Args:
            highs: High price series.
            lows: Low price series.
            closes: Close price series.
        
        Returns:
            Tuple of (is_consolidating, consolidation_range_pct).
        """
        if len(closes) < self.lookback_period:
            return False, 0.0
        
        upper, _, lower = self.calculate_donchian_channels(highs, lows)
        
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        
        range_pct = (current_upper - current_lower) / current_lower
        
        is_consolidating = range_pct <= self.consolidation_threshold
        
        return is_consolidating, range_pct
    
    def analyze(self, symbol: str, data: Any) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Trading symbol.
            data: Market data (pandas DataFrame with OHLCV).
        
        Returns:
            Signal if generated, None otherwise.
        """
        if data is None or len(data) == 0:
            return None
        
        # Extract price data
        if isinstance(data, pd.DataFrame):
            if "close" not in data.columns and "Close" not in data.columns:
                return None
            
            closes = data["close"] if "close" in data.columns else data["Close"]
            highs = data["high"] if "high" in data.columns else data["High"]
            lows = data["low"] if "low" in data.columns else data["Low"]
            volumes = data["volume"] if "volume" in data.columns else data.get("Volume", pd.Series([1]*len(closes)))
            
            min_required = self.lookback_period + 10
            if len(closes) < min_required:
                return None
        else:
            return None
        
        current_price = float(closes.iloc[-1])
        
        # Calculate indicators
        upper, middle, lower = self.calculate_donchian_channels(highs, lows)
        atr = self.calculate_atr(highs, lows, closes)
        volume_ratio = self.calculate_volume_ratio(volumes)
        momentum = self.calculate_momentum(closes)
        
        current_upper = float(upper.iloc[-1])
        current_lower = float(lower.iloc[-1])
        current_middle = float(middle.iloc[-1])
        current_atr = float(atr.iloc[-1])
        current_volume_ratio = float(volume_ratio.iloc[-1])
        current_momentum = float(momentum.iloc[-1])
        
        # Calculate previous channel values
        prev_upper = float(upper.iloc[-2]) if len(upper) > 1 else current_upper
        prev_lower = float(lower.iloc[-2]) if len(lower) > 1 else current_lower
        
        # Detect consolidation
        is_consolidating, consolidation_range = self.detect_consolidation(highs, lows, closes)
        
        # Calculate breakout metrics
        upper_breakout = (current_price - current_upper) / current_upper
        lower_breakout = (current_lower - current_price) / current_lower
        
        # Determine signal
        strength = 0.0
        side = OrderSide.BUY
        breakout_type = "none"
        
        # Check for upward breakout
        if upper_breakout > self.breakout_threshold_pct:
            breakout_type = "up"
            
            # Volume confirmation (more lenient)
            if self.volume_confirmation and current_volume_ratio < 0.5:
                return None
            
            strength = min(upper_breakout * 10, 1.0)
            side = OrderSide.BUY
            
            # Momentum boost
            if current_momentum > 0.02:
                strength = min(strength * 1.2, 1.0)
        
        # Check for downward breakout
        elif lower_breakout > self.breakout_threshold_pct:
            breakout_type = "down"
            
            # Volume confirmation (more lenient)
            if self.volume_confirmation and current_volume_ratio < 0.5:
                return None
            
            strength = min(lower_breakout * 10, 1.0)
            side = OrderSide.SELL
            
            # Momentum boost
            if current_momentum < -0.02:
                strength = min(strength * 1.2, 1.0)
        
        # Consolidation breakout - both directions potential
        elif is_consolidating:
            # Check for breakout direction
            range_center = (current_upper + current_lower) / 2
            
            if current_price > range_center and closes.iloc[-1] > closes.iloc[-2]:
                breakout_type = "up_from_consolidation"
                strength = 0.6
                side = OrderSide.BUY
            elif current_price < range_center and closes.iloc[-1] < closes.iloc[-2]:
                breakout_type = "down_from_consolidation"
                strength = 0.6
                side = OrderSide.SELL
        
        # Momentum breakout: price moves > 1% in last 3 bars with increasing volume
        if breakout_type == "none":
            price_change_3 = abs(float(closes.iloc[-1]) - float(closes.iloc[-4])) / float(closes.iloc[-4]) if len(closes) > 4 else 0.0
            if price_change_3 > 0.01 and current_volume_ratio > 1.0:
                if closes.iloc[-1] > closes.iloc[-4]:
                    breakout_type = "momentum_up"
                    strength = 0.5
                    side = OrderSide.BUY
                else:
                    breakout_type = "momentum_down"
                    strength = 0.5
                    side = OrderSide.SELL
        
        # Check if signal meets minimum threshold
        if strength < 0.15:
            return None
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            timestamp=datetime.now(),
            strategy_name=self.name,
            timeframe=TimeFrame.HOUR_1,
            indicators={
                "upper_channel": current_upper,
                "lower_channel": current_lower,
                "middle_channel": current_middle,
                "atr": current_atr,
                "volume_ratio": current_volume_ratio,
                "momentum": current_momentum,
                "breakout_type": breakout_type,
                "is_consolidating": is_consolidating,
                "consolidation_range": consolidation_range,
                "upper_breakout": upper_breakout,
                "lower_breakout": lower_breakout,
            },
            metadata={
                "price": current_price,
                "channel_width": current_upper - current_lower,
            },
        )
        
        # Validate signal
        if not self.validate_signal(signal):
            return None
        
        return signal
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal.
            portfolio_value: Total portfolio value.
        
        Returns:
            Position size as a fraction of portfolio.
        """
        base_size = self.config.get("position_size_pct", 0.15)
        
        # Adjust based on signal strength and ATR volatility
        atr = signal.indicators.get("atr", 0)
        price = signal.metadata.get("price", 1)
        
        if atr > 0 and price > 0:
            atr_pct = atr / price
            # Reduce size for high volatility
            volatility_adjustment = min(1.0, 0.02 / atr_pct) if atr_pct > 0 else 1.0
        else:
            volatility_adjustment = 1.0
        
        adjusted_size = base_size * signal.strength * volatility_adjustment
        
        # Cap at maximum position size
        max_size = 0.25  # 25% max for breakout trades
        return min(adjusted_size, max_size)
