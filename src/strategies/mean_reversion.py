"""
Mean reversion trading strategy for Mimia Quant.

Uses Bollinger Bands and z-score of price deviation from moving average
to identify overbought/oversold conditions and generate reversal signals.
"""

from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import OrderSide, TimeFrame


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.
    
    Generates signals based on Bollinger Bands and z-score of price
    deviation. Buys when price is below the lower band (oversold),
    sells when price is above the upper band (overbought).
    """
    
    def __init__(self, name: str = "mean_reversion", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the mean reversion strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration with keys:
                - bollinger_period: Period for Bollinger Bands (default: 20)
                - bollinger_std: Standard deviations for bands (default: 2.0)
                - zscore_period: Period for z-score calculation (default: 20)
                - zscore_threshold: Z-score threshold for signals (default: 2.0)
                - entry_threshold: Entry threshold (default: 1.5)
                - exit_threshold: Exit threshold (default: 0.5)
        """
        default_config = {
            "enabled": True,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "zscore_period": 20,
            "zscore_threshold": 2.0,
            "entry_threshold": 1.5,
            "exit_threshold": 0.5,
            "cooldown_period_seconds": 600,
            "min_strength": 0.6,
            "position_size_pct": 0.1,
        }
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.bollinger_period = self.config["bollinger_period"]
        self.bollinger_std = self.config["bollinger_std"]
        self.zscore_period = self.config["zscore_period"]
        self.zscore_threshold = self.config["zscore_threshold"]
        self.entry_threshold = self.config["entry_threshold"]
        self.exit_threshold = self.config["exit_threshold"]
    
    def calculate_bollinger_bands(self, prices: pd.Series) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Price series.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band) as pandas Series.
        """
        middle = prices.rolling(window=self.bollinger_period, min_periods=1).mean()
        std = prices.rolling(window=self.bollinger_period, min_periods=1).std()
        
        upper = middle + (std * self.bollinger_std)
        lower = middle - (std * self.bollinger_std)
        
        return upper, middle, lower
    
    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """
        Calculate z-score of price deviation from moving average.
        
        Args:
            prices: Price series.
        
        Returns:
            Z-score values as a pandas Series.
        """
        rolling_mean = prices.rolling(window=self.zscore_period, min_periods=1).mean()
        rolling_std = prices.rolling(window=self.zscore_period, min_periods=1).std()
        
        zscore = (prices - rolling_mean) / rolling_std
        return zscore
    
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
    
    def analyze(self, symbol: str, data: Any) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Trading symbol.
            data: Market data (pandas DataFrame with OHLCV or pandas Series of closes).
        
        Returns:
            Signal if generated, None otherwise.
        """
        if data is None or len(data) == 0:
            return None
        
        # Extract price data
        if isinstance(data, pd.DataFrame):
            if "close" in data.columns:
                closes = data["close"]
            elif "Close" in data.columns:
                closes = data["Close"]
            else:
                return None
            
            min_required = max(self.bollinger_period, self.zscore_period) + 10
            if len(closes) < min_required:
                return None
        elif isinstance(data, pd.Series):
            closes = data
            min_required = max(self.bollinger_period, self.zscore_period) + 10
            if len(closes) < min_required:
                return None
        else:
            return None
        
        # Calculate indicators
        upper, middle, lower = self.calculate_bollinger_bands(closes)
        zscore = self.calculate_zscore(closes)
        rsi = self.calculate_rsi(closes)
        
        current_price = closes.iloc[-1]
        current_upper = upper.iloc[-1]
        current_lower = lower.iloc[-1]
        current_middle = middle.iloc[-1]
        current_zscore = zscore.iloc[-1]
        current_rsi = rsi.iloc[-1]
        
        # Calculate band position (0 = at lower band, 100 = at upper band)
        band_range = current_upper - current_lower
        if band_range > 0:
            band_position = (current_price - current_lower) / band_range * 100
        else:
            band_position = 50
        
        # Calculate signal strength
        strength = 0.0
        side = OrderSide.BUY
        
        # Buy signal: price below lower band or very negative zscore
        # Sell signal: price above upper band or very positive zscore
        if current_zscore < -self.entry_threshold:
            # Strong oversold - buy signal
            strength = min(abs(current_zscore) / self.zscore_threshold, 1.0)
            side = OrderSide.BUY
        elif current_zscore > self.entry_threshold:
            # Strong overbought - sell signal
            strength = min(abs(current_zscore) / self.zscore_threshold, 1.0)
            side = OrderSide.SELL
        elif band_position < 10:
            # Price near lower band - potential buy
            strength = 0.6
            side = OrderSide.BUY
        elif band_position > 90:
            # Price near upper band - potential sell
            strength = 0.6
            side = OrderSide.SELL
        else:
            # Check for mean reversion from moderate levels
            if current_zscore < -self.exit_threshold and current_zscore > -self.entry_threshold:
                # RSI confirmation for buy
                if current_rsi < 40:
                    strength = 0.5
                    side = OrderSide.BUY
            elif current_zscore > self.exit_threshold and current_zscore < self.entry_threshold:
                # RSI confirmation for sell
                if current_rsi > 60:
                    strength = 0.5
                    side = OrderSide.SELL
        
        # Check if signal meets minimum threshold
        if strength < 0.3:
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
                "zscore": current_zscore,
                "band_position": band_position,
                "rsi": current_rsi,
                "upper_band": current_upper,
                "lower_band": current_lower,
                "middle_band": current_middle,
            },
            metadata={
                "price": current_price,
                "band_range": band_range,
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
        base_size = self.config.get("position_size_pct", 0.1)
        
        # Inverse position sizing: stronger signal = larger position
        adjusted_size = base_size * signal.strength
        
        # Cap at maximum position size
        max_size = 0.2  # 20% max
        return min(adjusted_size, max_size)
