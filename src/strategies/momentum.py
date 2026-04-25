"""
Momentum trading strategy for Mimia Quant.

Uses RSI and price momentum indicators to identify strong trends
and generate trading signals.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import OrderSide, TimeFrame


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.
    
    Generates signals based on RSI and price momentum. Buys when momentum
    is strong and RSI indicates upward pressure, sells when momentum
    reverses or becomes overbought/oversold.
    """
    
    def __init__(self, name: str = "momentum", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the momentum strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration with keys:
                - rsi_period: Period for RSI calculation (default: 14)
                - rsi_overbought: Overbought threshold (default: 70)
                - rsi_oversold: Oversold threshold (default: 30)
                - momentum_period: Period for momentum calculation (default: 20)
                - min_momentum: Minimum momentum threshold (default: 0.02)
        """
        default_config = {
            "enabled": True,
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "momentum_period": 20,
            "min_momentum": 0.005,
            "cooldown_period_seconds": 300,
            "min_strength": 0.15,
            "position_size_pct": 0.1,
        }
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.rsi_period = self.config["rsi_period"]
        self.rsi_overbought = self.config["rsi_overbought"]
        self.rsi_oversold = self.config["rsi_oversold"]
        self.momentum_period = self.config["momentum_period"]
        self.min_momentum = self.config["min_momentum"]
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series.
            period: RSI period (uses config default if None).
        
        Returns:
            RSI values as a pandas Series.
        """
        period = period or self.rsi_period
        
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_momentum(self, prices: pd.Series, period: int = None) -> pd.Series:
        """
        Calculate price momentum as percentage change.
        
        Args:
            prices: Price series.
            period: Momentum period (uses config default if None).
        
        Returns:
            Momentum values as a pandas Series.
        """
        period = period or self.momentum_period
        
        momentum = prices.pct_change(periods=period)
        return momentum
    
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
            
            if len(closes) < max(self.rsi_period, self.momentum_period) + 10:
                return None
        elif isinstance(data, pd.Series):
            closes = data
            if len(closes) < max(self.rsi_period, self.momentum_period) + 10:
                return None
        else:
            return None
        
        # Calculate indicators
        rsi = self.calculate_rsi(closes)
        momentum = self.calculate_momentum(closes)
        ema_fast = self.calculate_ema(closes, 12)
        ema_slow = self.calculate_ema(closes, 26)
        
        current_rsi = rsi.iloc[-1]
        current_momentum = momentum.iloc[-1]
        current_price = closes.iloc[-1]
        current_ema_fast = ema_fast.iloc[-1]
        current_ema_slow = ema_slow.iloc[-1]
        prev_momentum = momentum.iloc[-2] if len(momentum) > 1 else 0.0
        prev_ema_fast = ema_fast.iloc[-2] if len(ema_fast) > 1 else current_ema_fast
        prev_ema_slow = ema_slow.iloc[-2] if len(ema_slow) > 1 else current_ema_slow
        
        # Calculate signal strength based on multiple factors
        strength = 0.0
        side = OrderSide.BUY
        
        # RSI-based signal (trend-confirming: overbought = strength, oversold = weakness)
        rsi_signal = 0.0
        if current_rsi < self.rsi_oversold:
            # Oversold = weakness → sell signal
            rsi_signal = -(self.rsi_oversold - current_rsi) / self.rsi_oversold
        elif current_rsi > self.rsi_overbought:
            # Overbought = strength → buy signal
            rsi_signal = (current_rsi - self.rsi_overbought) / (100 - self.rsi_overbought)
        
        # Momentum-based signal
        momentum_signal = 0.0
        if current_momentum > self.min_momentum:
            momentum_signal = min(current_momentum / 0.1, 1.0)
        elif current_momentum < -self.min_momentum:
            momentum_signal = max(current_momentum / 0.1, -1.0)
        
        # Momentum direction change detection
        momentum_change_boost = 0.0
        if prev_momentum < 0 and current_momentum > 0:
            momentum_change_boost = 0.3  # Bullish momentum reversal
        elif prev_momentum > 0 and current_momentum < 0:
            momentum_change_boost = -0.3  # Bearish momentum reversal
        
        # EMA crossover detection (1.0 for fresh crossover, 0.5 for already crossed, 0 for no cross)
        ema_signal = 0.0
        if prev_ema_fast <= prev_ema_slow and current_ema_fast > current_ema_slow:
            ema_signal = 1.0  # Fresh bullish crossover
        elif prev_ema_fast >= prev_ema_slow and current_ema_fast < current_ema_slow:
            ema_signal = -1.0  # Fresh bearish crossover
        elif current_ema_fast > current_ema_slow:
            ema_signal = 0.5  # Already crossed bullish
        elif current_ema_fast < current_ema_slow:
            ema_signal = -0.5  # Already crossed bearish
        
        # Combine signals with weights (reduced EMA weight since crossover is binary)
        combined = (rsi_signal * 0.25 + momentum_signal * 0.25 + ema_signal * 0.35 + momentum_change_boost * 0.15)
        
        # Determine side and strength
        if combined > 0:
            side = OrderSide.BUY
            strength = min(abs(combined), 1.0)
        elif combined < 0:
            side = OrderSide.SELL
            strength = min(abs(combined), 1.0)
        else:
            return None
        
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
                "rsi": current_rsi,
                "momentum": current_momentum,
                "ema_fast": current_ema_fast,
                "ema_slow": current_ema_slow,
                "rsi_signal": rsi_signal,
                "momentum_signal": momentum_signal,
                "ema_signal": ema_signal,
            },
            metadata={
                "price": current_price,
                "combined_score": combined,
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
        
        # Adjust position size based on signal strength
        adjusted_size = base_size * signal.strength
        
        # Cap at maximum position size
        max_size = 0.2  # 20% max
        return min(adjusted_size, max_size)
