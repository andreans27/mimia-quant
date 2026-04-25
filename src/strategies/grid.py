"""
Grid trading strategy for Mimia Quant.

Uses a grid of orders placed at regular price intervals to profit from
oscillating markets without requiring trend prediction.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import OrderSide, OrderType, TimeFrame


class GridStrategy(BaseStrategy):
    """
    Grid trading strategy.
    
    Places buy orders at levels below the current price and sell orders
    above it, creating a grid of orders. Profits from price oscillations
    as orders are filled at different grid levels.
    """
    
    def __init__(self, name: str = "grid", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the grid strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration with keys:
                - grid_levels: Number of grid levels (default: 10)
                - grid_spacing_pct: Spacing between levels as % (default: 1.0)
                - total_investment_pct: Total investment per grid (default: 0.5)
                - rebalance_threshold: Threshold to rebalance grid (default: 0.2)
                - profit_target_pct: Target profit per grid level % (default: 0.5)
        """
        default_config = {
            "enabled": True,
            "grid_levels": 10,
            "grid_spacing_pct": 1.0,
            "total_investment_pct": 0.5,
            "rebalance_threshold": 0.05,
            "profit_target_pct": 0.5,
            "cooldown_period_seconds": 60,
            "min_strength": 0.0,  # Grid doesn't need strength validation
            "position_size_pct": 0.05,
        }
        if config:
            default_config.update(config)
        
        super().__init__(name, default_config)
        
        self.grid_levels = self.config["grid_levels"]
        self.grid_spacing_pct = self.config["grid_spacing_pct"] / 100.0
        self.total_investment_pct = self.config["total_investment_pct"]
        self.rebalance_threshold = self.config["rebalance_threshold"]
        self.profit_target_pct = self.config["profit_target_pct"] / 100.0
        
        self._grid_orders: List[Order] = []
        self._grid_prices: List[float] = []
        self._filled_levels: Dict[int, int] = {}  # level_index -> fill_count
    
    def calculate_grid_prices(self, current_price: float) -> tuple:
        """
        Calculate grid price levels.
        
        Args:
            current_price: Current market price.
        
        Returns:
            Tuple of (lower_prices, upper_prices) as lists.
        """
        lower_prices = []
        upper_prices = []
        
        # Calculate lower grid levels (buy orders)
        for i in range(1, self.grid_levels + 1):
            level_price = current_price * (1 - self.grid_spacing_pct * i)
            lower_prices.append(level_price)
        
        # Calculate upper grid levels (sell orders)
        for i in range(1, self.grid_levels + 1):
            level_price = current_price * (1 + self.grid_spacing_pct * i)
            upper_prices.append(level_price)
        
        self._grid_prices = lower_prices + [current_price] + upper_prices
        
        return lower_prices, upper_prices
    
    def calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """
        Calculate price volatility.
        
        Args:
            prices: Price series.
            period: Period for volatility calculation.
        
        Returns:
            Volatility as a decimal (e.g., 0.02 for 2%).
        """
        if len(prices) < period:
            period = len(prices)
        
        returns = prices.pct_change().dropna()
        if len(returns) < 2:
            return self.grid_spacing_pct
        
        volatility = returns.tail(period).std()
        return float(volatility)
    
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
    
    def should_rebalance(self, current_price: float, reference_price: float) -> bool:
        """
        Check if grid should be rebalanced.
        
        Args:
            current_price: Current market price.
            reference_price: Reference price when grid was created.
        
        Returns:
            True if rebalancing is needed.
        """
        price_change = abs(current_price - reference_price) / reference_price
        return price_change > self.rebalance_threshold
    
    def analyze(self, symbol: str, data: Any) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        For grid strategy, this method:
        1. Analyzes volatility to adjust grid spacing
        2. Generates signals for grid rebalancing
        3. Creates signals when price crosses grid levels
        
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
            
            if len(closes) < 20:
                return None
        elif isinstance(data, pd.Series):
            closes = data
            if len(closes) < 20:
                return None
        else:
            return None
        
        current_price = float(closes.iloc[-1])
        reference_price = self.config.get("reference_price", current_price)
        
        # Calculate indicators
        ema_20 = self.calculate_ema(closes, 20)
        ema_50 = self.calculate_ema(closes, 50)
        current_ema_20 = float(ema_20.iloc[-1])
        current_ema_50 = float(ema_50.iloc[-1])
        volatility = self.calculate_volatility(closes)
        
        # Calculate grid prices
        lower_prices, upper_prices = self.calculate_grid_prices(current_price)
        
        # Determine signal side based on trend and volatility
        strength = 0.5
        side = OrderSide.BUY
        
        # Determine market condition
        if current_price < current_ema_20:
            # Below short-term EMA - potential upward oscillation
            trend = "ranging_down"
        elif current_price > current_ema_20:
            # Above short-term EMA - potential downward oscillation
            trend = "ranging_up"
        else:
            trend = "neutral"
        
        # Calculate grid proximity score
        min_distance = float('inf')
        for price in lower_prices + upper_prices:
            distance = abs(current_price - price) / current_price
            min_distance = min(min_distance, distance)
        
        # Generate signal based on proximity to grid levels
        if min_distance < 0.015:  # Within 1.5% of a grid level
            # Find closest level
            for i, price in enumerate(lower_prices):
                if abs(current_price - price) / current_price < 0.015:
                    side = OrderSide.BUY
                    strength = 0.7
                    break
            else:
                for i, price in enumerate(upper_prices):
                    if abs(current_price - price) / current_price < 0.015:
                        side = OrderSide.SELL
                        strength = 0.7
                        break
        
        # Always generate a signal with at least 0.3 strength if the trend is clear
        elif current_price > current_ema_20 and current_price > current_ema_50:
            # Clear uptrend
            strength = max(strength, 0.3)
            side = OrderSide.BUY
        elif current_price < current_ema_20 and current_price < current_ema_50:
            # Clear downtrend
            strength = max(strength, 0.3)
            side = OrderSide.SELL
        
        # Check for rebalancing need
        if self.should_rebalance(current_price, reference_price):
            self.config["reference_price"] = current_price
            strength = 1.0
            # Rebalancing signal - determine side based on price movement
            if current_price > reference_price:
                side = OrderSide.SELL
            else:
                side = OrderSide.BUY
        
        # Create signal
        signal = Signal(
            symbol=symbol,
            side=side,
            strength=strength,
            timestamp=datetime.now(),
            strategy_name=self.name,
            timeframe=TimeFrame.HOUR_1,
            indicators={
                "volatility": volatility,
                "ema_20": current_ema_20,
                "ema_50": current_ema_50,
                "min_grid_distance": min_distance,
                "trend": trend,
            },
            metadata={
                "price": current_price,
                "reference_price": reference_price,
                "lower_prices": lower_prices,
                "upper_prices": upper_prices,
                "needs_rebalance": self.should_rebalance(current_price, reference_price),
            },
        )
        
        # Grid doesn't use cooldown validation since it generates frequent signals
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
        # Each grid level gets equal portion of total investment
        per_level_size = self.total_investment_pct / self.grid_levels
        
        # Adjust based on signal strength
        adjusted_size = per_level_size * signal.strength
        
        # Cap at maximum position size per level
        max_size = 0.1  # 10% max per level
        return min(adjusted_size, max_size)
    
    def generate_grid_orders(self, symbol: str, current_price: float, 
                            portfolio_value: float) -> List[Order]:
        """
        Generate all orders for the grid.
        
        Args:
            symbol: Trading symbol.
            current_price: Current market price.
            portfolio_value: Total portfolio value.
        
        Returns:
            List of Order objects.
        """
        orders = []
        lower_prices, upper_prices = self.calculate_grid_prices(current_price)
        
        per_level_size = self.calculate_position_size(
            Signal(symbol=symbol, side=OrderSide.BUY, strength=0.5),
            portfolio_value
        )
        
        # Generate buy orders for lower levels
        for i, price in enumerate(lower_prices):
            order = Order(
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=per_level_size * portfolio_value / price,
                price=price,
                strategy_name=self.name,
                metadata={
                    "grid_level": -i - 1,
                    "grid_type": "buy",
                    "signal_id": None,
                },
            )
            orders.append(order)
        
        # Generate sell orders for upper levels
        for i, price in enumerate(upper_prices):
            order = Order(
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=per_level_size * portfolio_value / price,
                price=price,
                strategy_name=self.name,
                metadata={
                    "grid_level": i + 1,
                    "grid_type": "sell",
                    "signal_id": None,
                },
            )
            orders.append(order)
        
        self._grid_orders = orders
        return orders
