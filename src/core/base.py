"""
Base classes for Mimia Quant Trading System.

Provides abstract base classes for strategies, handlers, and other
core components with common functionality and interface definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic
import uuid

from .constants import (
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
    TimeFrame,
)
from .logging import get_logger, TradingLogger


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Price:
    """Price data with timestamp."""
    symbol: str
    price: float
    timestamp: datetime = field(default_factory=datetime.now)
    volume: float = 0.0
    exchange: str = "binance"
    
    @property
    def age_ms(self) -> float:
        """Age of price data in milliseconds."""
        return (datetime.now() - self.timestamp).total_seconds() * 1000


@dataclass
class Order:
    """Order representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    filled_quantity: float = 0.0
    price: Optional[float] = None
    avg_fill_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_open(self) -> bool:
        """Check if order is open."""
        return self.status in (OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "price": self.price,
            "avg_fill_price": self.avg_fill_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }


@dataclass
class Position:
    """Position representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: PositionSide = PositionSide.NEUTRAL
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity != 0 and self.side != PositionSide.NEUTRAL
    
    @property
    def pnl_pct(self) -> float:
        """PnL as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100
    
    def update_price(self, price: float) -> None:
        """Update current price and recalculate PnL."""
        self.current_price = price
        self._calculate_pnl()
        self.updated_at = datetime.now()
    
    def _calculate_pnl(self) -> None:
        """Calculate unrealized PnL."""
        if self.entry_price == 0 or self.quantity == 0:
            self.unrealized_pnl = 0.0
            return
        
        price_diff = self.current_price - self.entry_price
        if self.side == PositionSide.SHORT:
            price_diff = -price_diff
        
        self.unrealized_pnl = price_diff * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "strategy_name": self.strategy_name,
            "metadata": self.metadata,
        }


@dataclass
class Signal:
    """Trading signal representation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    side: OrderSide = OrderSide.BUY
    strength: float = 1.0  # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    strategy_name: str = ""
    timeframe: TimeFrame = TimeFrame.HOUR_1
    indicators: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Check if signal is valid for trading."""
        return self.strength >= 0.6


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must inherit from this class and implement
    the required methods.
    
    Attributes:
        name: Strategy name.
        config: Strategy configuration dictionary.
        logger: Logger instance for the strategy.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            name: Strategy name.
            config: Strategy configuration dictionary.
        """
        self.name = name
        self.config = config
        self.enabled = config.get("enabled", True)
        self.logger = TradingLogger(strategy_name=name)
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._last_signal_time: Dict[str, datetime] = {}
    
    @abstractmethod
    def analyze(self, symbol: str, data: Any) -> Optional[Signal]:
        """
        Analyze market data and generate trading signals.
        
        Args:
            symbol: Trading symbol.
            data: Market data to analyze.
        
        Returns:
            Signal if generated, None otherwise.
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate position size for a signal.
        
        Args:
            signal: Trading signal.
            portfolio_value: Total portfolio value.
        
        Returns:
            Position size as a fraction of portfolio.
        """
        pass
    
    def on_signal(self, signal: Signal) -> Optional[Order]:
        """
        Handle a generated trading signal.
        
        Default implementation creates a market order.
        Override for custom order handling.
        
        Args:
            signal: Trading signal.
        
        Returns:
            Order if created, None otherwise.
        """
        if not signal.is_valid:
            return None
        
        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=1.0,  # Will be sized by executor
            strategy_name=self.name,
            metadata={"signal_id": signal.id},
        )
        self._orders[order.id] = order
        return order
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all positions for this strategy."""
        return self._positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self._positions.get(symbol)
    
    def update_position(self, position: Position) -> None:
        """Update position in strategy tracking."""
        if position.quantity == 0:
            self._positions.pop(position.symbol, None)
        else:
            self._positions[position.symbol] = position
    
    def validate_signal(self, signal: Signal) -> bool:
        """
        Validate if a signal should be executed.
        
        Args:
            signal: Trading signal to validate.
        
        Returns:
            True if signal passes validation.
        """
        # Check cooldown
        if signal.symbol in self._last_signal_time:
            last_time = self._last_signal_time[signal.symbol]
            cooldown = self.config.get("cooldown_period_seconds", 300)
            if (datetime.now() - last_time).total_seconds() < cooldown:
                self.logger.debug(f"Signal rejected: cooldown period active for {signal.symbol}")
                return False
        
        # Check minimum strength
        min_strength = self.config.get("min_strength", 0.6)
        if signal.strength < min_strength:
            self.logger.debug(f"Signal rejected: strength {signal.strength} < {min_strength}")
            return False
        
        return True
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name={self.name}, enabled={self.enabled})>"


class BaseExecutor(ABC):
    """
    Abstract base class for order execution handlers.
    
    Handles order submission, tracking, and fills.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the executor.
        
        Args:
            config: Executor configuration.
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._pending_orders: Dict[str, Order] = {}
    
    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """
        Submit an order to the exchange.
        
        Args:
            order: Order to submit.
        
        Returns:
            Updated order with exchange ID.
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel.
        
        Returns:
            True if cancelled successfully.
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get current order status.
        
        Args:
            order_id: Order ID.
        
        Returns:
            Current order status.
        """
        pass
    
    async def sync_orders(self) -> None:
        """Synchronize local order state with exchange."""
        for order_id in list(self._pending_orders.keys()):
            status = await self.get_order_status(order_id)
            order = self._pending_orders[order_id]
            order.status = status
            order.updated_at = datetime.now()
            
            if status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED):
                self._pending_orders.pop(order_id, None)


class BaseRiskManager(ABC):
    """
    Abstract base class for risk management.
    
    Handles position sizing, stop losses, and risk checks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize risk manager.
        
        Args:
            config: Risk management configuration.
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def check_order_risk(self, order: Order, portfolio_value: float) -> bool:
        """
        Check if an order passes risk checks.
        
        Args:
            order: Order to check.
            portfolio_value: Current portfolio value.
        
        Returns:
            True if order passes risk checks.
        """
        pass
    
    @abstractmethod
    def calculate_stop_loss(self, position: Position, order_side: OrderSide) -> Optional[float]:
        """
        Calculate stop loss price for a position.
        
        Args:
            position: Current position.
            order_side: Side of the new order.
        
        Returns:
            Stop loss price or None.
        """
        pass
    
    @abstractmethod
    def calculate_take_profit(self, position: Position, order_side: OrderSide) -> Optional[float]:
        """
        Calculate take profit price for a position.
        
        Args:
            position: Current position.
            order_side: Side of the new order.
        
        Returns:
            Take profit price or None.
        """
        pass


class BaseDataHandler(ABC):
    """
    Abstract base class for market data handling.
    
    Provides interface for fetching and processing market data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data handler.
        
        Args:
            config: Data handler configuration.
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def get_price(self, symbol: str) -> Price:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading symbol.
        
        Returns:
            Current Price object.
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: TimeFrame,
        limit: int = 100,
    ) -> List[Price]:
        """
        Get historical price data.
        
        Args:
            symbol: Trading symbol.
            timeframe: Data timeframe.
            limit: Number of data points.
        
        Returns:
            List of Price objects.
        """
        pass
    
    @abstractmethod
    async def subscribe_price(self, symbol: str, callback) -> None:
        """
        Subscribe to real-time price updates.
        
        Args:
            symbol: Trading symbol.
            callback: Callback function for price updates.
        """
        pass
