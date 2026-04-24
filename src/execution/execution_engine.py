"""
Mimia Quant Trading System - Execution Engine

Execution engine for order management, trade execution, and broker integration.
Handles order lifecycle, fills, rejections, and integrates with risk management.

Features:
- Order management (create, modify, cancel)
- Multiple order types (market, limit, stop, stop-limit, trailing)
- Risk-integrated order validation
- Position tracking and management
- Trade execution and confirmation
- Slippage and fee modeling
"""

import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading
import random

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    TRAILING_STOP = "TRAILING_STOP_MARKET"


class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PARTIALLY_CANCELLED = "PARTIALLY_CANCELLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(Enum):
    """Position side for futures"""
    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


@dataclass
class Order:
    """
    Order data class representing a trading order.
    """
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trailing_delta: Optional[float] = None
    time_in_force: str = "GTC"
    reduce_only: bool = False
    position_side: Optional[PositionSide] = None
    
    # Order identification
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    client_order_id: Optional[str] = None
    
    # Status tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = "USDT"
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    
    # Strategy metadata
    strategy_name: Optional[str] = None
    session_id: Optional[str] = None
    signal_id: Optional[str] = None
    
    # Risk metadata
    risk_check_passed: bool = False
    risk_adjustment_factor: float = 1.0
    
    # Error handling
    error_message: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    def __post_init__(self):
        """Initialize derived fields"""
        self.remaining_quantity = self.quantity
        if self.client_order_id is None:
            self.client_order_id = f"{self.symbol}_{self.order_id}"
    
    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active (not terminal)"""
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def is_terminal(self) -> bool:
        """Check if order is in terminal state"""
        return self.status in [
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        ]
    
    @property
    def fill_percentage(self) -> float:
        """Get fill percentage"""
        if self.quantity <= 0:
            return 0.0
        return (self.filled_quantity / self.quantity) * 100
    
    def update_fill(self, fill_quantity: float, fill_price: float, commission: float = 0.0) -> None:
        """Update order with fill information"""
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        
        # Calculate weighted average price
        total_value = (self.average_fill_price * self.filled_quantity) + (fill_price * fill_quantity)
        self.average_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        self.commission += commission
        self.updated_at = datetime.utcnow()
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self.filled_at = datetime.utcnow()
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self, reason: str = "") -> None:
        """Cancel the order"""
        if self.is_terminal:
            return
        
        self.status = OrderStatus.CANCELLED
        self.cancelled_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        if reason:
            self.error_message = reason
    
    def reject(self, reason: str) -> None:
        """Reject the order"""
        self.status = OrderStatus.REJECTED
        self.rejection_reason = reason
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary"""
        return {
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "trailing_delta": self.trailing_delta,
            "time_in_force": self.time_in_force,
            "reduce_only": self.reduce_only,
            "position_side": self.position_side.value if self.position_side else None,
            "status": self.status.value,
            "filled_quantity": self.filled_quantity,
            "remaining_quantity": self.remaining_quantity,
            "average_fill_price": self.average_fill_price,
            "commission": self.commission,
            "commission_asset": self.commission_asset,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "submitted_at": self.submitted_at.isoformat() if self.submitted_at else None,
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "strategy_name": self.strategy_name,
            "session_id": self.session_id,
            "error_message": self.error_message,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class Fill:
    """Trade fill information"""
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    commission: float
    commission_asset: str
    is_maker: bool = False
    is_best_match: bool = False
    trade_id: Optional[str] = None
    trade_time: Optional[datetime] = None


@dataclass
class Position:
    """Position information"""
    symbol: str
    side: PositionSide
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    liquidation_price: Optional[float] = None
    margin: float = 0.0
    open_orders: int = 0
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def update_market(self, mark_price: float) -> None:
        """Update position with new market price"""
        self.mark_price = mark_price
        self.last_update = datetime.utcnow()
        
        if self.entry_price > 0:
            if self.side == PositionSide.LONG:
                self.unrealized_pnl = (mark_price - self.entry_price) * self.size
            else:
                self.unrealized_pnl = (self.entry_price - mark_price) * self.size
            
            self.unrealized_pnl_pct = (self.unrealized_pnl / (self.entry_price * self.size)) * 100 if self.entry_price * self.size > 0 else 0


@dataclass
class ExecutionResult:
    """Result of an execution operation"""
    success: bool
    order: Optional[Order] = None
    message: str = ""
    error_code: Optional[str] = None
    fills: List[Fill] = field(default_factory=list)
    execution_time_ms: float = 0.0


class OrderBook:
    """Simplified order book for slippage modeling"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bids: List[tuple[float, float]] = []  # (price, quantity)
        self.asks: List[tuple[float, float]] = []  # (price, quantity)
        self.last_update: datetime = datetime.utcnow()
    
    def update_bids(self, bids: List[tuple[float, float]]) -> None:
        """Update bid levels"""
        self.bids = sorted(bids, key=lambda x: -x[0])[:20]  # Top 20
        self.last_update = datetime.utcnow()
    
    def update_asks(self, asks: List[tuple[float, float]]) -> None:
        """Update ask levels"""
        self.asks = sorted(asks, key=lambda x: x[0])[:20]  # Top 20
        self.last_update = datetime.utcnow()
    
    def get_mid_price(self) -> float:
        """Get mid price"""
        if not self.bids or not self.asks:
            return 0.0
        return (self.bids[0][0] + self.asks[0][0]) / 2
    
    def get_slippage(self, side: OrderSide, quantity: float, base_price: float) -> float:
        """
        Estimate slippage for an order.
        
        Args:
            side: Order side
            quantity: Order quantity
            base_price: Base price to calculate slippage from
            
        Returns:
            Slippage as a percentage
        """
        if side == OrderSide.BUY:
            book_side = self.asks
        else:
            book_side = self.bids
        
        if not book_side:
            return 0.0
        
        # Simulate execution against book
        remaining_qty = quantity
        total_cost = 0.0
        best_price = book_side[0][0] if book_side else base_price
        
        for price, qty in book_side:
            if remaining_qty <= 0:
                break
            exec_qty = min(remaining_qty, qty)
            total_cost += exec_qty * price
            remaining_qty -= exec_qty
        
        if remaining_qty > 0:
            # Liquidity exhausted, estimate worse price
            total_cost += remaining_qty * (best_price * 1.01 if side == OrderSide.BUY else best_price * 0.99)
        
        avg_exec_price = total_cost / quantity if quantity > 0 else base_price
        
        slippage = abs(avg_exec_price - base_price) / base_price * 100 if base_price > 0 else 0
        
        return slippage


class ExecutionEngine:
    """
    Execution engine for order management and trade execution.
    
    Handles:
    - Order creation and submission
    - Order modification and cancellation
    - Fill processing and position updates
    - Risk integration
    - Slippage modeling
    - Fee calculation
    - Order lifecycle management
    """
    
    def __init__(
        self,
        broker_client=None,
        risk_manager=None,
        position_sizer=None,
        redis_client=None,
        simulate: bool = True
    ):
        """
        Initialize ExecutionEngine.
        
        Args:
            broker_client: Broker API client (e.g., BinanceRESTClient)
            risk_manager: RiskManager instance for risk checks
            position_sizer: PositionSizer instance for position sizing
            redis_client: Redis client for state persistence
            simulate: If True, simulate execution without broker
        """
        self.broker_client = broker_client
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.redis_client = redis_client
        self.simulate = simulate
        
        # Order management
        self.orders: Dict[str, Order] = {}
        self.orders_by_symbol: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_strategy: Dict[str, List[str]] = defaultdict(list)
        self.orders_by_session: Dict[str, List[str]] = defaultdict(list)
        
        # Position management
        self.positions: Dict[str, Position] = {}
        
        # Fill tracking
        self.fills: List[Fill] = []
        
        # Order book for slippage modeling
        self.order_books: Dict[str, OrderBook] = {}
        
        # Slippage and fee models
        self.default_maker_fee = 0.0002  # 0.02%
        self.default_taker_fee = 0.0004  # 0.04%
        self.base_slippage_pct = 0.05  # 0.05% base slippage
        
        # Callbacks
        self.order_callbacks: List[Callable[[Order], None]] = []
        self.fill_callbacks: List[Callable[[Fill], None]] = []
        self.position_callbacks: List[Callable[[str, Position], None]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Rate limiting
        self.order_rate_limit = 10  # orders per second
        self.last_order_times: List[float] = []
        
        logger.info(f"ExecutionEngine initialized (simulate={simulate})")
    
    def set_order_callback(self, callback: Callable[[Order], None]) -> None:
        """Register a callback for order updates"""
        self.order_callbacks.append(callback)
    
    def set_fill_callback(self, callback: Callable[[Fill], None]) -> None:
        """Register a callback for fills"""
        self.fill_callbacks.append(callback)
    
    def set_position_callback(self, callback: Callable[[str, Position], None]) -> None:
        """Register a callback for position updates"""
        self.position_callbacks.append(callback)
    
    def _check_rate_limit(self) -> bool:
        """Check if order rate limit is exceeded"""
        now = time.time()
        # Remove orders older than 1 second
        self.last_order_times = [t for t in self.last_order_times if now - t < 1.0]
        
        if len(self.last_order_times) >= self.order_rate_limit:
            return False
        
        self.last_order_times.append(now)
        return True
    
    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trailing_delta: Optional[float] = None,
        time_in_force: str = "GTC",
        reduce_only: bool = False,
        position_side: Optional[PositionSide] = None,
        strategy_name: Optional[str] = None,
        session_id: Optional[str] = None,
        signal_id: Optional[str] = None,
        validate_risk: bool = True
    ) -> Order:
        """
        Create a new order.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Limit price (for limit orders)
            stop_price: Stop price (for stop orders)
            trailing_delta: Trailing delta (for trailing stops)
            time_in_force: Time in force (GTC, IOC, FOK)
            reduce_only: Reduce only flag
            position_side: Position side (LONG/SHORT/BOTH)
            strategy_name: Strategy name for tracking
            session_id: Session ID for tracking
            signal_id: Signal ID for tracking
            validate_risk: Whether to validate with risk manager
            
        Returns:
            Created Order object
        """
        order = Order(
            symbol=symbol.upper(),
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            trailing_delta=trailing_delta,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
            position_side=position_side,
            strategy_name=strategy_name,
            session_id=session_id,
            signal_id=signal_id
        )
        
        # Risk validation
        if validate_risk and self.risk_manager:
            self._validate_order_risk(order)
        
        return order
    
    def _validate_order_risk(self, order: Order) -> None:
        """Validate order against risk manager"""
        if not self.risk_manager:
            order.risk_check_passed = True
            return
        
        # Check if trading is halted
        if self.risk_manager.is_halted:
            order.reject(f"Trading halted: {self.risk_manager.halt_message}")
            return
        
        # Check position size
        entry_price = order.price or self._get_market_price(order.symbol)
        
        result = self.risk_manager.check_position_size(
            symbol=order.symbol,
            quantity=order.quantity,
            price=entry_price,
            side=order.side.value
        )
        
        order.risk_check_passed = result.approved
        order.risk_adjustment_factor = result.adjustment_factor
        
        if not result.approved:
            order.reject(f"Risk check failed: {result.message}")
            return
        
        # Apply adjustment if needed
        if result.reduced_quantity is not None:
            order.quantity = result.reduced_quantity
            order.remaining_quantity = order.quantity
    
    def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit an order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            ExecutionResult with execution status
        """
        start_time = time.time()
        
        with self._lock:
            # Check rate limit
            if not self._check_rate_limit():
                return ExecutionResult(
                    success=False,
                    order=order,
                    message="Rate limit exceeded",
                    error_code="RATE_LIMIT"
                )
            
            # Check if order is valid
            if order.status == OrderStatus.REJECTED:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=order.rejection_reason or "Order was rejected",
                    error_code="REJECTED"
                )
            
            # Check if trading is halted
            if self.risk_manager and self.risk_manager.is_halted:
                order.reject(f"Trading halted: {self.risk_manager.halt_message}")
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=order.rejection_reason,
                    error_code="HALTED"
                )
            
            # Submit to broker or simulate
            if self.simulate:
                result = self._simulate_execution(order)
            else:
                result = self._submit_to_broker(order)
            
            # Track order
            if result.success:
                order.status = OrderStatus.SUBMITTED
                order.submitted_at = datetime.utcnow()
                self.orders[order.order_id] = order
                self.orders_by_symbol[order.symbol].append(order.order_id)
                if order.strategy_name:
                    self.orders_by_strategy[order.strategy_name].append(order.order_id)
                if order.session_id:
                    self.orders_by_session[order.session_id].append(order.order_id)
            
            result.order = order
            result.execution_time_ms = (time.time() - start_time) * 1000
            
            return result
    
    def _submit_to_broker(self, order: Order) -> ExecutionResult:
        """Submit order to broker API"""
        if not self.broker_client:
            return ExecutionResult(
                success=False,
                order=order,
                message="No broker client configured",
                error_code="NO_BROKER"
            )
        
        try:
            # Map order to broker format
            broker_order = self.broker_client.create_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.price,
                stop_price=order.stop_price,
                time_in_force=order.time_in_force,
                reduce_only=order.reduce_only,
                position_side=order.position_side.value if order.position_side else None,
                new_client_order_id=order.client_order_id
            )
            
            # Update order with broker response
            order.order_id = broker_order.get("orderId", order.order_id)
            order.status = OrderStatus.NEW
            
            return ExecutionResult(
                success=True,
                order=order,
                message=f"Order submitted: {order.order_id}"
            )
            
        except Exception as e:
            order.reject(str(e))
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Broker submission failed: {str(e)}",
                error_code="BROKER_ERROR"
            )
    
    def _simulate_execution(self, order: Order) -> ExecutionResult:
        """Simulate order execution"""
        # Get market price
        market_price = self._get_market_price(order.symbol)
        
        if market_price <= 0:
            return ExecutionResult(
                success=False,
                order=order,
                message="Cannot execute: invalid market price",
                error_code="INVALID_PRICE"
            )
        
        # Determine execution price
        if order.order_type == OrderType.MARKET:
            # Market order: execute at current market price
            execution_price = market_price
            
            # Add slippage
            slippage = self._calculate_slippage(order.symbol, order.side, order.quantity, market_price)
            if order.side == OrderSide.BUY:
                execution_price *= (1 + slippage / 100)
            else:
                execution_price *= (1 - slippage / 100)
        
        elif order.order_type == OrderType.LIMIT:
            # Limit order: execute if price condition met
            if order.side == OrderSide.BUY and market_price <= order.price:
                execution_price = order.price
            elif order.side == OrderSide.SELL and market_price >= order.price:
                execution_price = order.price
            else:
                # Price not met, order stays pending
                order.status = OrderStatus.PENDING
                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Limit order waiting for price"
                )
        
        elif order.order_type in [OrderType.STOP, OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
            # Stop order: not triggered yet
            if not self._is_stop_triggered(order, market_price):
                order.status = OrderStatus.PENDING
                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Stop order waiting for trigger"
                )
            
            if order.order_type == OrderType.STOP_MARKET:
                execution_price = market_price
            else:
                execution_price = order.price or market_price
        
        elif order.order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
            # Take profit: not triggered yet
            if not self._is_take_profit_triggered(order, market_price):
                order.status = OrderStatus.PENDING
                return ExecutionResult(
                    success=True,
                    order=order,
                    message="Take profit order waiting for trigger"
                )
            
            if order.order_type == OrderType.TAKE_PROFIT_MARKET:
                execution_price = market_price
            else:
                execution_price = order.price or market_price
        
        else:
            execution_price = order.price or market_price
        
        # Calculate commission
        notional = order.quantity * execution_price
        commission = notional * self.default_taker_fee
        
        # Create fill
        fill = Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=execution_price,
            quantity=order.quantity,
            commission=commission,
            commission_asset="USDT",
            is_maker=False,
            is_best_match=True,
            trade_id=str(uuid.uuid4())[:12],
            trade_time=datetime.utcnow()
        )
        
        # Update order with fill
        order.update_fill(order.quantity, execution_price, commission)
        order.status = OrderStatus.FILLED
        
        # Update position
        self._update_position_from_fill(fill)
        
        # Record trade with risk manager
        if self.risk_manager:
            pnl = self._calculate_trade_pnl(fill, order)
            self.risk_manager.record_trade({
                "symbol": order.symbol,
                "side": order.side.value,
                "entry_price": execution_price,
                "exit_price": execution_price,
                "quantity": order.quantity,
                "pnl": pnl,
                "commission": commission,
                "timestamp": time.time()
            })
        
        # Trigger callbacks
        self._trigger_fill_callbacks(fill)
        
        return ExecutionResult(
            success=True,
            order=order,
            message=f"Order filled at {execution_price:.6f}",
            fills=[fill]
        )
    
    def _is_stop_triggered(self, order: Order, market_price: float) -> bool:
        """Check if stop order is triggered"""
        if order.stop_price is None:
            return False
        
        if order.side == OrderSide.BUY:
            return market_price >= order.stop_price
        else:
            return market_price <= order.stop_price
    
    def _is_take_profit_triggered(self, order: Order, market_price: float) -> bool:
        """Check if take profit order is triggered"""
        if order.stop_price is None:
            return False
        
        if order.side == OrderSide.BUY:  # Long position take profit
            return market_price <= order.stop_price
        else:  # Short position take profit
            return market_price >= order.stop_price
    
    def _get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol"""
        # Check order book first
        if symbol in self.order_books:
            ob = self.order_books[symbol]
            if ob.bids and ob.asks:
                return (ob.bids[0][0] + ob.asks[0][0]) / 2
        
        # Check positions for mark price
        if symbol in self.positions:
            return self.positions[symbol].mark_price
        
        # Default fallback
        return 0.0
    
    def _calculate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        base_price: float
    ) -> float:
        """Calculate slippage for an order"""
        slippage = self.base_slippage_pct
        
        # Check order book for better estimate
        if symbol in self.order_books:
            slippage = self.order_books[symbol].get_slippage(side, quantity, base_price)
        
        # Add randomness to simulate market conditions
        slippage *= (0.5 + random.random())  # 50-150% of base slippage
        
        return slippage
    
    def _calculate_trade_pnl(self, fill: Fill, order: Order) -> float:
        """Calculate PnL for a trade"""
        # For a completed trade, PnL would be calculated at close
        # Here we just return 0 as it's an opening trade
        return 0.0
    
    def _update_position_from_fill(self, fill: Fill) -> None:
        """Update position based on fill"""
        with self._lock:
            symbol = fill.symbol
            position_side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT
            
            if symbol in self.positions:
                pos = self.positions[symbol]
                
                # Check if adding to same side or closing
                if pos.side == position_side:
                    # Add to position
                    total_size = pos.size + fill.quantity
                    total_cost = (pos.entry_price * pos.size) + (fill.price * fill.quantity)
                    pos.entry_price = total_cost / total_size if total_size > 0 else 0
                    pos.size = total_size
                else:
                    # Reducing or reversing
                    if fill.quantity >= pos.size:
                        # Close position and potentially reverse
                        remaining = fill.quantity - pos.size
                        pos.size = remaining
                        pos.side = position_side if remaining > 0 else PositionSide.LONG
                        pos.entry_price = fill.price if remaining > 0 else 0
                    else:
                        # Just reduce
                        pos.size -= fill.quantity
                
                pos.update_market(fill.price)
                
                if pos.size <= 0:
                    del self.positions[symbol]
            else:
                # New position
                self.positions[symbol] = Position(
                    symbol=symbol,
                    side=position_side,
                    size=fill.quantity,
                    entry_price=fill.price,
                    mark_price=fill.price,
                    unrealized_pnl=0.0,
                    unrealized_pnl_pct=0.0,
                    leverage=1,
                    liquidation_price=None,
                    margin=fill.quantity * fill.price
                )
            
            # Trigger position callbacks
            if symbol in self.positions:
                self._trigger_position_callbacks(symbol, self.positions[symbol])
    
    def _trigger_fill_callbacks(self, fill: Fill) -> None:
        """Trigger registered fill callbacks"""
        for callback in self.fill_callbacks:
            try:
                callback(fill)
            except Exception as e:
                logger.error(f"Fill callback error: {e}")
    
    def _trigger_position_callbacks(self, symbol: str, position: Position) -> None:
        """Trigger registered position callbacks"""
        for callback in self.position_callbacks:
            try:
                callback(symbol, position)
            except Exception as e:
                logger.error(f"Position callback error: {e}")
    
    def cancel_order(self, order_id: str, reason: str = "User requested") -> ExecutionResult:
        """
        Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            reason: Cancellation reason
            
        Returns:
            ExecutionResult with cancellation status
        """
        with self._lock:
            if order_id not in self.orders:
                return ExecutionResult(
                    success=False,
                    message=f"Order not found: {order_id}",
                    error_code="NOT_FOUND"
                )
            
            order = self.orders[order_id]
            
            if order.is_terminal:
                return ExecutionResult(
                    success=False,
                    order=order,
                    message=f"Cannot cancel order in status: {order.status.value}",
                    error_code="INVALID_STATUS"
                )
            
            if self.simulate:
                order.cancel(reason)
                return ExecutionResult(
                    success=True,
                    order=order,
                    message=f"Order cancelled: {order_id}"
                )
            else:
                try:
                    self.broker_client.cancel_order(
                        symbol=order.symbol,
                        order_id=int(order.order_id)
                    )
                    order.cancel(reason)
                    return ExecutionResult(
                        success=True,
                        order=order,
                        message=f"Order cancelled: {order_id}"
                    )
                except Exception as e:
                    return ExecutionResult(
                        success=False,
                        order=order,
                        message=f"Cancel failed: {str(e)}",
                        error_code="BROKER_ERROR"
                    )
    
    def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        strategy_name: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> List[ExecutionResult]:
        """
        Cancel multiple orders.
        
        Args:
            symbol: Cancel orders for this symbol
            strategy_name: Cancel orders for this strategy
            session_id: Cancel orders for this session
            
        Returns:
            List of cancellation results
        """
        results = []
        
        # Find orders to cancel
        order_ids = []
        
        if symbol:
            order_ids.extend(self.orders_by_symbol.get(symbol, []))
        if strategy_name:
            order_ids.extend(self.orders_by_strategy.get(strategy_name, []))
        if session_id:
            order_ids.extend(self.orders_by_session.get(session_id, []))
        
        # Remove duplicates
        order_ids = list(set(order_ids))
        
        for order_id in order_ids:
            results.append(self.cancel_order(order_id))
        
        return results
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders, optionally filtered by symbol"""
        open_orders = [o for o in self.orders.values() if o.is_active]
        
        if symbol:
            open_orders = [o for o in open_orders if o.symbol == symbol]
        
        return open_orders
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all positions"""
        return self.positions.copy()
    
    def update_market_price(self, symbol: str, price: float) -> None:
        """
        Update market price and check stop orders.
        
        Args:
            symbol: Trading symbol
            price: Current market price
        """
        with self._lock:
            # Update position mark price
            if symbol in self.positions:
                self.positions[symbol].update_market(price)
            
            # Check for triggered stop/take-profit orders
            for order in list(self.orders.values()):
                if order.symbol != symbol or not order.is_active:
                    continue
                
                triggered = False
                
                if order.order_type in [OrderType.STOP, OrderType.STOP_MARKET, OrderType.STOP_LIMIT]:
                    triggered = self._is_stop_triggered(order, price)
                elif order.order_type in [OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_MARKET]:
                    triggered = self._is_take_profit_triggered(order, price)
                
                if triggered:
                    # Execute the order
                    self._simulate_execution(order)
    
    def update_order_book(self, symbol: str, bids: List[tuple], asks: List[tuple]) -> None:
        """
        Update order book for slippage modeling.
        
        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) tuples
            asks: List of (price, quantity) tuples
        """
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol)
        
        self.order_books[symbol].update_bids(bids)
        self.order_books[symbol].update_asks(asks)
    
    def sync_positions(self) -> None:
        """Sync positions with broker"""
        if self.simulate or not self.broker_client:
            return
        
        try:
            positions = self.broker_client.get_position_info()
            
            for pos_data in positions:
                symbol = pos_data.get("symbol")
                size = float(pos_data.get("positionAmt", 0))
                
                if abs(size) > 0:
                    entry_price = float(pos_data.get("entryPrice", 0))
                    mark_price = float(pos_data.get("markPrice", entry_price))
                    unrealized_pnl = float(pos_data.get("unRealizedProfit", 0))
                    leverage = int(pos_data.get("leverage", 1))
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=PositionSide.LONG if size > 0 else PositionSide.SHORT,
                        size=abs(size),
                        entry_price=entry_price,
                        mark_price=mark_price,
                        unrealized_pnl=unrealized_pnl,
                        unrealized_pnl_pct=(unrealized_pnl / (entry_price * size)) * 100 if entry_price * size > 0 else 0,
                        leverage=leverage,
                        liquidation_price=float(pos_data.get("liquidationPrice", 0)) or None,
                        margin=float(pos_data.get("isolatedMargin", 0))
                    )
                elif symbol in self.positions:
                    del self.positions[symbol]
            
            logger.info(f"Synced {len(self.positions)} positions from broker")
            
        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary"""
        with self._lock:
            total_orders = len(self.orders)
            filled_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.FILLED)
            cancelled_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.CANCELLED)
            rejected_orders = sum(1 for o in self.orders.values() if o.status == OrderStatus.REJECTED)
            
            total_commission = sum(o.commission for o in self.orders.values())
            total_filled_qty = sum(o.filled_quantity for o in self.orders.values())
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "simulate": self.simulate,
                "orders": {
                    "total": total_orders,
                    "filled": filled_orders,
                    "cancelled": cancelled_orders,
                    "rejected": rejected_orders,
                    "pending": total_orders - filled_orders - cancelled_orders - rejected_orders,
                },
                "filled_quantity": total_filled_qty,
                "total_commission": total_commission,
                "positions": {
                    "count": len(self.positions),
                    "details": [
                        {
                            "symbol": p.symbol,
                            "side": p.side.value,
                            "size": p.size,
                            "entry_price": p.entry_price,
                            "mark_price": p.mark_price,
                            "unrealized_pnl": p.unrealized_pnl,
                        }
                        for p in self.positions.values()
                    ]
                },
                "order_books": {
                    symbol: {
                        "mid_price": ob.get_mid_price(),
                        "best_bid": ob.bids[0][0] if ob.bids else None,
                        "best_ask": ob.asks[0][0] if ob.asks else None,
                    }
                    for symbol, ob in self.order_books.items()
                }
            }
    
    def save_state(self) -> bool:
        """Save execution engine state to Redis"""
        if not self.redis_client:
            return False
        
        try:
            state = {
                "positions": {
                    symbol: {
                        "symbol": pos.symbol,
                        "side": pos.side.value,
                        "size": pos.size,
                        "entry_price": pos.entry_price,
                        "mark_price": pos.mark_price,
                        "unrealized_pnl": pos.unrealized_pnl,
                        "leverage": pos.leverage,
                    }
                    for symbol, pos in self.positions.items()
                },
                "orders_count": len(self.orders),
            }
            self.redis_client.set("execution_engine:state", state, json_encode=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save execution engine state: {e}")
            return False
    
    def load_state(self) -> bool:
        """Load execution engine state from Redis"""
        if not self.redis_client:
            return False
        
        try:
            state = self.redis_client.get("execution_engine:state", json_decode=True)
            if not state:
                return False
            
            if "positions" in state:
                for symbol, pos_data in state["positions"].items():
                    self.positions[symbol] = Position(
                        symbol=pos_data["symbol"],
                        side=PositionSide(pos_data["side"]),
                        size=pos_data["size"],
                        entry_price=pos_data["entry_price"],
                        mark_price=pos_data["mark_price"],
                        unrealized_pnl=pos_data["unrealized_pnl"],
                        unrealized_pnl_pct=0.0,
                        leverage=pos_data.get("leverage", 1)
                    )
            
            logger.info(f"Loaded {len(self.positions)} positions from Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to load execution engine state: {e}")
            return False


def create_execution_engine(
    broker_client=None,
    risk_manager=None,
    position_sizer=None,
    redis_client=None,
    simulate: bool = True
) -> ExecutionEngine:
    """
    Factory function to create an ExecutionEngine.
    
    Args:
        broker_client: Broker API client
        risk_manager: RiskManager instance
        position_sizer: PositionSizer instance
        redis_client: Redis client for state persistence
        simulate: If True, simulate execution
        
    Returns:
        Configured ExecutionEngine instance
    """
    return ExecutionEngine(
        broker_client=broker_client,
        risk_manager=risk_manager,
        position_sizer=position_sizer,
        redis_client=redis_client,
        simulate=simulate
    )
