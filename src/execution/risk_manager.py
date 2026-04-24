"""
Mimia Quant Trading System - Risk Manager

Comprehensive risk management with position limits, drawdown controls,
and exposure management for the Mimia Quant trading system.

Risk Parameters:
- max_position_pct: 1.5% maximum position size as percentage of portfolio
- max_daily_drawdown: 3% maximum daily drawdown before trading halt
- max_monthly_drawdown: 8% maximum monthly drawdown before trading halt
"""

import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import threading

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TradingHaltReason(Enum):
    """Reasons for trading halts"""
    DAILY_DRAWDOWN = "daily_drawdown"
    MONTHLY_DRAWDOWN = "monthly_drawdown"
    MAX_POSITION_EXCEEDED = "max_position_exceeded"
    MAX_EXPOSURE_EXCEEDED = "max_exposure_exceeded"
    MANUAL_HALT = "manual_halt"


@dataclass
class RiskLimits:
    """Risk limit configuration"""
    max_position_pct: float = 1.5  # Maximum position size as % of portfolio
    max_daily_drawdown_pct: float = 3.0  # Maximum daily drawdown %
    max_monthly_drawdown_pct: float = 8.0  # Maximum monthly drawdown %
    max_total_exposure_pct: float = 10.0  # Maximum total exposure %
    max_leverage: int = 10  # Maximum allowed leverage
    max_correlated_exposure_pct: float = 5.0  # Max exposure to correlated assets
    daily_loss_limit_pct: float = 2.0  # Daily loss limit before reducing positions
    min_account_balance: float = 100.0  # Minimum account balance to trade
    
    def __post_init__(self):
        """Validate risk limits"""
        if not 0 < self.max_position_pct <= 100:
            raise ValueError(f"max_position_pct must be between 0 and 100, got {self.max_position_pct}")
        if not 0 < self.max_daily_drawdown_pct <= 100:
            raise ValueError(f"max_daily_drawdown_pct must be between 0 and 100, got {self.max_daily_drawdown_pct}")
        if not 0 < self.max_monthly_drawdown_pct <= 100:
            raise ValueError(f"max_monthly_drawdown_pct must be between 0 and 100, got {self.max_monthly_drawdown_pct}")


@dataclass
class PositionInfo:
    """Position information"""
    symbol: str
    side: str  # LONG or SHORT
    size: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    leverage: int
    liquidation_price: Optional[float] = None
    margin: float = 0.0


@dataclass
class DrawdownState:
    """Drawdown tracking state"""
    high_water_mark: float = 0.0
    current_drawdown: float = 0.0
    drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)
    
    def update(self, current_equity: float) -> None:
        """Update drawdown state with new equity value"""
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            self.peak_equity = current_equity
            self.current_drawdown = 0.0
            self.drawdown_pct = 0.0
        else:
            self.current_drawdown = self.high_water_mark - current_equity
            if self.high_water_mark > 0:
                self.drawdown_pct = (self.current_drawdown / self.high_water_mark) * 100
        self.last_update = datetime.utcnow()


@dataclass
class DailyState:
    """Daily tracking state"""
    date: str = ""  # YYYY-MM-DD format
    starting_equity: float = 0.0
    closing_equity: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = float('inf')
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def update_trade(self, pnl: float) -> None:
        """Update state after a trade"""
        self.trades_count += 1
        self.realized_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
        elif pnl < 0:
            self.losing_trades += 1
    
    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if current_equity < self.trough_equity:
            self.trough_equity = current_equity
            if self.peak_equity > 0:
                self.max_drawdown = self.peak_equity - self.trough_equity
                self.max_drawdown_pct = (self.max_drawdown / self.peak_equity) * 100


@dataclass 
class MonthlyState:
    """Monthly tracking state"""
    year_month: str = ""  # YYYY-MM format
    starting_equity: float = 0.0
    peak_equity: float = 0.0
    trough_equity: float = float('inf')
    total_realized_pnl: float = 0.0
    total_unrealized_pnl: float = 0.0
    trades_count: int = 0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    
    def update_equity(self, current_equity: float) -> None:
        """Update equity tracking"""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        if current_equity < self.trough_equity:
            self.trough_equity = current_equity
            if self.peak_equity > 0:
                self.max_drawdown = self.peak_equity - self.trough_equity
                self.max_drawdown_pct = (self.max_drawdown / self.peak_equity) * 100


@dataclass
class RiskCheckResult:
    """Result of risk check"""
    approved: bool
    risk_level: RiskLevel = RiskLevel.LOW
    message: str = ""
    adjustment_factor: float = 1.0
    halt_trading: bool = False
    halt_reason: Optional[TradingHaltReason] = None
    reduced_quantity: Optional[float] = None


class RiskManager:
    """
    Comprehensive risk manager for the Mimia Quant trading system.
    
    Features:
    - Position size limits
    - Daily and monthly drawdown tracking
    - Exposure management
    - Leverage controls
    - Trading halt mechanisms
    - Thread-safe operations
    """
    
    def __init__(
        self,
        limits: Optional[RiskLimits] = None,
        initial_equity: float = 10000.0,
        redis_client=None
    ):
        """
        Initialize the risk manager.
        
        Args:
            limits: Risk limits configuration
            initial_equity: Starting equity
            redis_client: Optional Redis client for state persistence
        """
        self.limits = limits or RiskLimits()
        self._lock = threading.RLock()
        
        # State
        self.initial_equity = initial_equity
        self.current_equity = initial_equity
        
        # Drawdown tracking
        self.portfolio_drawdown = DrawdownState(high_water_mark=initial_equity, peak_equity=initial_equity)
        self.daily_state = DailyState()
        self.monthly_state = MonthlyState()
        
        # Position tracking
        self.positions: Dict[str, PositionInfo] = {}
        self.positions_by_symbol: Dict[str, PositionInfo] = {}
        
        # Halt state
        self.is_halted = False
        self.halt_reason: Optional[TradingHaltReason] = None
        self.halt_timestamp: Optional[datetime] = None
        self.halt_message: str = ""
        
        # Redis client for persistence
        self.redis_client = redis_client
        
        # Trade history for Kelly calculation
        self.trade_history: List[Dict[str, Any]] = []
        self.max_trade_history = 1000
        
        # Correlation groups (symbols that are highly correlated)
        self.correlation_groups: Dict[str, List[str]] = defaultdict(list)
        
        logger.info(f"RiskManager initialized with limits: max_position_pct={self.limits.max_position_pct}%, "
                   f"max_daily_drawdown={self.limits.max_daily_drawdown_pct}%, "
                   f"max_monthly_drawdown={self.limits.max_monthly_drawdown_pct}%")
    
    @property
    def total_exposure(self) -> float:
        """Calculate total exposure in quote currency"""
        return sum(pos.size * pos.mark_price for pos in self.positions.values())
    
    @property
    def total_exposure_pct(self) -> float:
        """Calculate total exposure as percentage of equity"""
        if self.current_equity <= 0:
            return 100.0
        return (self.total_exposure / self.current_equity) * 100
    
    @property
    def daily_pnl(self) -> float:
        """Calculate daily PnL"""
        if self.daily_state.starting_equity > 0:
            return self.current_equity - self.daily_state.starting_equity
        return 0.0
    
    @property
    def daily_pnl_pct(self) -> float:
        """Calculate daily PnL as percentage"""
        if self.daily_state.starting_equity > 0:
            return (self.daily_pnl / self.daily_state.starting_equity) * 100
        return 0.0
    
    @property
    def monthly_pnl(self) -> float:
        """Calculate monthly PnL"""
        if self.monthly_state.starting_equity > 0:
            return self.current_equity - self.monthly_state.starting_equity
        return 0.0
    
    @property
    def monthly_pnl_pct(self) -> float:
        """Calculate monthly PnL as percentage"""
        if self.monthly_state.starting_equity > 0:
            return (self.monthly_pnl / self.monthly_state.starting_equity) * 100
        return 0.0
    
    @property
    def current_risk_level(self) -> RiskLevel:
        """Determine current risk level based on drawdown and exposure"""
        drawdown_pct = self.portfolio_drawdown.drawdown_pct
        exposure_pct = self.total_exposure_pct
        
        if drawdown_pct >= self.limits.max_daily_drawdown_pct or exposure_pct >= self.limits.max_total_exposure_pct:
            return RiskLevel.CRITICAL
        elif drawdown_pct >= self.limits.max_daily_drawdown_pct * 0.7:
            return RiskLevel.HIGH
        elif drawdown_pct >= self.limits.max_daily_drawdown_pct * 0.4 or exposure_pct >= self.limits.max_total_exposure_pct * 0.7:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def set_redis_client(self, redis_client) -> None:
        """Set Redis client for state persistence"""
        self.redis_client = redis_client
    
    def set_correlation_groups(self, groups: Dict[str, List[str]]) -> None:
        """Set correlation groups for exposure calculation"""
        self.correlation_groups = defaultdict(list, groups)
    
    def update_equity(self, new_equity: float) -> None:
        """Update current equity and track drawdowns"""
        with self._lock:
            old_equity = self.current_equity
            self.current_equity = new_equity
            
            # Update date tracking
            today = datetime.utcnow().strftime("%Y-%m-%d")
            year_month = datetime.utcnow().strftime("%Y-%m")
            
            # Initialize daily state if new day
            if self.daily_state.date != today:
                if self.daily_state.starting_equity > 0:
                    logger.info(f"New day: Previous day's PnL: {self.daily_state.realized_pnl:.2f}, "
                               f"Max drawdown: {self.daily_state.max_drawdown_pct:.2f}%")
                self.daily_state = DailyState(date=today, starting_equity=new_equity)
            
            # Initialize monthly state if new month
            if self.monthly_state.year_month != year_month:
                if self.monthly_state.starting_equity > 0:
                    logger.info(f"New month: Previous month's PnL: {self.monthly_state.total_realized_pnl:.2f}")
                self.monthly_state = MonthlyState(year_month=year_month, starting_equity=new_equity)
            
            # Update equity tracking
            self.daily_state.update_equity(new_equity)
            self.monthly_state.update_equity(new_equity)
            self.portfolio_drawdown.update(new_equity)
            
            # Check for halt conditions
            self._check_halt_conditions()
            
            logger.debug(f"Equity updated: {old_equity:.2f} -> {new_equity:.2f}, "
                        f"Daily PnL: {self.daily_pnl:.2f} ({self.daily_pnl_pct:.2f}%), "
                        f"Drawdown: {self.portfolio_drawdown.drawdown_pct:.2f}%")
    
    def update_position(self, position: PositionInfo) -> None:
        """Update position information"""
        with self._lock:
            self.positions[position.symbol] = position
            self.positions_by_symbol[position.symbol] = position
            self.daily_state.unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
    
    def remove_position(self, symbol: str) -> None:
        """Remove a position"""
        with self._lock:
            if symbol in self.positions:
                del self.positions[symbol]
            if symbol in self.positions_by_symbol:
                del self.positions_by_symbol[symbol]
            self.daily_state.unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())
    
    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Record a completed trade for analysis and Kelly calculation"""
        with self._lock:
            self.trade_history.append(trade)
            if len(self.trade_history) > self.max_trade_history:
                self.trade_history.pop(0)
            
            # Update daily state
            if 'pnl' in trade:
                pnl = trade['pnl']
                self.daily_state.update_trade(pnl)
                self.monthly_state.trades_count += 1
                
                if trade.get('side') == 'CLOSING':
                    self.current_equity += pnl
                    self.update_equity(self.current_equity)
    
    def check_position_size(
        self,
        symbol: str,
        quantity: float,
        price: float,
        side: str
    ) -> RiskCheckResult:
        """
        Check if a new position or position increase is allowed.
        
        Args:
            symbol: Trading symbol
            quantity: Order quantity
            price: Order price
            side: Order side (LONG or SHORT)
            
        Returns:
            RiskCheckResult with approval and any adjustments
        """
        with self._lock:
            position_value = quantity * price
            
            # Check minimum account balance
            if self.current_equity < self.limits.min_account_balance:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Account balance {self.current_equity:.2f} below minimum {self.limits.min_account_balance}",
                    halt_trading=True,
                    halt_reason=TradingHaltReason.MINIMUM_BALANCE if hasattr(TradingHaltReason, 'MINIMUM_BALANCE') else TradingHaltReason.MANUAL_HALT
                )
            
            # Calculate position as percentage of equity
            position_pct = (position_value / self.current_equity) * 100 if self.current_equity > 0 else 100
            
            # Check if adding to existing position
            existing_position = self.positions_by_symbol.get(symbol)
            if existing_position:
                total_value = existing_position.size * existing_position.entry_price + position_value
                total_pct = (total_value / self.current_equity) * 100 if self.current_equity > 0 else 100
            else:
                total_pct = position_pct
            
            # Check max position size
            if total_pct > self.limits.max_position_pct:
                # Calculate reduced quantity
                max_value = (self.current_equity * self.limits.max_position_pct) / 100
                if existing_position:
                    max_value = max_value - (existing_position.size * existing_position.entry_price)
                max_quantity = max_value / price if price > 0 else 0
                
                if max_quantity <= 0:
                    return RiskCheckResult(
                        approved=False,
                        risk_level=RiskLevel.HIGH,
                        message=f"Position for {symbol} would exceed maximum {self.limits.max_position_pct}% of portfolio",
                        adjustment_factor=0.0
                    )
                
                adjustment_factor = max_quantity / quantity if quantity > 0 else 0
                return RiskCheckResult(
                    approved=True,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Position size reduced from {quantity} to {max_quantity:.6f} ({self.limits.max_position_pct}% limit)",
                    adjustment_factor=adjustment_factor,
                    reduced_quantity=max_quantity
                )
            
            # Check total exposure
            new_total_exposure = self.total_exposure + position_value
            new_exposure_pct = (new_total_exposure / self.current_equity) * 100 if self.current_equity > 0 else 100
            
            if new_exposure_pct > self.limits.max_total_exposure_pct:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.HIGH,
                    message=f"Total exposure would exceed maximum {self.limits.max_total_exposure_pct}%",
                    halt_trading=True,
                    halt_reason=TradingHaltReason.MAX_EXPOSURE_EXCEEDED
                )
            
            # Check correlation group exposure
            correlated_exposure = self._calculate_correlated_exposure(symbol, position_value)
            correlated_exposure_pct = (correlated_exposure / self.current_equity) * 100 if self.current_equity > 0 else 100
            
            if correlated_exposure_pct > self.limits.max_correlated_exposure_pct:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Correlated exposure {correlated_exposure_pct:.2f}% would exceed limit {self.limits.max_correlated_exposure_pct}%"
                )
            
            return RiskCheckResult(
                approved=True,
                risk_level=self.current_risk_level,
                message="Position approved"
            )
    
    def _calculate_correlated_exposure(self, symbol: str, additional_value: float) -> float:
        """Calculate total exposure including correlated assets"""
        total = additional_value
        
        # Check if symbol belongs to any correlation group
        for group_name, group_symbols in self.correlation_groups.items():
            if symbol in group_symbols:
                for sym in group_symbols:
                    if sym in self.positions_by_symbol:
                        pos = self.positions_by_symbol[sym]
                        total += pos.size * pos.mark_price
        
        return total
    
    def check_drawdown(self) -> RiskCheckResult:
        """
        Check if trading should continue based on drawdown limits.
        
        Returns:
            RiskCheckResult indicating if trading should continue
        """
        with self._lock:
            # Check daily drawdown
            if self.daily_state.max_drawdown_pct >= self.limits.max_daily_drawdown_pct:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Daily drawdown {self.daily_state.max_drawdown_pct:.2f}% exceeds limit {self.limits.max_daily_drawdown_pct}%",
                    halt_trading=True,
                    halt_reason=TradingHaltReason.DAILY_DRAWDOWN
                )
            
            # Check monthly drawdown
            if self.monthly_state.max_drawdown_pct >= self.limits.max_monthly_drawdown_pct:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.CRITICAL,
                    message=f"Monthly drawdown {self.monthly_state.max_drawdown_pct:.2f}% exceeds limit {self.limits.max_monthly_drawdown_pct}%",
                    halt_trading=True,
                    halt_reason=TradingHaltReason.MONTHLY_DRAWDOWN
                )
            
            # Check daily loss limit for position reduction
            if self.daily_pnl_pct <= -self.limits.daily_loss_limit_pct:
                return RiskCheckResult(
                    approved=True,
                    risk_level=RiskLevel.HIGH,
                    message=f"Daily loss {self.daily_pnl_pct:.2f}% exceeds limit {self.limits.daily_loss_limit_pct}%, reducing position sizes",
                    adjustment_factor=0.5  # Reduce positions by 50%
                )
            
            return RiskCheckResult(
                approved=True,
                risk_level=self.current_risk_level,
                message="Drawdown check passed"
            )
    
    def check_leverage(self, symbol: str, leverage: int, price: float, quantity: float) -> RiskCheckResult:
        """
        Check if requested leverage is allowed.
        
        Args:
            symbol: Trading symbol
            leverage: Requested leverage
            price: Entry price
            quantity: Order quantity
            
        Returns:
            RiskCheckResult with leverage approval
        """
        with self._lock:
            if leverage > self.limits.max_leverage:
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.HIGH,
                    message=f"Leverage {leverage} exceeds maximum {self.limits.max_leverage}",
                    adjustment_factor=0.0
                )
            
            # Check if position with leverage would be too large
            position_value = price * quantity
            required_margin = position_value / leverage if leverage > 0 else position_value
            
            if required_margin > self.current_equity * 0.5:  # Margin shouldn't exceed 50% of equity
                return RiskCheckResult(
                    approved=False,
                    risk_level=RiskLevel.MEDIUM,
                    message=f"Margin requirement {required_margin:.2f} too high relative to equity",
                    adjustment_factor=0.5
                )
            
            return RiskCheckResult(
                approved=True,
                risk_level=self.current_risk_level,
                message="Leverage approved"
            )
    
    def _check_halt_conditions(self) -> None:
        """Check and update halt conditions"""
        # Check if already halted
        if self.is_halted:
            return
        
        # Check daily drawdown
        if self.daily_state.max_drawdown_pct >= self.limits.max_daily_drawdown_pct:
            self._halt_trading(TradingHaltReason.DAILY_DRAWDOWN, 
                              f"Daily drawdown {self.daily_state.max_drawdown_pct:.2f}% exceeds limit")
            return
        
        # Check monthly drawdown
        if self.monthly_state.max_drawdown_pct >= self.limits.max_monthly_drawdown_pct:
            self._halt_trading(TradingHaltReason.MONTHLY_DRAWDOWN,
                              f"Monthly drawdown {self.monthly_state.max_drawdown_pct:.2f}% exceeds limit")
            return
        
        # Check total exposure
        if self.total_exposure_pct >= self.limits.max_total_exposure_pct:
            self._halt_trading(TradingHaltReason.MAX_EXPOSURE_EXCEEDED,
                              f"Total exposure {self.total_exposure_pct:.2f}% exceeds limit")
            return
    
    def _halt_trading(self, reason: TradingHaltReason, message: str) -> None:
        """Halt all trading"""
        self.is_halted = True
        self.halt_reason = reason
        self.halt_timestamp = datetime.utcnow()
        self.halt_message = message
        logger.critical(f"TRADING HALTED: {message} at {self.halt_timestamp}")
    
    def resume_trading(self) -> bool:
        """
        Attempt to resume trading after a halt.
        Returns True if trading can resume.
        """
        with self._lock:
            if not self.is_halted:
                return True
            
            # Check if conditions have improved
            can_resume = True
            
            if self.daily_state.max_drawdown_pct >= self.limits.max_daily_drawdown_pct:
                can_resume = False
            if self.monthly_state.max_drawdown_pct >= self.limits.max_monthly_drawdown_pct:
                can_resume = False
            if self.total_exposure_pct >= self.limits.max_total_exposure_pct:
                can_resume = False
            
            if can_resume:
                logger.info(f"Trading resumed. Previous halt reason: {self.halt_reason}")
                self.is_halted = False
                self.halt_reason = None
                self.halt_timestamp = None
                self.halt_message = ""
                return True
            
            return False
    
    def manual_halt(self, reason: str) -> None:
        """Manually halt trading"""
        self._halt_trading(TradingHaltReason.MANUAL_HALT, reason)
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        with self._lock:
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "is_halted": self.is_halted,
                "halt_reason": self.halt_reason.value if self.halt_reason else None,
                "halt_message": self.halt_message,
                "risk_level": self.current_risk_level.value,
                "equity": {
                    "current": self.current_equity,
                    "initial": self.initial_equity,
                    "daily_pnl": self.daily_pnl,
                    "daily_pnl_pct": self.daily_pnl_pct,
                    "monthly_pnl": self.monthly_pnl,
                    "monthly_pnl_pct": self.monthly_pnl_pct,
                },
                "drawdown": {
                    "current_pct": self.portfolio_drawdown.drawdown_pct,
                    "daily_max_pct": self.daily_state.max_drawdown_pct,
                    "monthly_max_pct": self.monthly_state.max_drawdown_pct,
                    "high_water_mark": self.portfolio_drawdown.high_water_mark,
                },
                "exposure": {
                    "total": self.total_exposure,
                    "total_pct": self.total_exposure_pct,
                    "max_allowed_pct": self.limits.max_total_exposure_pct,
                },
                "positions": {
                    "count": len(self.positions),
                    "details": [
                        {
                            "symbol": p.symbol,
                            "side": p.side,
                            "size": p.size,
                            "value": p.size * p.mark_price,
                            "unrealized_pnl": p.unrealized_pnl,
                            "leverage": p.leverage,
                        }
                        for p in self.positions.values()
                    ]
                },
                "trade_stats": {
                    "total_trades": self.daily_state.trades_count,
                    "winning_trades": self.daily_state.winning_trades,
                    "losing_trades": self.daily_state.losing_trades,
                    "win_rate": (self.daily_state.winning_trades / self.daily_state.trades_count * 100 
                                if self.daily_state.trades_count > 0 else 0),
                },
                "limits": {
                    "max_position_pct": self.limits.max_position_pct,
                    "max_daily_drawdown_pct": self.limits.max_daily_drawdown_pct,
                    "max_monthly_drawdown_pct": self.limits.max_monthly_drawdown_pct,
                    "max_total_exposure_pct": self.limits.max_total_exposure_pct,
                    "max_leverage": self.limits.max_leverage,
                }
            }
    
    def get_kelly_stats(self) -> Dict[str, float]:
        """
        Calculate statistics needed for Kelly Criterion from trade history.
        
        Returns:
            Dict with win_rate, avg_win, avg_loss, and calculated Kelly fraction
        """
        with self._lock:
            if len(self.trade_history) < 10:
                return {
                    "win_rate": 0.5,
                    "avg_win": 0.0,
                    "avg_loss": 0.0,
                    "kelly_fraction": 0.25,  # Conservative default
                    "sample_size": len(self.trade_history)
                }
            
            wins = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) > 0]
            losses = [t['pnl'] for t in self.trade_history if t.get('pnl', 0) < 0]
            
            win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0.5
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 1.0
            
            # Edge cases
            if avg_loss == 0:
                avg_loss = 1.0
            
            # Kelly formula: f* = (bp - q) / b
            # Where b = odds received, p = win probability, q = lose probability
            b = avg_win / avg_loss  # Win/loss ratio
            p = win_rate
            q = 1 - p
            
            # Kelly fraction with half-Kelly safety
            if b * p - q > 0:
                kelly = (b * p - q) / b
                kelly = max(0, min(1, kelly))  # Clamp to [0, 1]
            else:
                kelly = 0.0
            
            # Apply safety factor (use half Kelly)
            kelly = kelly * 0.5
            
            return {
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "kelly_fraction": kelly,
                "b": b,
                "p": p,
                "q": q,
                "sample_size": len(self.trade_history)
            }
    
    def calculate_max_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss_pct: float = 2.0,
        side: str = "LONG"
    ) -> float:
        """
        Calculate maximum position size based on risk rules.
        
        Args:
            symbol: Trading symbol
            price: Current price
            stop_loss_pct: Stop loss as percentage of entry price
            side: Position side (LONG or SHORT)
            
        Returns:
            Maximum position quantity
        """
        with self._lock:
            # Get Kelly stats
            kelly_stats = self.get_kelly_stats()
            kelly_fraction = kelly_stats["kelly_fraction"]
            
            # Base position size: risk per trade = 1-2% of equity * Kelly fraction
            risk_per_trade_pct = 0.02 * kelly_fraction if kelly_fraction > 0 else 0.01
            risk_amount = self.current_equity * risk_per_trade_pct
            
            # Risk is based on stop loss distance
            if stop_loss_pct > 0:
                # Position size = risk_amount / (price * stop_loss_pct)
                max_quantity = risk_amount / (price * (stop_loss_pct / 100))
            else:
                max_quantity = risk_amount / price
            
            # Apply position limit
            max_position_value = (self.current_equity * self.limits.max_position_pct) / 100
            max_quantity_by_limit = max_position_value / price if price > 0 else 0
            
            # Return the smaller of the two
            return min(max_quantity, max_quantity_by_limit)
    
    def save_state(self) -> bool:
        """Save risk manager state to Redis"""
        if not self.redis_client:
            return False
        
        try:
            state = {
                "current_equity": self.current_equity,
                "is_halted": self.is_halted,
                "halt_reason": self.halt_reason.value if self.halt_reason else None,
                "daily_state": {
                    "date": self.daily_state.date,
                    "starting_equity": self.daily_state.starting_equity,
                    "realized_pnl": self.daily_state.realized_pnl,
                    "trades_count": self.daily_state.trades_count,
                    "max_drawdown_pct": self.daily_state.max_drawdown_pct,
                },
                "monthly_state": {
                    "year_month": self.monthly_state.year_month,
                    "starting_equity": self.monthly_state.starting_equity,
                    "total_realized_pnl": self.monthly_state.total_realized_pnl,
                    "trades_count": self.monthly_state.trades_count,
                    "max_drawdown_pct": self.monthly_state.max_drawdown_pct,
                },
                "portfolio_drawdown": {
                    "high_water_mark": self.portfolio_drawdown.high_water_mark,
                    "drawdown_pct": self.portfolio_drawdown.drawdown_pct,
                },
                "trade_history_count": len(self.trade_history),
            }
            self.redis_client.set("risk_manager:state", state, json_encode=True)
            return True
        except Exception as e:
            logger.error(f"Failed to save risk manager state: {e}")
            return False
    
    def load_state(self) -> bool:
        """Load risk manager state from Redis"""
        if not self.redis_client:
            return False
        
        try:
            state = self.redis_client.get("risk_manager:state", json_decode=True)
            if not state:
                return False
            
            self.current_equity = state.get("current_equity", self.initial_equity)
            self.is_halted = state.get("is_halted", False)
            
            if state.get("halt_reason"):
                self.halt_reason = TradingHaltReason(state["halt_reason"])
            
            if "daily_state" in state:
                ds = state["daily_state"]
                self.daily_state.date = ds.get("date", "")
                self.daily_state.starting_equity = ds.get("starting_equity", 0.0)
                self.daily_state.realized_pnl = ds.get("realized_pnl", 0.0)
                self.daily_state.trades_count = ds.get("trades_count", 0)
                self.daily_state.max_drawdown_pct = ds.get("max_drawdown_pct", 0.0)
            
            if "monthly_state" in state:
                ms = state["monthly_state"]
                self.monthly_state.year_month = ms.get("year_month", "")
                self.monthly_state.starting_equity = ms.get("starting_equity", 0.0)
                self.monthly_state.total_realized_pnl = ms.get("total_realized_pnl", 0.0)
                self.monthly_state.trades_count = ms.get("trades_count", 0)
                self.monthly_state.max_drawdown_pct = ms.get("max_drawdown_pct", 0.0)
            
            if "portfolio_drawdown" in state:
                pd = state["portfolio_drawdown"]
                self.portfolio_drawdown.high_water_mark = pd.get("high_water_mark", self.current_equity)
                self.portfolio_drawdown.drawdown_pct = pd.get("drawdown_pct", 0.0)
            
            logger.info("Risk manager state loaded from Redis")
            return True
        except Exception as e:
            logger.error(f"Failed to load risk manager state: {e}")
            return False


def create_risk_manager(
    max_position_pct: float = 1.5,
    max_daily_drawdown: float = 3.0,
    max_monthly_drawdown: float = 8.0,
    initial_equity: float = 10000.0,
    redis_client = None
) -> RiskManager:
    """
    Factory function to create a RiskManager with standard limits.
    
    Args:
        max_position_pct: Maximum position size as % of portfolio (default 1.5%)
        max_daily_drawdown: Maximum daily drawdown % (default 3.0%)
        max_monthly_drawdown: Maximum monthly drawdown % (default 8.0%)
        initial_equity: Starting equity
        redis_client: Optional Redis client
        
    Returns:
        Configured RiskManager instance
    """
    limits = RiskLimits(
        max_position_pct=max_position_pct,
        max_daily_drawdown_pct=max_daily_drawdown,
        max_monthly_drawdown_pct=max_monthly_drawdown
    )
    return RiskManager(limits=limits, initial_equity=initial_equity, redis_client=redis_client)
