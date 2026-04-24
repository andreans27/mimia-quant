"""
Mimia Quant Trading System - Position Sizer

Position sizing using Kelly Criterion and related sizing methods.
Provides adaptive position sizing based on historical trade performance
and current market conditions.

Kelly Criterion: f* = (bp - q) / b
Where:
- f* = Kelly fraction (fraction of bankroll to bet)
- b = odds received on the bet (profit/loss ratio)
- p = probability of winning
- q = probability of losing (1 - p)
"""

import math
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing method enumeration"""
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    QUARTER_KELLY = "quarter_kelly"
    FIXED = "fixed"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"


@dataclass
class TradeRecord:
    """Record of a single trade for analysis"""
    timestamp: float
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    holding_period: float  # in hours
    exit_reason: str  # STOP_LOSS, TAKE_PROFIT, SIGNAL, MANUAL


@dataclass
class KellyStats:
    """Kelly Criterion statistics"""
    win_rate: float  # Probability of winning (p)
    avg_win: float  # Average win amount
    avg_loss: float  # Average loss amount
    loss_rate: float  # Probability of losing (q) = 1 - win_rate
    b: float  # Win/loss ratio = avg_win / avg_loss
    kelly_fraction: float  # Raw Kelly fraction
    effective_kelly: float  # Kelly after safety adjustments
    confidence: float  # Confidence in the estimate (based on sample size)
    sample_size: int


@dataclass
class PositionSize:
    """Calculated position size result"""
    quantity: float
    notional_value: float
    risk_amount: float
    risk_pct: float
    kelly_fraction: float
    sizing_method: SizingMethod
    metadata: Dict[str, Any]


class KellyCriterion:
    """
    Kelly Criterion calculator and analyzer.
    
    The Kelly Criterion determines the optimal fraction of capital to risk
    on a single bet/trade based on the edge and odds.
    """
    
    # Safety multipliers for Kelly
    KELLY_MULTIPLIERS = {
        SizingMethod.KELLY: 1.0,
        SizingMethod.HALF_KELLY: 0.5,
        SizingMethod.QUARTER_KELLY: 0.25,
    }
    
    # Minimum sample size for reliable Kelly calculation
    MIN_SAMPLE_SIZE = 30
    
    # Maximum Kelly fraction to prevent overbetting
    MAX_KELLY_FRACTION = 0.5  # 50% of capital
    
    def __init__(self):
        """Initialize Kelly Criterion calculator"""
        self.trade_history: List[TradeRecord] = []
        self._cached_stats: Optional[KellyStats] = None
        self._cache_valid = False
    
    def add_trade(self, trade: TradeRecord) -> None:
        """Add a trade record for analysis"""
        self.trade_history.append(trade)
        self._cache_valid = False
    
    def add_trades(self, trades: List[TradeRecord]) -> None:
        """Add multiple trade records"""
        self.trade_history.extend(trades)
        self._cache_valid = False
    
    def clear_history(self) -> None:
        """Clear trade history"""
        self.trade_history.clear()
        self._cache_valid = False
        self._cached_stats = None
    
    def get_stats(self, use_cache: bool = True) -> KellyStats:
        """
        Calculate Kelly statistics from trade history.
        
        Args:
            use_cache: Whether to use cached stats if available
            
        Returns:
            KellyStats with all calculated metrics
        """
        if use_cache and self._cache_valid and self._cached_stats:
            return self._cached_stats
        
        if len(self.trade_history) < 5:
            # Not enough data, return conservative defaults
            stats = KellyStats(
                win_rate=0.5,
                avg_win=1.0,
                avg_loss=1.0,
                loss_rate=0.5,
                b=1.0,
                kelly_fraction=0.0,
                effective_kelly=0.0,
                confidence=0.0,
                sample_size=len(self.trade_history)
            )
            self._cached_stats = stats
            self._cache_valid = True
            return stats
        
        # Calculate basic statistics
        wins = [t.pnl for t in self.trade_history if t.pnl > 0]
        losses = [t.pnl for t in self.trade_history if t.pnl < 0]
        all_pnls = [t.pnl for t in self.trade_history]
        
        n_total = len(self.trade_history)
        n_wins = len(wins)
        n_losses = len(losses)
        
        win_rate = n_wins / n_total if n_total > 0 else 0.5
        loss_rate = 1.0 - win_rate
        
        avg_win = sum(wins) / n_wins if n_wins > 0 else 1.0
        avg_loss = abs(sum(losses) / n_losses) if n_losses > 0 else 1.0
        
        # Avoid division by zero
        if avg_loss == 0:
            avg_loss = 1.0
        
        # Win/loss ratio
        b = avg_win / avg_loss
        
        # Kelly formula: f* = (bp - q) / b
        kelly = (b * win_rate - loss_rate) / b
        
        # Confidence based on sample size (logarithmic scaling)
        confidence = min(1.0, math.log(n_total + 1) / math.log(self.MIN_SAMPLE_SIZE * 10))
        
        # Apply confidence adjustment
        adjusted_kelly = kelly * confidence
        
        # Clamp to safe range
        kelly = max(0.0, min(self.MAX_KELLY_FRACTION, kelly))
        adjusted_kelly = max(0.0, min(self.MAX_KELLY_FRACTION, adjusted_kelly))
        
        stats = KellyStats(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            loss_rate=loss_rate,
            b=b,
            kelly_fraction=kelly,
            effective_kelly=adjusted_kelly,
            confidence=confidence,
            sample_size=n_total
        )
        
        self._cached_stats = stats
        self._cache_valid = True
        
        logger.debug(f"Kelly stats updated: win_rate={win_rate:.2%}, b={b:.2f}, "
                    f"kelly={kelly:.4f}, effective_kelly={adjusted_kelly:.4f}, "
                    f"confidence={confidence:.2%}, n={n_total}")
        
        return stats
    
    def calculate_kelly(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        confidence: float = 1.0,
        method: SizingMethod = SizingMethod.HALF_KELLY
    ) -> float:
        """
        Calculate Kelly fraction from given parameters.
        
        Args:
            win_rate: Probability of winning (0-1)
            avg_win: Average winning amount
            avg_loss: Average losing amount
            confidence: Confidence in the estimate (0-1)
            method: Sizing method to apply
            
        Returns:
            Kelly fraction (0-1)
        """
        if avg_loss <= 0:
            return 0.0
        
        b = avg_win / avg_loss
        p = win_rate
        q = 1.0 - p
        
        # Kelly formula
        if b <= 0:
            return 0.0
        
        kelly = (b * p - q) / b
        
        # Clamp
        kelly = max(0.0, min(1.0, kelly))
        
        # Apply confidence adjustment
        kelly *= confidence
        
        # Apply method multiplier
        multiplier = self.KELLY_MULTIPLIERS.get(method, 0.5)
        kelly *= multiplier
        
        return kelly
    
    def estimate_probability(
        self,
        symbol: str,
        lookback_trades: int = 100
    ) -> Tuple[float, float]:
        """
        Estimate win probability for a symbol based on trade history.
        
        Args:
            symbol: Trading symbol
            lookback_trades: Number of recent trades to consider
            
        Returns:
            Tuple of (win_rate, sample_size)
        """
        symbol_trades = [t for t in self.trade_history if t.symbol == symbol]
        recent_trades = symbol_trades[-lookback_trades:] if len(symbol_trades) > lookback_trades else symbol_trades
        
        if not recent_trades:
            return 0.5, 0
        
        wins = sum(1 for t in recent_trades if t.pnl > 0)
        win_rate = wins / len(recent_trades)
        
        return win_rate, len(recent_trades)
    
    def get_recent_performance(
        self,
        window: str = "all"
    ) -> Dict[str, Any]:
        """
        Get recent trading performance metrics.
        
        Args:
            window: Time window - "1d", "1w", "1m", "all"
            
        Returns:
            Dict with performance metrics
        """
        import time
        from datetime import datetime, timedelta
        
        now = time.time()
        
        if window == "1d":
            cutoff = now - 86400
        elif window == "1w":
            cutoff = now - 604800
        elif window == "1m":
            cutoff = now - 2592000
        else:
            cutoff = 0
        
        recent = [t for t in self.trade_history if t.timestamp >= cutoff]
        
        if not recent:
            return {
                "trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_pnl": 0.0,
                "profit_factor": 0.0,
            }
        
        wins = [t.pnl for t in recent if t.pnl > 0]
        losses = [t.pnl for t in recent if t.pnl < 0]
        
        total_pnl = sum(t.pnl for t in recent)
        avg_pnl = total_pnl / len(recent) if recent else 0
        
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "trades": len(recent),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(recent) if recent else 0,
            "total_pnl": total_pnl,
            "avg_pnl": avg_pnl,
            "profit_factor": profit_factor,
            "avg_win": sum(wins) / len(wins) if wins else 0,
            "avg_loss": abs(sum(losses) / len(losses)) if losses else 0,
        }


class PositionSizer:
    """
    Position sizing calculator with multiple sizing methods.
    
    Supports:
    - Kelly Criterion (full, half, quarter)
    - Fixed fractional risk
    - Volatility-adjusted sizing
    - Risk parity
    - Equal weight
    """
    
    def __init__(
        self,
        account_equity: float = 10000.0,
        default_risk_pct: float = 1.0,
        max_position_pct: float = 1.5,
        sizing_method: SizingMethod = SizingMethod.HALF_KELLY
    ):
        """
        Initialize PositionSizer.
        
        Args:
            account_equity: Current account equity
            default_risk_pct: Default risk per trade as % of equity
            max_position_pct: Maximum position size as % of equity
            sizing_method: Default sizing method
        """
        self.account_equity = account_equity
        self.default_risk_pct = default_risk_pct
        self.max_position_pct = max_position_pct
        self.sizing_method = sizing_method
        
        self.kelly = KellyCriterion()
        
        # Rolling volatility estimates
        self.volatility_cache: Dict[str, List[float]] = {}
        self.default_volatility = 0.02  # 2% default daily volatility
        
        # Position history for analysis
        self.position_history: List[Dict[str, Any]] = []
    
    def update_equity(self, new_equity: float) -> None:
        """Update account equity"""
        self.account_equity = new_equity
    
    def set_sizing_method(self, method: SizingMethod) -> None:
        """Change default sizing method"""
        self.sizing_method = method
    
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        side: str = "LONG",
        volatility: Optional[float] = None,
        confidence: float = 1.0,
        method: Optional[SizingMethod] = None,
        risk_override: Optional[float] = None
    ) -> PositionSize:
        """
        Calculate optimal position size.
        
        Args:
            symbol: Trading symbol
            entry_price: Planned entry price
            stop_loss_price: Stop loss price (optional)
            take_profit_price: Take profit price (optional)
            side: Position side (LONG or SHORT)
            volatility: Current volatility estimate (optional)
            confidence: Trade confidence (0-1)
            method: Sizing method override (optional)
            risk_override: Risk amount override in quote currency
            
        Returns:
            PositionSize with calculated quantity and metadata
        """
        method = method or self.sizing_method
        
        # Get Kelly stats
        kelly_stats = self.kelly.get_stats()
        
        # Calculate risk amount
        if risk_override is not None:
            risk_amount = risk_override
        elif stop_loss_price is not None and entry_price > 0:
            # Risk = position size * (entry - stop_loss) for LONG
            # Risk = position size * (stop_loss - entry) for SHORT
            if side == "LONG":
                risk_per_unit = entry_price - stop_loss_price
            else:
                risk_per_unit = stop_loss_price - entry_price
            risk_per_unit = max(0, risk_per_unit)
            risk_pct = self.default_risk_pct
            risk_amount = self.account_equity * (risk_pct / 100)
        else:
            # Use default risk percentage
            risk_amount = self.account_equity * (self.default_risk_pct / 100)
        
        # Calculate base quantity from risk
        if stop_loss_price is not None and entry_price > stop_loss_price:
            if side == "LONG":
                risk_per_unit = entry_price - stop_loss_price
            else:
                risk_per_unit = stop_loss_price - entry_price
            risk_per_unit = max(risk_per_unit, entry_price * 0.001)  # Minimum 0.1% risk
            base_quantity = risk_amount / risk_per_unit
        else:
            base_quantity = risk_amount / entry_price if entry_price > 0 else 0
        
        # Apply Kelly adjustment
        kelly_fraction = kelly_stats.effective_kelly if method in [
            SizingMethod.KELLY, SizingMethod.HALF_KELLY, SizingMethod.QUARTER_KELLY
        ] else 1.0
        
        # Apply confidence adjustment
        adjusted_fraction = kelly_fraction * confidence
        
        # Apply volatility adjustment if using that method
        vol_adjustment = 1.0
        if method == SizingMethod.VOLATILITY_ADJUSTED:
            vol = volatility or self._get_volatility(symbol)
            target_vol = self.default_volatility
            vol_adjustment = target_vol / vol if vol > 0 else 1.0
            vol_adjustment = max(0.25, min(2.0, vol_adjustment))  # Clamp to 0.25x - 2x
            adjusted_fraction *= vol_adjustment
        
        # Calculate final quantity
        quantity = base_quantity * adjusted_fraction
        
        # Apply maximum position limit
        max_notional = (self.account_equity * self.max_position_pct) / 100
        max_quantity = max_notional / entry_price if entry_price > 0 else 0
        quantity = min(quantity, max_quantity)
        
        # Ensure minimum quantity
        quantity = max(quantity, 0.0)
        
        notional_value = quantity * entry_price
        actual_risk = risk_amount * adjusted_fraction
        
        # Calculate risk percentage
        risk_pct = (actual_risk / self.account_equity) * 100 if self.account_equity > 0 else 0
        
        metadata = {
            "kelly_stats": {
                "win_rate": kelly_stats.win_rate,
                "avg_win": kelly_stats.avg_win,
                "avg_loss": kelly_stats.avg_loss,
                "b": kelly_stats.b,
                "kelly_fraction": kelly_stats.kelly_fraction,
                "effective_kelly": kelly_stats.effective_kelly,
                "confidence": kelly_stats.confidence,
            },
            "risk_per_unit": stop_loss_price and abs(entry_price - stop_loss_price) or None,
            "volatility_adjustment": vol_adjustment if method == SizingMethod.VOLATILITY_ADJUSTED else 1.0,
            "confidence": confidence,
            "base_quantity": base_quantity,
        }
        
        # Record for analysis
        self.position_history.append({
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "quantity": quantity,
            "notional_value": notional_value,
            "method": method.value,
            "timestamp": __import__('time').time()
        })
        
        return PositionSize(
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=actual_risk,
            risk_pct=risk_pct,
            kelly_fraction=kelly_fraction * confidence,
            sizing_method=method,
            metadata=metadata
        )
    
    def calculate_kelly_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        side: str = "LONG",
        confidence: float = 1.0
    ) -> PositionSize:
        """
        Calculate position size using Kelly Criterion.
        
        Shorthand for calculate_position_size with Kelly method.
        """
        return self.calculate_position_size(
            symbol="",
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            side=side,
            confidence=confidence,
            method=SizingMethod.KELLY
        )
    
    def calculate_fixed_risk_size(
        self,
        entry_price: float,
        stop_loss_price: float,
        risk_amount: float,
        side: str = "LONG"
    ) -> PositionSize:
        """
        Calculate position size for fixed risk amount.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_amount: Risk amount in quote currency
            side: Position side
            
        Returns:
            PositionSize with fixed risk sizing
        """
        if side == "LONG":
            risk_per_unit = entry_price - stop_loss_price
        else:
            risk_per_unit = stop_loss_price - entry_price
        
        risk_per_unit = max(risk_per_unit, entry_price * 0.001)
        quantity = risk_amount / risk_per_unit
        
        # Apply max position limit
        max_notional = (self.account_equity * self.max_position_pct) / 100
        max_quantity = max_notional / entry_price if entry_price > 0 else 0
        quantity = min(quantity, max_quantity)
        
        notional_value = quantity * entry_price
        actual_risk = quantity * risk_per_unit
        risk_pct = (actual_risk / self.account_equity) * 100 if self.account_equity > 0 else 0
        
        return PositionSize(
            quantity=quantity,
            notional_value=notional_value,
            risk_amount=actual_risk,
            risk_pct=risk_pct,
            kelly_fraction=1.0,
            sizing_method=SizingMethod.FIXED,
            metadata={"risk_amount": risk_amount, "risk_per_unit": risk_per_unit}
        )
    
    def calculate_volatility_adjusted_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss_pct: float,
        side: str = "LONG",
        target_volatility: float = 0.02,
        confidence: float = 1.0
    ) -> PositionSize:
        """
        Calculate position size adjusted for volatility.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_pct: Stop loss as percentage of entry
            side: Position side
            target_volatility: Target volatility (default 2%)
            confidence: Trade confidence (0-1)
            
        Returns:
            PositionSize with volatility-adjusted sizing
        """
        # Calculate stop loss price
        if side == "LONG":
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_pct / 100)
        
        # Get current volatility
        current_vol = self._get_volatility(symbol)
        
        # Volatility adjustment: reduce size when volatility is high
        vol_ratio = target_volatility / current_vol if current_vol > 0 else 1.0
        vol_ratio = max(0.25, min(2.0, vol_ratio))
        
        # Base risk amount with volatility adjustment
        base_risk = self.account_equity * (self.default_risk_pct / 100)
        adjusted_risk = base_risk * vol_ratio
        
        return self.calculate_fixed_risk_size(
            entry_price=entry_price,
            stop_loss_price=stop_loss_price,
            risk_amount=adjusted_risk,
            side=side
        )
    
    def _get_volatility(self, symbol: str) -> float:
        """Get estimated volatility for a symbol"""
        if symbol in self.volatility_cache and len(self.volatility_cache[symbol]) > 0:
            # Return rolling average
            vols = self.volatility_cache[symbol]
            return sum(vols) / len(vols)
        return self.default_volatility
    
    def update_volatility(self, symbol: str, volatility: float) -> None:
        """Update volatility estimate for a symbol"""
        if symbol not in self.volatility_cache:
            self.volatility_cache[symbol] = []
        
        self.volatility_cache[symbol].append(volatility)
        
        # Keep rolling window of 20 estimates
        if len(self.volatility_cache[symbol]) > 20:
            self.volatility_cache[symbol].pop(0)
    
    def calculate_risk_parity_size(
        self,
        symbols: List[str],
        entry_prices: Dict[str, float],
        volatilities: Dict[str, float],
        total_risk_budget: float,
        side: str = "LONG"
    ) -> Dict[str, PositionSize]:
        """
        Calculate position sizes using risk parity method.
        
        Each position contributes equally to total portfolio risk.
        
        Args:
            symbols: List of trading symbols
            entry_prices: Dict of symbol -> entry price
            volatilities: Dict of symbol -> volatility estimate
            total_risk_budget: Total risk budget in quote currency
            side: Position side for all positions
            
        Returns:
            Dict of symbol -> PositionSize
        """
        n_positions = len(symbols)
        if n_positions == 0:
            return {}
        
        # Equal risk allocation per position
        risk_per_position = total_risk_budget / n_positions
        
        results = {}
        for symbol in symbols:
            entry_price = entry_prices.get(symbol, 0)
            vol = volatilities.get(symbol, self.default_volatility)
            
            if entry_price <= 0:
                continue
            
            # Risk per unit based on volatility (ATR-style)
            risk_per_unit = entry_price * vol
            
            quantity = risk_per_position / risk_per_unit if risk_per_unit > 0 else 0
            
            # Apply max position limit
            max_notional = (self.account_equity * self.max_position_pct) / 100
            max_quantity = max_notional / entry_price if entry_price > 0 else 0
            quantity = min(quantity, max_quantity)
            
            notional_value = quantity * entry_price
            actual_risk = quantity * risk_per_unit
            
            results[symbol] = PositionSize(
                quantity=quantity,
                notional_value=notional_value,
                risk_amount=actual_risk,
                risk_pct=(actual_risk / self.account_equity * 100) if self.account_equity > 0 else 0,
                kelly_fraction=1.0,
                sizing_method=SizingMethod.RISK_PARITY,
                metadata={
                    "volatility": vol,
                    "risk_per_unit": risk_per_unit,
                    "target_risk": risk_per_position
                }
            )
        
        return results
    
    def calculate_equal_weight_size(
        self,
        symbols: List[str],
        entry_prices: Dict[str, float],
        side: str = "LONG"
    ) -> Dict[str, PositionSize]:
        """
        Calculate equal-weight position sizes.
        
        Args:
            symbols: List of trading symbols
            entry_prices: Dict of symbol -> entry price
            side: Position side for all positions
            
        Returns:
            Dict of symbol -> PositionSize
        """
        n_positions = len(symbols)
        if n_positions == 0:
            return {}
        
        # Equal notional allocation
        total_equity = self.account_equity * 0.9  # Use 90% of equity
        notional_per_position = total_equity / n_positions
        
        results = {}
        for symbol in symbols:
            entry_price = entry_prices.get(symbol, 0)
            
            if entry_price <= 0:
                continue
            
            quantity = notional_per_position / entry_price
            
            # Apply max position limit
            max_notional = (self.account_equity * self.max_position_pct) / 100
            max_quantity = max_notional / entry_price if entry_price > 0 else 0
            quantity = min(quantity, max_quantity)
            
            notional_value = quantity * entry_price
            
            results[symbol] = PositionSize(
                quantity=quantity,
                notional_value=notional_value,
                risk_amount=0,  # Equal weight doesn't specify risk
                risk_pct=(notional_value / self.account_equity * 100) if self.account_equity > 0 else 0,
                kelly_fraction=1.0,
                sizing_method=SizingMethod.EQUAL_WEIGHT,
                metadata={"target_notional": notional_per_position}
            )
        
        return results
    
    def get_optimal_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        method: SizingMethod = SizingMethod.HALF_KELLY
    ) -> float:
        """
        Get optimal Kelly fraction with safety adjustments.
        
        Args:
            win_rate: Historical win rate (0-1)
            avg_win: Average win amount
            avg_loss: Average loss amount
            method: Kelly variant to use
            
        Returns:
            Optimal fraction to risk (0-1)
        """
        kelly = self.kelly.calculate_kelly(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            method=method
        )
        
        # Additional safety: reduce by confidence
        kelly_stats = self.kelly.get_stats()
        kelly *= kelly_stats.confidence
        
        return max(0.0, min(0.5, kelly))  # Cap at 50%
    
    def get_position_summary(self) -> Dict[str, Any]:
        """Get summary of position sizing history"""
        if not self.position_history:
            return {"total_positions": 0}
        
        total_notional = sum(p["notional_value"] for p in self.position_history)
        avg_notional = total_notional / len(self.position_history)
        
        method_counts = {}
        for p in self.position_history:
            method = p["method"]
            method_counts[method] = method_counts.get(method, 0) + 1
        
        return {
            "total_positions": len(self.position_history),
            "total_notional": total_notional,
            "avg_notional": avg_notional,
            "method_breakdown": method_counts,
            "current_equity": self.account_equity,
        }


def create_position_sizer(
    account_equity: float = 10000.0,
    risk_pct: float = 1.0,
    max_position_pct: float = 1.5,
    method: SizingMethod = SizingMethod.HALF_KELLY
) -> PositionSizer:
    """
    Factory function to create a PositionSizer.
    
    Args:
        account_equity: Starting account equity
        risk_pct: Default risk per trade as %
        max_position_pct: Maximum position as % of equity
        method: Default sizing method
        
    Returns:
        Configured PositionSizer instance
    """
    return PositionSizer(
        account_equity=account_equity,
        default_risk_pct=risk_pct,
        max_position_pct=max_position_pct,
        sizing_method=method
    )
