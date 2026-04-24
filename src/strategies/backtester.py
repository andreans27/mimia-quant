"""
Backtesting engine for Mimia Quant.

Provides a comprehensive framework for testing trading strategies
against historical data with realistic simulation of order execution.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np
from enum import Enum

from ..core.base import BaseStrategy, Signal, Order, Position
from ..core.constants import (
    OrderSide, OrderType, OrderStatus, PositionSide, TimeFrame
)


class TradeDirection(Enum):
    """Trade direction enumeration."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class BacktestTrade:
    """Represents a completed trade in backtesting."""
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: float
    side: TradeDirection
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    strategy_name: str
    signal_id: str


@dataclass
class BacktestMetrics:
    """Performance metrics from backtesting."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_duration: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1% per trade
    slippage_rate: float = 0.0005  # 0.05% slippage
    max_position_size: float = 0.2  # 20% max position
    enable_short_selling: bool = True
    enable_fractional_shares: bool = True
    risk_free_rate: float = 0.02  # 2% annual


class Backtester:
    """
    Backtesting engine for trading strategies.
    
    Provides historical simulation of strategy performance with
    realistic order execution, position tracking, and metrics calculation.
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize the backtester.
        
        Args:
            config: Backtest configuration. Uses defaults if None.
        """
        self.config = config or BacktestConfig()
        
        # State
        self.strategies: Dict[str, BaseStrategy] = {}
        self.current_capital: float = self.config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[Order] = []
        self.completed_trades: List[BacktestTrade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []
        
        # Historical data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        # Current state
        self._current_time: Optional[datetime] = None
        self._current_prices: Dict[str, float] = {}
    
    def register_strategy(self, strategy: BaseStrategy) -> None:
        """
        Register a strategy for backtesting.
        
        Args:
            strategy: Strategy instance to register.
        """
        self.strategies[strategy.name] = strategy
    
    def load_data(self, symbol: str, data: pd.DataFrame, timeframe: TimeFrame = TimeFrame.HOUR_1) -> None:
        """
        Load historical data for a symbol.
        
        Args:
            symbol: Trading symbol.
            data: DataFrame with OHLCV data.
            timeframe: Data timeframe.
        """
        self.data_cache[symbol] = data.copy()
    
    def run(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        data_provider: Optional[Callable[[str, datetime, datetime], pd.DataFrame]] = None,
    ) -> Dict[str, BacktestMetrics]:
        """
        Run backtest for registered strategies.
        
        Args:
            symbols: List of symbols to trade.
            start_date: Start date for backtest.
            end_date: End date for backtest.
            data_provider: Optional function to fetch data on the fly.
        
        Returns:
            Dictionary of metrics keyed by strategy name.
        """
        if not self.strategies:
            raise ValueError("No strategies registered")
        
        # Reset state
        self._reset_state()
        
        # Get combined date range
        all_dates = []
        for symbol in symbols:
            if symbol in self.data_cache:
                df = self.data_cache[symbol]
                if isinstance(df.index, pd.DatetimeIndex):
                    all_dates.extend(df.index.tolist())
                elif "timestamp" in df.columns:
                    all_dates.extend(pd.to_datetime(df["timestamp"]).tolist())
        
        if not all_dates:
            raise ValueError("No data available")
        
        all_dates = sorted(set(all_dates))
        
        # Filter by date range
        if start_date:
            all_dates = [d for d in all_dates if d >= start_date]
        if end_date:
            all_dates = [d for d in all_dates if d <= end_date]
        
        if not all_dates:
            raise ValueError("No data in specified date range")
        
        # Run backtest for each timestamp
        for timestamp in all_dates:
            self._current_time = timestamp
            self._update_prices(symbols, timestamp, data_provider)
            self._process_signals(symbols)
            self._update_positions()
            self._update_equity()
        
        # Calculate final metrics
        return self._calculate_metrics()
    
    def _reset_state(self) -> None:
        """Reset backtest state."""
        self.current_capital = self.config.initial_capital
        self.positions = {}
        self.pending_orders = []
        self.completed_trades = []
        self.equity_curve = [self.current_capital]
        self.daily_returns = []
    
    def _update_prices(
        self,
        symbols: List[str],
        timestamp: datetime,
        data_provider: Optional[Callable] = None,
    ) -> None:
        """Update current prices from data cache or provider."""
        for symbol in symbols:
            if symbol in self.data_cache:
                df = self.data_cache[symbol]
                
                # Find the row for this timestamp
                if isinstance(df.index, pd.DatetimeIndex):
                    mask = df.index == timestamp
                elif "timestamp" in df.columns:
                    mask = pd.to_datetime(df["timestamp"]) == timestamp
                else:
                    continue
                
                if mask.any():
                    row = df[mask].iloc[0]
                    close_col = "close" if "close" in df.columns else "Close"
                    self._current_prices[symbol] = float(row[close_col])
                else:
                    # Use most recent price before timestamp
                    if isinstance(df.index, pd.DatetimeIndex):
                        past = df[df.index <= timestamp]
                    elif "timestamp" in df.columns:
                        past = df[pd.to_datetime(df["timestamp"]) <= timestamp]
                    else:
                        past = df
                    
                    if len(past) > 0:
                        close_col = "close" if "close" in df.columns else "Close"
                        self._current_prices[symbol] = float(past.iloc[-1][close_col])
            
            elif data_provider:
                self._current_prices[symbol] = data_provider(symbol, timestamp, timestamp)
    
    def _process_signals(self, symbols: List[str]) -> None:
        """Process signals from all strategies."""
        for strategy_name, strategy in self.strategies.items():
            if not strategy.enabled:
                continue
            
            for symbol in symbols:
                if symbol not in self._current_prices:
                    continue
                
                # Get data for this symbol
                if symbol in self.data_cache:
                    df = self.data_cache[symbol]
                    
                    # Get data up to current timestamp
                    if isinstance(df.index, pd.DatetimeIndex):
                        data = df[df.index <= self._current_time]
                    elif "timestamp" in df.columns:
                        data = df[pd.to_datetime(df["timestamp"]) <= self._current_time]
                    else:
                        data = df
                    
                    if len(data) < 2:
                        continue
                    
                    # Generate signal
                    signal = strategy.analyze(symbol, data)
                    
                    if signal and strategy.validate_signal(signal):
                        self._execute_signal(signal, strategy)
    
    def _execute_signal(self, signal: Signal, strategy: BaseStrategy) -> None:
        """Execute a trading signal."""
        current_price = self._current_prices.get(signal.symbol, 0)
        
        if current_price <= 0:
            return
        
        # Calculate position size
        position_size = strategy.calculate_position_size(signal, self.current_capital)
        position_size = min(position_size, self.config.max_position_size)
        
        # Calculate quantity
        quantity = (self.current_capital * position_size) / current_price
        
        if not self.config.enable_fractional_shares:
            quantity = int(quantity)
        
        if quantity <= 0:
            return
        
        # Calculate costs
        commission = quantity * current_price * self.config.commission_rate
        slippage = quantity * current_price * self.config.slippage_rate
        total_cost = commission + slippage
        
        # Check if we can afford it
        if signal.side == OrderSide.BUY:
            required = quantity * current_price + total_cost
            if required > self.current_capital:
                quantity = (self.current_capital - total_cost) / current_price
                quantity = max(0, quantity)
                if not self.config.enable_fractional_shares:
                    quantity = int(quantity)
        
        if quantity <= 0:
            return
        
        # Update capital for purchase
        if signal.side == OrderSide.BUY:
            cost = quantity * current_price + total_cost
            self.current_capital -= cost
        else:
            # For sells, we receive money
            revenue = quantity * current_price - total_cost
            self.current_capital += revenue
        
        # Create or update position
        self._update_position_from_signal(signal, quantity, current_price)
    
    def _update_position_from_signal(
        self,
        signal: Signal,
        quantity: float,
        price: float,
    ) -> None:
        """Update position based on signal."""
        symbol = signal.symbol
        existing = self.positions.get(symbol)
        
        if signal.side == OrderSide.BUY:
            if existing and existing.side == PositionSide.LONG:
                # Add to long position
                total_qty = existing.quantity + quantity
                avg_price = (existing.entry_price * existing.quantity + price * quantity) / total_qty
                existing.quantity = total_qty
                existing.entry_price = avg_price
                existing.current_price = price
                existing.updated_at = self._current_time
            elif existing and existing.side == PositionSide.SHORT:
                # Cover short position
                if quantity >= existing.quantity:
                    # Fully cover
                    pnl = (existing.entry_price - price) * existing.quantity
                    self.current_capital += pnl
                    self._close_position(symbol, signal.strategy_name, price)
                else:
                    # Partially cover
                    pnl = (existing.entry_price - price) * quantity
                    self.current_capital += pnl
                    existing.quantity -= quantity
                    existing.current_price = price
            else:
                # New long position
                position = Position(
                    symbol=symbol,
                    side=PositionSide.LONG,
                    quantity=quantity,
                    entry_price=price,
                    current_price=price,
                    opened_at=self._current_time,
                    strategy_name=signal.strategy_name,
                )
                self.positions[symbol] = position
        
        elif signal.side == OrderSide.SELL:
            if existing and existing.side == PositionSide.SHORT:
                # Add to short position
                total_qty = existing.quantity + quantity
                avg_price = (existing.entry_price * existing.quantity + price * quantity) / total_qty
                existing.quantity = total_qty
                existing.entry_price = avg_price
                existing.current_price = price
                existing.updated_at = self._current_time
            elif existing and existing.side == PositionSide.LONG:
                # Take profit or stop loss
                if quantity >= existing.quantity:
                    # Fully close
                    pnl = (price - existing.entry_price) * existing.quantity
                    self.current_capital += pnl
                    self._close_position(symbol, signal.strategy_name, price)
                else:
                    # Partially close
                    pnl = (price - existing.entry_price) * quantity
                    self.current_capital += pnl
                    existing.quantity -= quantity
                    existing.current_price = price
            else:
                # New short position (if enabled)
                if self.config.enable_short_selling:
                    position = Position(
                        symbol=symbol,
                        side=PositionSide.SHORT,
                        quantity=quantity,
                        entry_price=price,
                        current_price=price,
                        opened_at=self._current_time,
                        strategy_name=signal.strategy_name,
                    )
                    self.positions[symbol] = position
    
    def _close_position(self, symbol: str, strategy_name: str, exit_price: float) -> None:
        """Close a position and record trade."""
        position = self.positions.pop(symbol, None)
        
        if position is None:
            return
        
        # Record trade
        pnl = 0.0
        if position.side == PositionSide.LONG:
            pnl = (exit_price - position.entry_price) * position.quantity
        elif position.side == PositionSide.SHORT:
            pnl = (position.entry_price - exit_price) * position.quantity
        
        trade = BacktestTrade(
            entry_time=position.opened_at,
            exit_time=self._current_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            side=TradeDirection.LONG if position.side == PositionSide.LONG else TradeDirection.SHORT,
            pnl=pnl,
            pnl_pct=pnl / (position.entry_price * position.quantity) * 100,
            commission=position.quantity * (position.entry_price + exit_price) * self.config.commission_rate,
            slippage=position.quantity * exit_price * self.config.slippage_rate,
            strategy_name=strategy_name,
            signal_id="",
        )
        
        self.completed_trades.append(trade)
    
    def _update_positions(self) -> None:
        """Update position prices and check for stop losses."""
        for symbol, position in list(self.positions.items()):
            if symbol in self._current_prices:
                position.current_price = self._current_prices[symbol]
                position.updated_at = self._current_time
                
                # Check if position should be closed (simple stop loss)
                stop_loss_pct = self.config.max_position_size * 0.5  # 50% of max position as stop
                
                if position.side == PositionSide.LONG:
                    loss_pct = (position.entry_price - position.current_price) / position.entry_price
                    if loss_pct >= stop_loss_pct:
                        self._close_position(symbol, position.strategy_name, position.current_price)
                
                elif position.side == PositionSide.SHORT:
                    loss_pct = (position.current_price - position.entry_price) / position.entry_price
                    if loss_pct >= stop_loss_pct:
                        self._close_position(symbol, position.strategy_name, position.current_price)
    
    def _update_equity(self) -> None:
        """Update equity curve."""
        position_value = 0.0
        for symbol, position in self.positions.items():
            if position.side != PositionSide.NEUTRAL:
                price = self._current_prices.get(symbol, position.current_price)
                position_value += position.quantity * price
        
        total_equity = self.current_capital + position_value
        self.equity_curve.append(total_equity)
    
    def _calculate_metrics(self) -> Dict[str, BacktestMetrics]:
        """Calculate performance metrics for all strategies."""
        metrics_by_strategy = {}
        
        for strategy_name in self.strategies.keys():
            strategy_trades = [t for t in self.completed_trades if t.strategy_name == strategy_name]
            
            if not strategy_trades:
                metrics_by_strategy[strategy_name] = BacktestMetrics()
                continue
            
            metrics = self._calculate_strategy_metrics(strategy_trades)
            metrics_by_strategy[strategy_name] = metrics
        
        return metrics_by_strategy
    
    def _calculate_strategy_metrics(self, trades: List[BacktestTrade]) -> BacktestMetrics:
        """Calculate metrics for a single strategy."""
        if not trades:
            return BacktestMetrics()
        
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        
        total_pnl = sum(t.pnl for t in trades)
        avg_win = sum(t.pnl for t in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(t.pnl for t in losing_trades) / loss_count if loss_count > 0 else 0
        
        best_trade = max(t.pnl for t in trades)
        worst_trade = min(t.pnl for t in trades)
        
        # Win rate
        win_rate = win_count / total_trades if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades) if winning_trades else 0
        gross_loss = abs(sum(t.pnl for t in losing_trades)) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdowns = (running_max - equity) / running_max
        max_drawdown = np.max(drawdowns)
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        if len(returns) > 0:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            avg_return = np.mean(returns) * 252  # Annualized
            
            # Sharpe ratio
            if volatility > 0:
                sharpe_ratio = (avg_return - self.config.risk_free_rate) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_std = np.std(downside_returns) * np.sqrt(252)
                sortino_ratio = (avg_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0.0
            else:
                sortino_ratio = float('inf')
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            sortino_ratio = 0.0
            avg_return = 0.0
        
        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]  # hours
        avg_duration = np.mean(durations) if durations else 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=win_count,
            losing_trades=loss_count,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_pct=total_pnl / self.config.initial_capital * 100,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown * 100,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_trade_duration=avg_duration,
            annualized_return=avg_return * 100,
            volatility=volatility * 100,
        )
    
    def get_equity_curve(self) -> pd.Series:
        """
        Get the equity curve.
        
        Returns:
            Series of equity values over time.
        """
        return pd.Series(self.equity_curve)
    
    def get_trades(self, strategy_name: Optional[str] = None) -> List[BacktestTrade]:
        """
        Get completed trades.
        
        Args:
            strategy_name: Filter by strategy name.
        
        Returns:
            List of completed trades.
        """
        if strategy_name:
            return [t for t in self.completed_trades if t.strategy_name == strategy_name]
        return self.completed_trades
    
    def generate_report(self, metrics: Dict[str, BacktestMetrics]) -> str:
        """
        Generate a text report of backtest results.
        
        Args:
            metrics: Dictionary of metrics by strategy name.
        
        Returns:
            Formatted report string.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("BACKTEST RESULTS")
        lines.append("=" * 70)
        lines.append("")
        
        for strategy_name, m in metrics.items():
            lines.append(f"Strategy: {strategy_name}")
            lines.append("-" * 50)
            lines.append(f"  Total Trades:        {m.total_trades}")
            lines.append(f"  Winning Trades:     {m.winning_trades}")
            lines.append(f"  Losing Trades:      {m.losing_trades}")
            lines.append(f"  Win Rate:           {m.win_rate:.2%}")
            lines.append(f"  Total PnL:          ${m.total_pnl:.2f} ({m.total_pnl_pct:.2f}%)")
            lines.append(f"  Average Win:        ${m.avg_win:.2f}")
            lines.append(f"  Average Loss:       ${m.avg_loss:.2f}")
            lines.append(f"  Best Trade:         ${m.best_trade:.2f}")
            lines.append(f"  Worst Trade:        ${m.worst_trade:.2f}")
            lines.append(f"  Max Drawdown:       {m.max_drawdown_pct:.2f}%")
            lines.append(f"  Sharpe Ratio:       {m.sharpe_ratio:.2f}")
            lines.append(f"  Sortino Ratio:      {m.sortino_ratio:.2f}")
            lines.append(f"  Profit Factor:      {m.profit_factor:.2f}")
            lines.append(f"  Avg Trade Duration: {m.avg_trade_duration:.1f} hours")
            lines.append(f"  Annualized Return:  {m.annualized_return:.2f}%")
            lines.append(f"  Volatility:         {m.volatility:.2f}%")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
