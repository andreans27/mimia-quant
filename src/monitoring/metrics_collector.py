"""
Metrics Collector for Mimia Quant Trading System.

Collects and stores trading metrics, performance data, and system health indicators.
Supports both Redis caching and SQLite database persistence.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
import statistics

from ..core.redis_client import RedisClient
from ..core.database import Database, StrategyPerformance, EquityCurve


class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOL = "high_vol"
    LOW_VOL = "low_vol"


@dataclass
class TradeMetrics:
    """Trade-level metrics."""
    trade_id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    executed_at: datetime
    exit_at: Optional[datetime] = None
    holding_period_seconds: Optional[float] = None
    strategy_name: str = ""


@dataclass
class StrategyMetrics:
    """Strategy-level aggregated metrics."""
    strategy_name: str
    session_id: str
    timestamp: datetime
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility: float = 0.0
    risk_reward_ratio: float = 0.0
    avg_trade_duration: float = 0.0
    expectancy: float = 0.0


@dataclass
class PortfolioMetrics:
    """Portfolio-level metrics."""
    timestamp: datetime
    total_equity: float
    cash: float
    positions_value: float
    daily_pnl: float
    daily_return: float
    cumulative_return: float
    drawdown: float
    drawdown_pct: float
    leverage: float = 1.0
    exposure: float = 0.0


@dataclass
class HealthMetrics:
    """System health metrics."""
    timestamp: datetime
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    redis_connected: bool = False
    database_connected: bool = False
    api_latency_ms: float = 0.0
    error_count: int = 0
    warning_count: int = 0
    active_strategies: int = 0
    open_positions: int = 0


class MetricsCollector:
    """
    Collects and stores trading metrics.
    
    Supports:
    - Real-time metrics via Redis
    - Persistent storage via SQLite
    - Sliding window calculations
    - Regime detection
    """
    
    REDIS_PREFIX = "metrics:"
    
    def __init__(
        self,
        redis_client: Optional[RedisClient] = None,
        database: Optional[Database] = None
    ):
        self.redis = redis_client
        self.db = database
        self._lock = threading.RLock()
        
        # In-memory cache for fast access
        self._trade_cache: Dict[str, TradeMetrics] = {}
        self._recent_trades: List[TradeMetrics] = []
        self._equity_history: List[PortfolioMetrics] = []
        
        # Configuration
        self._max_cached_trades = 1000
        self._equity_history_max = 10000
        
    def set_redis_client(self, redis_client: RedisClient) -> None:
        """Set the Redis client."""
        self.redis = redis_client
    
    def set_database(self, database: Database) -> None:
        """Set the database instance."""
        self.db = database
    
    # ==================== Trade Metrics ====================
    
    def record_trade(self, trade: TradeMetrics) -> bool:
        """
        Record a completed trade.
        
        Args:
            trade: TradeMetrics object with trade details
            
        Returns:
            True if recorded successfully
        """
        with self._lock:
            try:
                # Update cache
                self._trade_cache[trade.trade_id] = trade
                self._recent_trades.append(trade)
                
                # Trim if needed
                if len(self._recent_trades) > self._max_cached_trades:
                    self._recent_trades = self._recent_trades[-self._max_cached_trades:]
                
                # Store in Redis for real-time access
                if self.redis:
                    key = f"{self.REDIS_PREFIX}trade:{trade.trade_id}"
                    self.redis.set(key, asdict(trade), ttl=86400)  # 24h TTL
                    
                    # Add to recent trades list
                    self.redis.lpush(
                        f"{self.REDIS_PREFIX}recent_trades:{trade.strategy_name}",
                        trade.trade_id,
                        json_encode=False
                    )
                    self.redis.ltrim(
                        f"{self.REDIS_PREFIX}recent_trades:{trade.strategy_name}",
                        0,
                        999
                    )
                
                # Persist to database
                if self.db:
                    self._persist_trade(trade)
                
                return True
                
            except Exception as e:
                print(f"Error recording trade: {e}")
                return False
    
    def _persist_trade(self, trade: TradeMetrics) -> None:
        """Persist trade to database."""
        # This would use SQLAlchemy to insert
        # Implementation depends on having a Trades model
        pass
    
    def get_trade(self, trade_id: str) -> Optional[TradeMetrics]:
        """Get a trade by ID."""
        # Check cache first
        if trade_id in self._trade_cache:
            return self._trade_cache[trade_id]
        
        # Check Redis
        if self.redis:
            key = f"{self.REDIS_PREFIX}trade:{trade_id}"
            data = self.redis.get(key)
            if data:
                return TradeMetrics(**data)
        
        return None
    
    def get_recent_trades(
        self,
        strategy_name: str,
        limit: int = 100
    ) -> List[TradeMetrics]:
        """Get recent trades for a strategy."""
        trades = []
        
        if self.redis:
            trade_ids = self.redis.lrange(
                f"{self.REDIS_PREFIX}recent_trades:{strategy_name}",
                0,
                limit - 1,
                json_decode=False
            )
            for tid in trade_ids:
                trade = self.get_trade(tid)
                if trade:
                    trades.append(trade)
        
        return trades
    
    # ==================== Strategy Metrics ====================
    
    def calculate_strategy_metrics(
        self,
        strategy_name: str,
        session_id: str,
        lookback_trades: int = 100
    ) -> StrategyMetrics:
        """
        Calculate aggregated strategy metrics.
        
        Args:
            strategy_name: Name of the strategy
            session_id: Trading session ID
            lookback_trades: Number of recent trades to analyze
            
        Returns:
            StrategyMetrics with calculated values
        """
        with self._lock:
            trades = self.get_recent_trades(strategy_name, lookback_trades)
            
            if not trades:
                return StrategyMetrics(
                    strategy_name=strategy_name,
                    session_id=session_id,
                    timestamp=datetime.utcnow()
                )
            
            # Calculate metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            
            total_pnl = sum(t.pnl for t in trades)
            avg_win = statistics.mean([t.pnl for t in winning_trades]) if winning_trades else 0.0
            avg_loss = abs(statistics.mean([t.pnl for t in losing_trades])) if losing_trades else 0.0
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = sum(t.pnl for t in losing_trades)
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # Expectancy
            expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
            
            # Risk-reward ratio
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 0.0
            
            # Drawdown calculation
            cumulative_pnl = []
            running_total = 0
            for t in trades:
                running_total += t.pnl
                cumulative_pnl.append(running_total)
            
            peak = cumulative_pnl[0]
            max_drawdown = 0.0
            for val in cumulative_pnl:
                if val > peak:
                    peak = val
                drawdown = peak - val
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            max_drawdown_pct = (max_drawdown / peak * 100) if peak > 0 else 0.0
            
            # Volatility (standard deviation of returns)
            returns = [t.pnl_pct for t in trades if t.pnl_pct != 0]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0.0
            
            # Sharpe ratio (simplified, assuming 0% risk-free rate)
            avg_return = statistics.mean(returns) if returns else 0.0
            sharpe = (avg_return / volatility) if volatility > 0 else 0.0
            
            # Sortino ratio (using downside deviation)
            downside_returns = [r for r in returns if r < 0]
            downside_dev = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0.0
            sortino = (avg_return / downside_dev) if downside_dev > 0 else 0.0
            
            # Calmar ratio (return / max drawdown)
            total_return_pct = sum(returns)
            calmar = (total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0
            
            # Average trade duration
            durations = [t.holding_period_seconds for t in trades if t.holding_period_seconds]
            avg_duration = statistics.mean(durations) if durations else 0.0
            
            metrics = StrategyMetrics(
                strategy_name=strategy_name,
                session_id=session_id,
                timestamp=datetime.utcnow(),
                total_trades=total_trades,
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=win_rate,
                total_pnl=total_pnl,
                total_pnl_pct=total_return_pct,
                avg_win=avg_win,
                avg_loss=avg_loss,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                max_drawdown_pct=max_drawdown_pct,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                calmar_ratio=calmar,
                volatility=volatility,
                risk_reward_ratio=risk_reward,
                avg_trade_duration=avg_duration,
                expectancy=expectancy
            )
            
            # Cache in Redis
            if self.redis:
                key = f"{self.REDIS_PREFIX}strategy_metrics:{strategy_name}:{session_id}"
                self.redis.set(key, asdict(metrics), ttl=300)  # 5 min TTL
            
            return metrics
    
    def record_strategy_metrics(self, metrics: StrategyMetrics) -> bool:
        """Record strategy metrics to database."""
        if not self.db:
            return False
        
        try:
            perf = StrategyPerformance(
                strategy_name=metrics.strategy_name,
                session_id=metrics.session_id,
                timestamp=metrics.timestamp,
                total_trades=metrics.total_trades,
                winning_trades=metrics.winning_trades,
                losing_trades=metrics.losing_trades,
                win_rate=metrics.win_rate,
                total_pnl=metrics.total_pnl,
                total_pnl_pct=metrics.total_pnl_pct,
                avg_win=metrics.avg_win,
                avg_loss=metrics.avg_loss,
                profit_factor=metrics.profit_factor,
                max_drawdown=metrics.max_drawdown,
                max_drawdown_pct=metrics.max_drawdown_pct,
                sharpe_ratio=metrics.sharpe_ratio,
                sortino_ratio=metrics.sortino_ratio,
                calmar_ratio=metrics.calmar_ratio,
                volatility=metrics.volatility,
                risk_reward_ratio=metrics.risk_reward_ratio
            )
            
            session = self.db.get_session()
            session.add(perf)
            session.commit()
            session.close()
            return True
            
        except Exception as e:
            print(f"Error recording strategy metrics: {e}")
            return False
    
    # ==================== Portfolio Metrics ====================
    
    def record_portfolio_metrics(self, metrics: PortfolioMetrics) -> bool:
        """Record portfolio metrics."""
        with self._lock:
            self._equity_history.append(metrics)
            
            if len(self._equity_history) > self._equity_history_max:
                self._equity_history = self._equity_history[-self._equity_history_max:]
            
            if self.redis:
                key = f"{self.REDIS_PREFIX}portfolio_metrics"
                self.redis.set(key, asdict(metrics), ttl=60)
            
            if self.db:
                try:
                    equity = EquityCurve(
                        timestamp=metrics.timestamp,
                        equity=metrics.total_equity,
                        cash=metrics.cash,
                        positions_value=metrics.positions_value,
                        total_value=metrics.total_equity,
                        daily_pnl=metrics.daily_pnl,
                        daily_return=metrics.daily_return,
                        cumulative_return=metrics.cumulative_return,
                        drawdown=metrics.drawdown
                    )
                    session = self.db.get_session()
                    session.add(equity)
                    session.commit()
                    session.close()
                except Exception as e:
                    print(f"Error recording equity curve: {e}")
            
            return True
    
    def get_portfolio_metrics(self) -> Optional[PortfolioMetrics]:
        """Get latest portfolio metrics."""
        if self._equity_history:
            return self._equity_history[-1]
        
        if self.redis:
            data = self.redis.get(f"{self.REDIS_PREFIX}portfolio_metrics")
            if data:
                return PortfolioMetrics(**data)
        
        return None
    
    # ==================== Health Metrics ====================
    
    def record_health_metrics(self, metrics: HealthMetrics) -> bool:
        """Record system health metrics."""
        if self.redis:
            key = f"{self.REDIS_PREFIX}health"
            self.redis.set(key, asdict(metrics), ttl=60)
        return True
    
    def get_health_metrics(self) -> Optional[HealthMetrics]:
        """Get latest health metrics."""
        if self.redis:
            data = self.redis.get(f"{self.REDIS_PREFIX}health")
            if data:
                return HealthMetrics(**data)
        return None
    
    # ==================== Regime Detection ====================
    
    def detect_regime(
        self,
        symbol: str,
        lookback_bars: int = 100
    ) -> RegimeType:
        """
        Detect current market regime based on price action.
        
        Uses multiple indicators:
        - Trend direction (SMA slope)
        - Volatility (ATR or std dev)
        - Range bounds (Bollinger position)
        
        Args:
            symbol: Trading symbol
            lookback_bars: Number of bars to analyze
            
        Returns:
            Detected RegimeType
        """
        # In production, this would fetch real market data
        # For now, return a default based on analysis
        return RegimeType.RANGING
    
    def calculate_regime_confidence(
        self,
        symbol: str
    ) -> Tuple[RegimeType, float]:
        """
        Calculate regime with confidence score.
        
        Returns:
            Tuple of (RegimeType, confidence 0-1)
        """
        regime = self.detect_regime(symbol)
        confidence = 0.75  # Default confidence
        
        return regime, confidence
    
    # ==================== Historical Analysis ====================
    
    def get_performance_history(
        self,
        strategy_name: str,
        days: int = 30
    ) -> List[StrategyMetrics]:
        """Get historical performance for a strategy."""
        if not self.db:
            return []
        
        try:
            session = self.db.get_session()
            cutoff = datetime.utcnow() - timedelta(days=days)
            
            records = session.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_name == strategy_name,
                StrategyPerformance.timestamp >= cutoff
            ).order_by(StrategyPerformance.timestamp).all()
            
            session.close()
            
            metrics_list = []
            for r in records:
                metrics_list.append(StrategyMetrics(
                    strategy_name=r.strategy_name,
                    session_id=r.session_id,
                    timestamp=r.timestamp,
                    total_trades=r.total_trades,
                    winning_trades=r.winning_trades,
                    losing_trades=r.losing_trades,
                    win_rate=r.win_rate,
                    total_pnl=r.total_pnl,
                    total_pnl_pct=r.total_pnl_pct,
                    avg_win=r.avg_win,
                    avg_loss=r.avg_loss,
                    profit_factor=r.profit_factor,
                    max_drawdown=r.max_drawdown,
                    max_drawdown_pct=r.max_drawdown_pct,
                    sharpe_ratio=r.sharpe_ratio,
                    sortino_ratio=r.sortino_ratio,
                    calmar_ratio=r.calmar_ratio,
                    volatility=r.volatility,
                    risk_reward_ratio=r.risk_reward_ratio
                ))
            
            return metrics_list
            
        except Exception as e:
            print(f"Error fetching performance history: {e}")
            return []
    
    def get_equity_curve(
        self,
        strategy_name: Optional[str] = None,
        session_id: Optional[str] = None,
        days: int = 30
    ) -> List[Dict]:
        """Get equity curve data."""
        if not self.db:
            return []
        
        try:
            session = self.db.get_session()
            query = session.query(EquityCurve)
            
            if strategy_name:
                query = query.filter(EquityCurve.strategy_name == strategy_name)
            if session_id:
                query = query.filter(EquityCurve.session_id == session_id)
            
            cutoff = datetime.utcnow() - timedelta(days=days)
            records = query.filter(
                EquityCurve.timestamp >= cutoff
            ).order_by(EquityCurve.timestamp).all()
            
            session.close()
            
            return [
                {
                    "timestamp": r.timestamp,
                    "equity": r.equity,
                    "total_value": r.total_value,
                    "daily_pnl": r.daily_pnl,
                    "daily_return": r.daily_return,
                    "cumulative_return": r.cumulative_return,
                    "drawdown": r.drawdown
                }
                for r in records
            ]
            
        except Exception as e:
            print(f"Error fetching equity curve: {e}")
            return []
    
    # ==================== Summary Statistics ====================
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all strategies."""
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_trades_today": 0,
            "total_pnl_today": 0.0,
            "best_strategy": None,
            "worst_strategy": None,
            "overall_win_rate": 0.0,
            "overall_profit_factor": 0.0
        }
        
        # This would aggregate from all strategies
        return stats
