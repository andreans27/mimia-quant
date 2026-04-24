"""
Main Monitor class for Mimia Quant Trading System.

Ties together all monitoring components and provides a unified interface
for system monitoring, alerting, and reporting.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
import threading

from .metrics_collector import (
    MetricsCollector,
    StrategyMetrics,
    PortfolioMetrics,
    HealthMetrics,
    RegimeType,
    TradeMetrics
)
from .reporter import Reporter, ReportConfig
from .telegram_notifier import TelegramNotifier, AlertLevel
from .edge_decay_detector import (
    EdgeDecayDetector,
    EdgeStatus,
    DecayAlert,
    DecayType,
    DecayThreshold
)


logger = logging.getLogger(__name__)


@dataclass
class MonitorConfig:
    """Configuration for the Monitor."""
    check_interval_seconds: int = 60  # Main monitoring loop interval
    edge_check_interval_seconds: int = 300  # Edge decay check interval (5 min)
    report_interval_seconds: int = 3600  # Report generation interval (1 hour)
    daily_report_hour_utc: int = 8  # Hour to send daily report (8 AM UTC)
    
    # Thresholds
    win_rate_drop_threshold: float = 10.0  # 10% drop
    profit_factor_threshold: float = 1.5
    sharpe_threshold: float = 1.0
    max_drawdown_threshold: float = 15.0
    
    # Auto-actions
    auto_pause_on_critical: bool = True
    auto_reduce_size_on_decay: bool = True
    reduction_factor: float = 0.5  # Reduce position size by 50% on decay


class Monitor:
    """
    Main monitoring system that coordinates all monitoring components.
    
    Provides:
    - Centralized monitoring coordination
    - Automated checks and alerts
    - Integration between components
    - Background monitoring threads
    """
    
    def __init__(
        self,
        config: Optional[MonitorConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        telegram_notifier: Optional[TelegramNotifier] = None
    ):
        """
        Initialize Monitor.
        
        Args:
            config: Monitor configuration
            metrics_collector: MetricsCollector instance
            telegram_notifier: TelegramNotifier instance
        """
        self.config = config or MonitorConfig()
        
        # Initialize components
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.edge_detector = EdgeDecayDetector(
            metrics_collector=self.metrics_collector
        )
        self.reporter = Reporter(
            metrics_collector=self.metrics_collector,
            edge_detector=self.edge_detector,
            telegram_notifier=telegram_notifier
        )
        self.telegram = telegram_notifier
        
        # State
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_edge_check = datetime.min
        self._last_report_time = datetime.min
        self._strategies: List[str] = []
        
        # Callbacks
        self._pause_callback: Optional[Callable[[str], None]] = None
        self._reduce_size_callback: Optional[Callable[[str, float], None]] = None
        
        # Lock
        self._lock = threading.RLock()
        
        # Alert history
        self._alert_history: List[Dict[str, Any]] = []
    
    # ==================== Component Setters ====================
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set metrics collector."""
        self.metrics_collector = collector
        self.edge_detector.set_metrics_collector(collector)
        self.reporter.set_metrics_collector(collector)
    
    def set_telegram_notifier(self, notifier: TelegramNotifier) -> None:
        """Set Telegram notifier."""
        self.telegram = notifier
        self.reporter.set_telegram_notifier(notifier)
    
    def set_pause_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for pausing strategies."""
        self._pause_callback = callback
    
    def set_reduce_size_callback(self, callback: Callable[[str, float], None]) -> None:
        """Set callback for reducing position sizes."""
        self._reduce_size_callback = callback
    
    # ==================== Strategy Management ====================
    
    def register_strategy(self, strategy_name: str) -> None:
        """Register a strategy for monitoring."""
        with self._lock:
            if strategy_name not in self._strategies:
                self._strategies.append(strategy_name)
                logger.info(f"Registered strategy for monitoring: {strategy_name}")
    
    def unregister_strategy(self, strategy_name: str) -> None:
        """Unregister a strategy from monitoring."""
        with self._lock:
            if strategy_name in self._strategies:
                self._strategies.remove(strategy_name)
                self.edge_detector.clear_all_alerts(strategy_name)
                logger.info(f"Unregistered strategy from monitoring: {strategy_name}")
    
    def get_registered_strategies(self) -> List[str]:
        """Get list of registered strategies."""
        with self._lock:
            return list(self._strategies)
    
    # ==================== Monitoring Operations ====================
    
    def record_trade(self, trade: TradeMetrics) -> bool:
        """Record a completed trade."""
        return self.metrics_collector.record_trade(trade)
    
    def record_portfolio_metrics(
        self,
        total_equity: float,
        cash: float,
        positions_value: float,
        daily_pnl: float,
        daily_return: float,
        cumulative_return: float,
        drawdown: float,
        leverage: float = 1.0,
        exposure: float = 0.0
    ) -> bool:
        """Record portfolio metrics."""
        metrics = PortfolioMetrics(
            timestamp=datetime.utcnow(),
            total_equity=total_equity,
            cash=cash,
            positions_value=positions_value,
            daily_pnl=daily_pnl,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            drawdown=drawdown,
            drawdown_pct=drawdown / total_equity * 100 if total_equity > 0 else 0,
            leverage=leverage,
            exposure=exposure
        )
        return self.metrics_collector.record_portfolio_metrics(metrics)
    
    def record_health_metrics(
        self,
        cpu_usage: float = 0.0,
        memory_usage: float = 0.0,
        api_latency_ms: float = 0.0,
        error_count: int = 0,
        warning_count: int = 0,
        active_strategies: int = 0,
        open_positions: int = 0,
        redis_connected: bool = True,
        database_connected: bool = True
    ) -> bool:
        """Record system health metrics."""
        metrics = HealthMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            redis_connected=redis_connected,
            database_connected=database_connected,
            api_latency_ms=api_latency_ms,
            error_count=error_count,
            warning_count=warning_count,
            active_strategies=active_strategies,
            open_positions=open_positions
        )
        return self.metrics_collector.record_health_metrics(metrics)
    
    # ==================== Edge Detection ====================
    
    def establish_strategy_baseline(
        self,
        strategy_name: str,
        lookback_trades: int = 100
    ) -> bool:
        """
        Establish performance baseline for a strategy.
        
        Args:
            strategy_name: Strategy name
            lookback_trades: Number of trades to use for baseline
            
        Returns:
            True if baseline was established
        """
        history = self.metrics_collector.get_performance_history(
            strategy_name, days=30
        )
        
        if len(history) < 20:
            logger.warning(
                f"Insufficient data to establish baseline for {strategy_name}: "
                f"{len(history)} trades (need at least 20)"
            )
            return False
        
        success = self.edge_detector.establish_baseline(
            strategy_name, history, min_trades=lookback_trades
        )
        
        if success:
            logger.info(f"Established baseline for {strategy_name}")
        
        return success
    
    def check_strategy_decay(
        self,
        strategy_name: str,
        session_id: str = "current"
    ) -> List[DecayAlert]:
        """
        Check for edge decay in a strategy.
        
        Args:
            strategy_name: Strategy name
            session_id: Session ID
            
        Returns:
            List of decay alerts
        """
        # Calculate current metrics
        metrics = self.metrics_collector.calculate_strategy_metrics(
            strategy_name, session_id
        )
        
        # Check for decay
        alerts = self.edge_detector.check_decay(
            strategy_name, metrics, session_id
        )
        
        # Handle alerts
        for alert in alerts:
            self._handle_decay_alert(alert)
        
        return alerts
    
    def _handle_decay_alert(self, alert: DecayAlert) -> None:
        """Handle a decay alert - take action based on severity."""
        # Log the alert
        logger.warning(
            f"Edge decay detected: {alert.strategy_name} - {alert.message}"
        )
        
        # Record in history
        self._alert_history.append({
            "timestamp": alert.timestamp,
            "strategy_name": alert.strategy_name,
            "decay_type": alert.decay_type.value,
            "severity": alert.severity,
            "message": alert.message
        })
        
        # Send Telegram notification
        if self.telegram:
            self.telegram.send_edge_decay_alert(
                strategy_name=alert.strategy_name,
                decay_type=alert.decay_type.value,
                current_value=alert.current_value,
                threshold_value=alert.threshold_value,
                drop_percentage=alert.drop_percentage,
                additional_info=alert.message
            )
        
        # Auto-actions for critical alerts
        if alert.severity == "critical" and self.config.auto_pause_on_critical:
            if self._pause_callback:
                try:
                    self._pause_callback(alert.strategy_name)
                    logger.info(f"Auto-paused strategy: {alert.strategy_name}")
                except Exception as e:
                    logger.error(f"Error pausing strategy: {e}")
        
        # Auto-reduce position size
        if self.config.auto_reduce_size_on_decay:
            if self._reduce_size_callback:
                try:
                    self._reduce_size_callback(
                        alert.strategy_name,
                        self.config.reduction_factor
                    )
                    logger.info(
                        f"Reduced position size for {alert.strategy_name} "
                        f"by factor {self.config.reduction_factor}"
                    )
                except Exception as e:
                    logger.error(f"Error reducing position size: {e}")
    
    def get_strategy_edge_status(self, strategy_name: str) -> EdgeStatus:
        """Get edge status for a strategy."""
        return self.edge_detector.get_edge_status(strategy_name)
    
    # ==================== Reporting ====================
    
    def generate_daily_report(self) -> str:
        """Generate daily summary report."""
        return self.reporter.generate_daily_summary(strategies=self._strategies)
    
    def generate_strategy_report(
        self,
        strategy_name: str,
        session_id: str = "default",
        period_days: int = 7
    ) -> str:
        """Generate strategy performance report."""
        return self.reporter.generate_strategy_report(
            strategy_name, session_id, period_days
        )
    
    def generate_status_report(self, include_details: bool = False) -> str:
        """Generate real-time status report."""
        return self.reporter.generate_status_report(
            strategies=self._strategies,
            include_details=include_details
        )
    
    def send_daily_report(self) -> bool:
        """Send daily report via Telegram."""
        return self.reporter.send_daily_report(strategies=self._strategies)
    
    def send_health_alert(
        self,
        title: str,
        message: str,
        level: str = "warning"
    ) -> bool:
        """Send health alert via Telegram."""
        if not self.telegram:
            return False
        
        try:
            return self.telegram.send_alert(title, message, level=getattr(AlertLevel, level.upper(), AlertLevel.WARNING))
        except Exception:
            return False
    
    # ==================== Background Monitoring ====================
    
    def start(self) -> None:
        """Start background monitoring."""
        if self._running:
            logger.warning("Monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="MonitorThread"
        )
        self._monitor_thread.start()
        logger.info("Monitor started")
    
    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("Monitor stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        logger.info("Monitoring loop started")
        
        while self._running:
            try:
                self._run_checks()
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep for configured interval
            time.sleep(self.config.check_interval_seconds)
        
        logger.info("Monitoring loop stopped")
    
    def _run_checks(self) -> None:
        """Run all monitoring checks."""
        now = datetime.utcnow()
        
        # Check edge decay at configured interval
        if (now - self._last_edge_check).total_seconds() >= self.config.edge_check_interval_seconds:
            self._run_edge_checks()
            self._last_edge_check = now
        
        # Check for daily report time
        if now.hour == self.config.daily_report_hour_utc and now.minute < 5:
            if (now - self._last_report_time).total_seconds() >= 3600:
                self.send_daily_report()
                self._last_report_time = now
        
        # Record health metrics (simple check)
        self._record_system_health()
    
    def _run_edge_checks(self) -> None:
        """Run edge decay checks for all strategies."""
        for strategy_name in self._strategies:
            try:
                # Check if we have a baseline
                if self.edge_detector.get_baseline(strategy_name, DecayType.WIN_RATE_DROP) is None:
                    # Try to establish baseline
                    self.establish_strategy_baseline(strategy_name)
                
                # Check decay
                self.check_strategy_decay(strategy_name)
                
            except Exception as e:
                logger.error(f"Error checking decay for {strategy_name}: {e}")
    
    def _record_system_health(self) -> None:
        """Record basic system health."""
        try:
            import psutil
            
            self.record_health_metrics(
                cpu_usage=psutil.cpu_percent(),
                memory_usage=psutil.virtual_memory().percent,
                active_strategies=len(self._strategies),
                redis_connected=self.metrics_collector.redis.ping() if self.metrics_collector.redis else False
            )
        except ImportError:
            # psutil not available, skip detailed health
            self.record_health_metrics()
        except Exception as e:
            logger.warning(f"Error recording system health: {e}")
    
    # ==================== Status Information ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall monitoring status."""
        edge_statuses = []
        for strategy in self._strategies:
            edge_statuses.append({
                "strategy_name": strategy,
                "status": self.edge_detector.get_edge_status(strategy)
            })
        
        return {
            "running": self._running,
            "strategies_count": len(self._strategies),
            "last_edge_check": self._last_edge_check.isoformat() if self._last_edge_check != datetime.min else None,
            "edge_statuses": edge_statuses,
            "alert_history_count": len(self._alert_history)
        }
    
    def get_alert_history(
        self,
        strategy_name: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get alert history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        filtered = [
            a for a in self._alert_history
            if a["timestamp"] >= cutoff
        ]
        
        if strategy_name:
            filtered = [
                a for a in filtered
                if a["strategy_name"] == strategy_name
            ]
        
        return filtered
    
    # ==================== Context Manager ====================
    
    def __enter__(self) -> "Monitor":
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


class RegimeMonitor:
    """
    Monitors market regimes across symbols.
    
    Detects regime changes and provides regime-aware trading recommendations.
    """
    
    REGIME_EMOJI = {
        RegimeType.TRENDING_UP: "📈",
        RegimeType.TRENDING_DOWN: "📉",
        RegimeType.RANGING: "↔️",
        RegimeType.HIGH_VOL: "🌊",
        RegimeType.LOW_VOL: "🌅"
    }
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        telegram_notifier: Optional[TelegramNotifier] = None
    ):
        self.metrics_collector = metrics_collector
        self.telegram = telegram_notifier
        self._regime_cache: Dict[str, RegimeType] = {}
        self._lock = threading.RLock()
    
    def detect_and_update_regime(
        self,
        symbol: str
    ) -> Tuple[RegimeType, float]:
        """
        Detect and update regime for a symbol.
        
        Returns:
            Tuple of (RegimeType, confidence)
        """
        if not self.metrics_collector:
            return RegimeType.RANGING, 0.0
        
        regime, confidence = self.metrics_collector.calculate_regime_confidence(symbol)
        
        with self._lock:
            old_regime = self._regime_cache.get(symbol)
            self._regime_cache[symbol] = regime
            
            # Send notification on regime change
            if old_regime is not None and old_regime != regime:
                self._on_regime_change(symbol, old_regime, regime, confidence)
        
        return regime, confidence
    
    def _on_regime_change(
        self,
        symbol: str,
        old_regime: RegimeType,
        new_regime: RegimeType,
        confidence: float
    ) -> None:
        """Handle regime change."""
        recommendations = self._get_regime_recommendations(new_regime)
        
        if self.telegram:
            self.telegram.send_regime_change_alert(
                symbol=symbol,
                old_regime=old_regime.value,
                new_regime=new_regime.value,
                confidence=confidence,
                recommendations=recommendations
            )
    
    def _get_regime_recommendations(self, regime: RegimeType) -> List[str]:
        """Get trading recommendations for a regime."""
        recommendations = {
            RegimeType.TRENDING_UP: [
                "Consider momentum strategies",
                "Use trailing stops to lock profits",
                "Avoid mean reversion approaches"
            ],
            RegimeType.TRENDING_DOWN: [
                "Focus on short positions",
                "Use tighter stop losses",
                "Consider defensive assets"
            ],
            RegimeType.RANGING: [
                "Mean reversion strategies may work",
                "Range-bound position sizing",
                "Avoid trend-following approaches"
            ],
            RegimeType.HIGH_VOL: [
                "Reduce position sizes",
                "Widen stop losses",
                "Consider volatility-targeting"
            ],
            RegimeType.LOW_VOL: [
                "Reduce position sizes",
                "Expect breakout soon",
                "Consider straddle strategies"
            ]
        }
        return recommendations.get(regime, [])
    
    def get_current_regimes(self) -> Dict[str, RegimeType]:
        """Get current regimes for all tracked symbols."""
        with self._lock:
            return dict(self._regime_cache)
    
    def get_regime_overview(self) -> Dict[str, int]:
        """Get count of symbols in each regime."""
        with self._lock:
            overview: Dict[str, int] = {}
            for regime in self._regime_cache.values():
                regime_name = regime.value
                overview[regime_name] = overview.get(regime_name, 0) + 1
            return overview
