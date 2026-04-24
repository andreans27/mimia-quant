"""
Edge Decay Detector for Mimia Quant Trading System.

Monitors trading strategy performance and detects when strategies are
losing their "edge" - becoming unprofitable or underperforming.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import threading

from .metrics_collector import MetricsCollector, StrategyMetrics


class DecayType(Enum):
    """Types of edge decay."""
    WIN_RATE_DROP = "win_rate"
    PROFIT_FACTOR_DROP = "profit_factor"
    SHARPE_DROP = "sharpe"
    EXPECTANCY_DROP = "expectancy"
    VOLATILITY_INCREASE = "volatility"
    DRAWDOWN_INCREASE = "drawdown"


@dataclass
class DecayThreshold:
    """Configuration for decay detection thresholds."""
    decay_type: DecayType
    threshold_value: float
    drop_percentage: float  # Percentage drop to trigger alert
    lookback_trades: int = 100
    confirmation_trades: int = 3  # Number of consecutive confirming trades


@dataclass
class DecayAlert:
    """Alert generated when edge decay is detected."""
    timestamp: datetime
    strategy_name: str
    decay_type: DecayType
    current_value: float
    baseline_value: float
    threshold_value: float
    drop_percentage: float
    severity: str  # "warning", "critical"
    message: str
    trades_analyzed: int


@dataclass
class EdgeStatus:
    """Current edge status of a strategy."""
    strategy_name: str
    is_healthy: bool
    health_score: float  # 0-100
    active_alerts: List[DecayAlert]
    last_update: datetime
    recommendations: List[str]


class EdgeDecayDetector:
    """
    Detects when trading strategies are losing their edge.
    
    Monitors key performance metrics and triggers alerts when
    thresholds are crossed. Uses sliding windows and trend
    analysis to identify genuine decay vs normal variance.
    
    Thresholds:
    - Win rate drop > 10%
    - Profit Factor < 1.5
    - Sharpe Ratio < 1.0
    """
    
    # Default thresholds as specified
    DEFAULT_THRESHOLDS = {
        DecayType.WIN_RATE_DROP: DecayThreshold(
            decay_type=DecayType.WIN_RATE_DROP,
            threshold_value=0.0,  # Will be calculated as baseline - 10%
            drop_percentage=10.0,
            lookback_trades=100,
            confirmation_trades=3
        ),
        DecayType.PROFIT_FACTOR_DROP: DecayThreshold(
            decay_type=DecayType.PROFIT_FACTOR_DROP,
            threshold_value=1.5,
            drop_percentage=0.0,  # Direct threshold comparison
            lookback_trades=100,
            confirmation_trades=3
        ),
        DecayType.SHARPE_DROP: DecayThreshold(
            decay_type=DecayType.SHARPE_DROP,
            threshold_value=1.0,
            drop_percentage=0.0,  # Direct threshold comparison
            lookback_trades=100,
            confirmation_trades=3
        ),
    }
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        check_interval_seconds: int = 300  # 5 minutes
    ):
        """
        Initialize edge decay detector.
        
        Args:
            metrics_collector: MetricsCollector instance
            check_interval_seconds: Interval between decay checks
        """
        self.metrics_collector = metrics_collector
        self.check_interval = check_interval_seconds
        
        # Strategy baselines (learned over time)
        self._baselines: Dict[str, Dict[DecayType, float]] = {}
        
        # Active alerts
        self._active_alerts: Dict[str, List[DecayAlert]] = {}
        
        # Alert history
        self._alert_history: List[DecayAlert] = []
        
        # Custom thresholds
        self._thresholds: Dict[str, Dict[DecayType, DecayThreshold]] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Callback for alerts
        self._alert_callback: Optional[callable] = None
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set the metrics collector."""
        self.metrics_collector = collector
    
    def set_alert_callback(self, callback: callable) -> None:
        """Set callback function for decay alerts."""
        self._alert_callback = callback
    
    def set_threshold(
        self,
        strategy_name: str,
        decay_type: DecayType,
        threshold: DecayThreshold
    ) -> None:
        """Set custom threshold for a strategy."""
        with self._lock:
            if strategy_name not in self._thresholds:
                self._thresholds[strategy_name] = {}
            self._thresholds[strategy_name][decay_type] = threshold
    
    def get_threshold(
        self,
        strategy_name: str,
        decay_type: DecayType
    ) -> DecayThreshold:
        """Get threshold for a strategy, using defaults if not set."""
        if strategy_name in self._thresholds:
            if decay_type in self._thresholds[strategy_name]:
                return self._thresholds[strategy_name][decay_type]
        return self.DEFAULT_THRESHOLDS.get(decay_type)
    
    def establish_baseline(
        self,
        strategy_name: str,
        metrics_history: List[StrategyMetrics],
        min_trades: int = 50
    ) -> bool:
        """
        Establish performance baseline for a strategy.
        
        Args:
            strategy_name: Strategy name
            metrics_history: Historical metrics
            min_trades: Minimum trades required to establish baseline
            
        Returns:
            True if baseline was established
        """
        if len(metrics_history) < min_trades:
            return False
        
        with self._lock:
            # Calculate baseline values from recent history
            recent = metrics_history[-min_trades:]
            
            baseline = {
                DecayType.WIN_RATE_DROP: statistics.mean([m.win_rate for m in recent]),
                DecayType.PROFIT_FACTOR_DROP: statistics.mean([m.profit_factor for m in recent]),
                DecayType.SHARPE_DROP: statistics.mean([m.sharpe_ratio for m in recent]),
                DecayType.EXPECTANCY_DROP: statistics.mean([m.expectancy for m in recent]),
                DecayType.VOLATILITY_INCREASE: statistics.mean([m.volatility for m in recent]),
                DecayType.DRAWDOWN_INCREASE: statistics.mean([m.max_drawdown_pct for m in recent]),
            }
            
            self._baselines[strategy_name] = baseline
            return True
    
    def get_baseline(
        self,
        strategy_name: str,
        decay_type: DecayType
    ) -> Optional[float]:
        """Get baseline value for a strategy and decay type."""
        if strategy_name in self._baselines:
            return self._baselines[strategy_name].get(decay_type)
        return None
    
    def check_decay(
        self,
        strategy_name: str,
        current_metrics: StrategyMetrics,
        session_id: str
    ) -> List[DecayAlert]:
        """
        Check for edge decay in a strategy.
        
        Args:
            strategy_name: Strategy name
            current_metrics: Current strategy metrics
            session_id: Trading session ID
            
        Returns:
            List of DecayAlert objects
        """
        alerts = []
        
        with self._lock:
            baseline = self._baselines.get(strategy_name, {})
            
            # Check win rate drop
            wr_alert = self._check_win_rate_decay(
                strategy_name, current_metrics, baseline.get(DecayType.WIN_RATE_DROP)
            )
            if wr_alert:
                alerts.append(wr_alert)
            
            # Check profit factor
            pf_alert = self._check_profit_factor_decay(
                strategy_name, current_metrics, baseline.get(DecayType.PROFIT_FACTOR_DROP)
            )
            if pf_alert:
                alerts.append(pf_alert)
            
            # Check Sharpe ratio
            sh_alert = self._check_sharpe_decay(
                strategy_name, current_metrics, baseline.get(DecayType.SHARPE_DROP)
            )
            if sh_alert:
                alerts.append(sh_alert)
            
            # Check expectancy
            exp_alert = self._check_expectancy_decay(
                strategy_name, current_metrics, baseline.get(DecayType.EXPECTANCY_DROP)
            )
            if exp_alert:
                alerts.append(exp_alert)
            
            # Update active alerts
            if alerts:
                self._active_alerts[strategy_name] = alerts
                self._alert_history.extend(alerts)
                
                # Trigger callback
                if self._alert_callback:
                    for alert in alerts:
                        try:
                            self._alert_callback(alert)
                        except Exception as e:
                            print(f"Alert callback error: {e}")
        
        return alerts
    
    def _check_win_rate_decay(
        self,
        strategy_name: str,
        metrics: StrategyMetrics,
        baseline: Optional[float]
    ) -> Optional[DecayAlert]:
        """Check for win rate decay."""
        threshold = self.get_threshold(strategy_name, DecayType.WIN_RATE_DROP)
        
        if baseline is None:
            baseline = threshold.threshold_value
        
        # Calculate threshold value (baseline - drop%)
        threshold_value = baseline * (1 - threshold.drop_percentage / 100)
        
        if metrics.win_rate < threshold_value:
            drop_pct = ((baseline - metrics.win_rate) / baseline * 100) if baseline > 0 else 100
            
            return DecayAlert(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                decay_type=DecayType.WIN_RATE_DROP,
                current_value=metrics.win_rate,
                baseline_value=baseline,
                threshold_value=threshold_value,
                drop_percentage=drop_pct,
                severity="critical" if drop_pct > 20 else "warning",
                message=f"Win rate dropped from {baseline:.1%} to {metrics.win_rate:.1%} ({drop_pct:.1f}% drop)",
                trades_analyzed=metrics.total_trades
            )
        
        return None
    
    def _check_profit_factor_decay(
        self,
        strategy_name: str,
        metrics: StrategyMetrics,
        baseline: Optional[float]
    ) -> Optional[DecayAlert]:
        """Check for profit factor decay."""
        threshold = self.get_threshold(strategy_name, DecayType.PROFIT_FACTOR_DROP)
        
        if metrics.profit_factor < threshold.threshold_value:
            drop_pct = ((threshold.threshold_value - metrics.profit_factor) / threshold.threshold_value * 100) if threshold.threshold_value > 0 else 100
            
            return DecayAlert(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                decay_type=DecayType.PROFIT_FACTOR_DROP,
                current_value=metrics.profit_factor,
                baseline_value=baseline or 0.0,
                threshold_value=threshold.threshold_value,
                drop_percentage=drop_pct,
                severity="critical" if metrics.profit_factor < 1.0 else "warning",
                message=f"Profit Factor dropped to {metrics.profit_factor:.2f} (threshold: {threshold.threshold_value})",
                trades_analyzed=metrics.total_trades
            )
        
        return None
    
    def _check_sharpe_decay(
        self,
        strategy_name: str,
        metrics: StrategyMetrics,
        baseline: Optional[float]
    ) -> Optional[DecayAlert]:
        """Check for Sharpe ratio decay."""
        threshold = self.get_threshold(strategy_name, DecayType.SHARPE_DROP)
        
        if metrics.sharpe_ratio < threshold.threshold_value:
            return DecayAlert(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                decay_type=DecayType.SHARPE_DROP,
                current_value=metrics.sharpe_ratio,
                baseline_value=baseline or 0.0,
                threshold_value=threshold.threshold_value,
                drop_percentage=0.0,
                severity="warning",
                message=f"Sharpe Ratio dropped to {metrics.sharpe_ratio:.2f} (threshold: {threshold.threshold_value})",
                trades_analyzed=metrics.total_trades
            )
        
        return None
    
    def _check_expectancy_decay(
        self,
        strategy_name: str,
        metrics: StrategyMetrics,
        baseline: Optional[float]
    ) -> Optional[DecayAlert]:
        """Check for expectancy decay."""
        if baseline is not None and metrics.expectancy < 0:
            drop_pct = ((baseline - metrics.expectancy) / abs(baseline) * 100) if baseline != 0 else 100
            
            return DecayAlert(
                timestamp=datetime.utcnow(),
                strategy_name=strategy_name,
                decay_type=DecayType.EXPECTANCY_DROP,
                current_value=metrics.expectancy,
                baseline_value=baseline,
                threshold_value=0.0,
                drop_percentage=drop_pct,
                severity="critical",
                message=f"Expectancy turned negative: {metrics.expectancy:.4f} (baseline: {baseline:.4f})",
                trades_analyzed=metrics.total_trades
            )
        
        return None
    
    def get_edge_status(self, strategy_name: str) -> EdgeStatus:
        """
        Get current edge status for a strategy.
        
        Returns:
            EdgeStatus object with health information
        """
        with self._lock:
            alerts = self._active_alerts.get(strategy_name, [])
            
            # Calculate health score
            health_score = 100.0
            if alerts:
                for alert in alerts:
                    if alert.severity == "critical":
                        health_score -= 30
                    else:
                        health_score -= 10
            
            health_score = max(0.0, health_score)
            
            is_healthy = health_score >= 70 and not any(
                a.severity == "critical" for a in alerts
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(strategy_name, alerts)
            
            return EdgeStatus(
                strategy_name=strategy_name,
                is_healthy=is_healthy,
                health_score=health_score,
                active_alerts=alerts,
                last_update=datetime.utcnow(),
                recommendations=recommendations
            )
    
    def _generate_recommendations(
        self,
        strategy_name: str,
        alerts: List[DecayAlert]
    ) -> List[str]:
        """Generate recommendations based on active alerts."""
        recommendations = []
        
        alert_types = {a.decay_type for a in alerts}
        
        if DecayType.WIN_RATE_DROP in alert_types:
            recommendations.append("Review entry criteria - win rate declining")
            recommendations.append("Consider tightening stop-loss placement")
        
        if DecayType.PROFIT_FACTOR_DROP in alert_types:
            recommendations.append("Average winners may be shrinking - review position sizing")
            recommendations.append("Check if market regime has changed")
        
        if DecayType.SHARPE_DROP in alert_types:
            recommendations.append("Risk-adjusted returns deteriorating - consider reducing position size")
            recommendations.append("Review correlation with other strategies")
        
        if DecayType.EXPECTANCY_DROP in alert_types:
            recommendations.append("CRITICAL: Strategy may need to be paused")
            recommendations.append("Review recent market conditions and slippage")
        
        if not recommendations:
            recommendations.append("Continue monitoring - no action required")
        
        return recommendations
    
    def get_all_edge_statuses(self) -> List[EdgeStatus]:
        """Get edge status for all tracked strategies."""
        with self._lock:
            statuses = []
            for strategy_name in self._baselines.keys():
                statuses.append(self.get_edge_status(strategy_name))
            return statuses
    
    def get_alert_history(
        self,
        strategy_name: Optional[str] = None,
        hours: int = 24
    ) -> List[DecayAlert]:
        """Get alert history."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        with self._lock:
            filtered = [
                a for a in self._alert_history
                if a.timestamp >= cutoff
            ]
            
            if strategy_name:
                filtered = [
                    a for a in filtered
                    if a.strategy_name == strategy_name
                ]
            
            return filtered
    
    def clear_alert(
        self,
        strategy_name: str,
        decay_type: DecayType
    ) -> bool:
        """Clear a specific active alert."""
        with self._lock:
            if strategy_name in self._active_alerts:
                before = len(self._active_alerts[strategy_name])
                self._active_alerts[strategy_name] = [
                    a for a in self._active_alerts[strategy_name]
                    if a.decay_type != decay_type
                ]
                return len(self._active_alerts[strategy_name]) < before
        return False
    
    def clear_all_alerts(self, strategy_name: str) -> None:
        """Clear all alerts for a strategy."""
        with self._lock:
            if strategy_name in self._active_alerts:
                del self._active_alerts[strategy_name]
    
    def analyze_trend(
        self,
        strategy_name: str,
        metrics_history: List[StrategyMetrics],
        decay_type: DecayType
    ) -> Dict[str, Any]:
        """
        Analyze the trend of a specific metric.
        
        Args:
            strategy_name: Strategy name
            metrics_history: Historical metrics
            decay_type: Type of decay to analyze
            
        Returns:
            Dict with trend analysis
        """
        if not metrics_history:
            return {"error": "No data available"}
        
        # Extract relevant values
        if decay_type == DecayType.WIN_RATE_DROP:
            values = [m.win_rate for m in metrics_history]
        elif decay_type == DecayType.PROFIT_FACTOR_DROP:
            values = [m.profit_factor for m in metrics_history]
        elif decay_type == DecayType.SHARPE_DROP:
            values = [m.sharpe_ratio for m in metrics_history]
        elif decay_type == DecayType.EXPECTANCY_DROP:
            values = [m.expectancy for m in metrics_history]
        elif decay_type == DecayType.VOLATILITY_INCREASE:
            values = [m.volatility for m in metrics_history]
        elif decay_type == DecayType.DRAWDOWN_INCREASE:
            values = [m.max_drawdown_pct for m in metrics_history]
        else:
            return {"error": "Unknown decay type"}
        
        if len(values) < 2:
            return {"error": "Insufficient data"}
        
        # Calculate trend (simple linear regression slope)
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Calculate if trend is concerning
        recent_avg = statistics.mean(values[-10:]) if len(values) >= 10 else statistics.mean(values)
        older_avg = statistics.mean(values[:10]) if len(values) >= 10 else statistics.mean(values[:len(values)//2])
        
        trend_direction = "declining" if slope < 0 else "improving"
        
        return {
            "decay_type": decay_type.value,
            "current_value": values[-1],
            "recent_average": recent_avg,
            "older_average": older_avg,
            "trend_slope": slope,
            "trend_direction": trend_direction,
            "is_concerning": slope < 0 and recent_avg < older_avg * 0.9,  # 10% drop
            "data_points": n
        }
