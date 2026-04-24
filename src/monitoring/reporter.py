"""
Reporter for Mimia Quant Trading System.

Generates periodic and on-demand reports for trading performance,
system health, and strategy analysis.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import threading

from .metrics_collector import (
    MetricsCollector,
    StrategyMetrics,
    PortfolioMetrics,
    HealthMetrics,
    RegimeType
)
from .edge_decay_detector import EdgeDecayDetector, EdgeStatus, DecayAlert
from .telegram_notifier import TelegramNotifier


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    include_trades: bool = True
    include_metrics: bool = True
    include_equity_curve: bool = True
    include_regime: bool = True
    include_edge_status: bool = True
    include_health: bool = True
    format_type: str = "text"  # text, json, html
    max_trades_shown: int = 10


class Reporter:
    """
    Generates reports for the trading system.
    
    Supports:
    - Daily summary reports
    - Strategy performance reports
    - Real-time status reports
    - Edge decay reports
    - Export to multiple formats
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        edge_detector: Optional[EdgeDecayDetector] = None,
        telegram_notifier: Optional[TelegramNotifier] = None
    ):
        """
        Initialize reporter.
        
        Args:
            metrics_collector: MetricsCollector instance
            edge_detector: EdgeDecayDetector instance
            telegram_notifier: TelegramNotifier instance
        """
        self.metrics_collector = metrics_collector
        self.edge_detector = edge_detector
        self.telegram = telegram_notifier
        self._lock = threading.RLock()
    
    def set_metrics_collector(self, collector: MetricsCollector) -> None:
        """Set the metrics collector."""
        self.metrics_collector = collector
    
    def set_edge_detector(self, detector: EdgeDecayDetector) -> None:
        """Set the edge detector."""
        self.edge_detector = detector
    
    def set_telegram_notifier(self, notifier: TelegramNotifier) -> None:
        """Set the Telegram notifier."""
        self.telegram = notifier
    
    # ==================== Daily Summary Report ====================
    
    def generate_daily_summary(
        self,
        date: Optional[datetime] = None,
        strategies: Optional[List[str]] = None
    ) -> str:
        """
        Generate daily summary report.
        
        Args:
            date: Date for the report (default: today)
            strategies: List of strategies to include
            
        Returns:
            Formatted report string
        """
        if date is None:
            date = datetime.utcnow()
        
        report_lines = [
            "=" * 50,
            f"DAILY SUMMARY REPORT",
            f"Date: {date.strftime('%Y-%m-%d')}",
            "=" * 50,
            ""
        ]
        
        # Get equity data
        equity_data = []
        if self.metrics_collector:
            equity_data = self.metrics_collector.get_equity_curve(days=1)
        
        if equity_data:
            first = equity_data[0]
            last = equity_data[-1]
            
            daily_pnl = last.get("daily_pnl", 0)
            daily_return = last.get("daily_return", 0) * 100
            total_value = last.get("total_value", 0)
            drawdown = last.get("drawdown", 0)
            
            pnl_emoji = "💰" if daily_pnl >= 0 else "📉"
            
            report_lines.extend([
                f"{pnl_emoji} Daily P&L: ${daily_pnl:,.2f} ({daily_return:+.2f}%)",
                f"💵 Total Value: ${total_value:,.2f}",
                f"📉 Current Drawdown: {drawdown:.2f}%",
                ""
            ])
        
        # Strategy summaries
        if self.metrics_collector and strategies:
            for strategy_name in strategies:
                summary = self._generate_strategy_summary(strategy_name, days=1)
                if summary:
                    report_lines.append(summary)
                    report_lines.append("")
        
        # Edge status overview
        if self.edge_detector:
            edge_report = self._generate_edge_status_report()
            if edge_report:
                report_lines.append(edge_report)
                report_lines.append("")
        
        # Health status
        if self.metrics_collector:
            health = self.metrics_collector.get_health_metrics()
            if health:
                health_line = f"🏥 System Health: {'✅ Healthy' if health.error_count == 0 else '⚠️ Issues'}"
                report_lines.append(health_line)
        
        report_lines.append("")
        report_lines.append(f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return "\n".join(report_lines)
    
    def _generate_strategy_summary(
        self,
        strategy_name: str,
        days: int = 1
    ) -> Optional[str]:
        """Generate summary for a single strategy."""
        if not self.metrics_collector:
            return None
        
        history = self.metrics_collector.get_performance_history(strategy_name, days=days)
        if not history:
            return None
        
        latest = history[-1]
        
        lines = [
            f"📊 Strategy: {strategy_name}",
            f"   Trades: {latest.total_trades}",
            f"   Win Rate: {latest.win_rate:.1%}",
            f"   Profit Factor: {latest.profit_factor:.2f}",
            f"   P&L: ${latest.total_pnl:,.2f}",
            f"   Sharpe: {latest.sharpe_ratio:.2f}",
            f"   Max DD: {latest.max_drawdown_pct:.1f}%"
        ]
        
        return "\n".join(lines)
    
    def _generate_edge_status_report(self) -> str:
        """Generate edge status overview."""
        statuses = self.edge_detector.get_all_edge_statuses()
        
        if not statuses:
            return ""
        
        lines = [
            "🔍 Edge Status Overview:",
        ]
        
        healthy_count = sum(1 for s in statuses if s.is_healthy)
        unhealthy_count = len(statuses) - healthy_count
        
        for status in statuses:
            emoji = "✅" if status.is_healthy else "⚠️"
            health_bar = "█" * int(status.health_score / 10) + "░" * (10 - int(status.health_score / 10))
            lines.append(f"   {emoji} {status.strategy_name}: [{health_bar}] {status.health_score:.0f}%")
        
        lines.append(f"   Total: {healthy_count} healthy, {unhealthy_count} need attention")
        
        return "\n".join(lines)
    
    # ==================== Strategy Performance Report ====================
    
    def generate_strategy_report(
        self,
        strategy_name: str,
        session_id: str,
        period_days: int = 7,
        config: Optional[ReportConfig] = None
    ) -> str:
        """
        Generate detailed strategy performance report.
        
        Args:
            strategy_name: Strategy name
            session_id: Session ID
            period_days: Number of days to analyze
            config: Report configuration
            
        Returns:
            Formatted report string
        """
        if config is None:
            config = ReportConfig()
        
        report_lines = [
            "=" * 50,
            f"STRATEGY PERFORMANCE REPORT",
            "=" * 50,
            f"Strategy: {strategy_name}",
            f"Session: {session_id}",
            f"Period: Last {period_days} days",
            ""
        ]
        
        # Get metrics history
        metrics_history = []
        if self.metrics_collector:
            metrics_history = self.metrics_collector.get_performance_history(
                strategy_name, days=period_days
            )
        
        if metrics_history:
            # Aggregate metrics
            latest = metrics_history[-1]
            total_trades = sum(m.total_trades for m in metrics_history)
            total_pnl = sum(m.total_pnl for m in metrics_history)
            avg_win_rate = sum(m.win_rate for m in metrics_history) / len(metrics_history)
            avg_pf = sum(m.profit_factor for m in metrics_history) / len(metrics_history)
            avg_sharpe = sum(m.sharpe_ratio for m in metrics_history) / len(metrics_history)
            
            report_lines.extend([
                "📈 Performance Summary:",
                f"   Total Trades: {total_trades}",
                f"   Total P&L: ${total_pnl:,.2f}",
                f"   Avg Win Rate: {avg_win_rate:.1%}",
                f"   Avg Profit Factor: {avg_pf:.2f}",
                f"   Avg Sharpe Ratio: {avg_sharpe:.2f}",
                ""
            ])
            
            # Current metrics
            report_lines.extend([
                "📊 Current Metrics:",
                f"   Win Rate: {latest.win_rate:.1%}",
                f"   Profit Factor: {latest.profit_factor:.2f}",
                f"   Sharpe Ratio: {latest.sharpe_ratio:.2f}",
                f"   Sortino Ratio: {latest.sortino_ratio:.2f}",
                f"   Max Drawdown: {latest.max_drawdown_pct:.1f}%",
                f"   Expectancy: ${latest.expectancy:.4f}",
                ""
            ])
            
            # Trade breakdown
            winning = latest.winning_trades
            losing = latest.losing_trades
            report_lines.extend([
                "📋 Trade Breakdown:",
                f"   Winning Trades: {winning}",
                f"   Losing Trades: {losing}",
                f"   Avg Win: ${latest.avg_win:,.2f}",
                f"   Avg Loss: ${latest.avg_loss:,.2f}",
                ""
            ])
        
        # Edge status
        if self.edge_detector:
            status = self.edge_detector.get_edge_status(strategy_name)
            health_bar = "█" * int(status.health_score / 10) + "░" * (10 - int(status.health_score / 10))
            
            report_lines.extend([
                "🔍 Edge Status:",
                f"   Health Score: [{health_bar}] {status.health_score:.0f}%",
                f"   Status: {'✅ Healthy' if status.is_healthy else '⚠️ Needs Attention'}",
                ""
            ])
            
            if status.active_alerts:
                report_lines.append("   Active Alerts:")
                for alert in status.active_alerts:
                    report_lines.append(f"   • {alert.message}")
                report_lines.append("")
            
            if status.recommendations:
                report_lines.append("   Recommendations:")
                for rec in status.recommendations:
                    report_lines.append(f"   • {rec}")
                report_lines.append("")
        
        # Equity curve summary
        if config.include_equity_curve and self.metrics_collector:
            equity = self.metrics_collector.get_equity_curve(
                strategy_name=strategy_name,
                session_id=session_id,
                days=period_days
            )
            if equity:
                returns = [e.get("daily_return", 0) for e in equity if e.get("daily_return")]
                if returns:
                    best_day = max(returns) * 100
                    worst_day = min(returns) * 100
                    report_lines.extend([
                        "📈 Equity Summary:",
                        f"   Best Day: {best_day:+.2f}%",
                        f"   Worst Day: {worst_day:+.2f}%",
                        ""
                    ])
        
        report_lines.append(f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return "\n".join(report_lines)
    
    # ==================== Real-time Status Report ====================
    
    def generate_status_report(
        self,
        strategies: List[str],
        include_details: bool = False
    ) -> str:
        """
        Generate real-time system status report.
        
        Args:
            strategies: List of strategy names
            include_details: Include detailed metrics
            
        Returns:
            Formatted status report
        """
        report_lines = [
            "=" * 40,
            "SYSTEM STATUS REPORT",
            "=" * 40,
            f"Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            ""
        ]
        
        # Portfolio summary
        if self.metrics_collector:
            portfolio = self.metrics_collector.get_portfolio_metrics()
            if portfolio:
                report_lines.extend([
                    "💼 Portfolio:",
                    f"   Total Equity: ${portfolio.total_equity:,.2f}",
                    f"   Cash: ${portfolio.cash:,.2f}",
                    f"   Positions: ${portfolio.positions_value:,.2f}",
                    f"   Daily P&L: ${portfolio.daily_pnl:,.2f}",
                    f"   Daily Return: {portfolio.daily_return:+.2f}%",
                    f"   Drawdown: {portfolio.drawdown:.2f}%",
                    ""
                ])
        
        # Strategy statuses
        report_lines.append("📊 Strategy Status:")
        
        for strategy_name in strategies:
            metrics = None
            if self.metrics_collector:
                metrics = self.metrics_collector.calculate_strategy_metrics(
                    strategy_name, session_id="current"
                )
            
            edge_status = None
            if self.edge_detector:
                edge_status = self.edge_detector.get_edge_status(strategy_name)
            
            status_emoji = "✅"
            if edge_status:
                if not edge_status.is_healthy:
                    status_emoji = "⚠️"
                if any(a.severity == "critical" for a in edge_status.active_alerts):
                    status_emoji = "🚨"
            
            if include_details and metrics:
                pnl_str = f"${metrics.total_pnl:,.2f}"
                wr_str = f"{metrics.win_rate:.1%}"
                pf_str = f"{metrics.profit_factor:.2f}"
                report_lines.append(
                    f"   {status_emoji} {strategy_name}: "
                    f"P&L={pnl_str}, WR={wr_str}, PF={pf_str}"
                )
            else:
                report_lines.append(f"   {status_emoji} {strategy_name}")
        
        report_lines.append("")
        
        # System health
        if self.metrics_collector:
            health = self.metrics_collector.get_health_metrics()
            if health:
                redis_status = "✅" if health.redis_connected else "❌"
                db_status = "✅" if health.database_connected else "❌"
                
                report_lines.extend([
                    "🏥 System Health:",
                    f"   Redis: {redis_status}",
                    f"   Database: {db_status}",
                    f"   API Latency: {health.api_latency_ms:.0f}ms",
                    f"   Errors: {health.error_count}",
                    f"   Warnings: {health.warning_count}",
                    ""
                ])
        
        return "\n".join(report_lines)
    
    # ==================== Edge Decay Report ====================
    
    def generate_edge_decay_report(
        self,
        strategy_name: Optional[str] = None,
        hours: int = 24
    ) -> str:
        """
        Generate edge decay report.
        
        Args:
            strategy_name: Optional specific strategy
            hours: Time window in hours
            
        Returns:
            Formatted edge decay report
        """
        if not self.edge_detector:
            return "Edge detector not configured"
        
        report_lines = [
            "=" * 40,
            "EDGE DECAY REPORT",
            "=" * 40,
            f"Period: Last {hours} hours",
            ""
        ]
        
        # Get alert history
        alerts = self.edge_detector.get_alert_history(
            strategy_name=strategy_name,
            hours=hours
        )
        
        if not alerts:
            report_lines.append("✅ No edge decay alerts in this period")
        else:
            # Group by strategy
            by_strategy: Dict[str, List[DecayAlert]] = {}
            for alert in alerts:
                if alert.strategy_name not in by_strategy:
                    by_strategy[alert.strategy_name] = []
                by_strategy[alert.strategy_name].append(alert)
            
            for strat, strat_alerts in by_strategy.items():
                report_lines.append(f"\n📊 {strat}:")
                
                for alert in strat_alerts:
                    severity_emoji = "🚨" if alert.severity == "critical" else "⚠️"
                    report_lines.append(
                        f"   {severity_emoji} [{alert.decay_type.value}] "
                        f"{alert.message}"
                    )
        
        # Edge statuses
        report_lines.append("\n📈 Current Edge Status:")
        
        if strategy_name:
            statuses = [self.edge_detector.get_edge_status(strategy_name)]
        else:
            statuses = self.edge_detector.get_all_edge_statuses()
        
        for status in statuses:
            health_bar = "█" * int(status.health_score / 10) + "░" * (10 - int(status.health_score / 10))
            status_emoji = "✅" if status.is_healthy else "⚠️"
            
            report_lines.append(
                f"   {status_emoji} {status.strategy_name}: "
                f"[{health_bar}] {status.health_score:.0f}%"
            )
        
        report_lines.append("")
        report_lines.append(f"Report generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        return "\n".join(report_lines)
    
    # ==================== Export Functions ====================
    
    def export_to_json(
        self,
        strategy_name: str,
        period_days: int = 7
    ) -> Dict[str, Any]:
        """
        Export strategy data to JSON format.
        
        Args:
            strategy_name: Strategy name
            period_days: Days to include
            
        Returns:
            Dictionary suitable for JSON export
        """
        result = {
            "strategy_name": strategy_name,
            "exported_at": datetime.utcnow().isoformat(),
            "period_days": period_days,
            "metrics_history": [],
            "equity_curve": [],
            "edge_status": None
        }
        
        if self.metrics_collector:
            # Metrics history
            history = self.metrics_collector.get_performance_history(
                strategy_name, days=period_days
            )
            result["metrics_history"] = [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_trades": m.total_trades,
                    "win_rate": m.win_rate,
                    "profit_factor": m.profit_factor,
                    "sharpe_ratio": m.sharpe_ratio,
                    "total_pnl": m.total_pnl,
                    "max_drawdown_pct": m.max_drawdown_pct
                }
                for m in history
            ]
            
            # Equity curve
            equity = self.metrics_collector.get_equity_curve(
                strategy_name=strategy_name,
                days=period_days
            )
            result["equity_curve"] = equity
        
        if self.edge_detector:
            status = self.edge_detector.get_edge_status(strategy_name)
            result["edge_status"] = {
                "is_healthy": status.is_healthy,
                "health_score": status.health_score,
                "active_alerts": [
                    {
                        "decay_type": a.decay_type.value,
                        "message": a.message,
                        "severity": a.severity,
                        "timestamp": a.timestamp.isoformat()
                    }
                    for a in status.active_alerts
                ],
                "recommendations": status.recommendations
            }
        
        return result
    
    def export_to_csv(
        self,
        strategy_name: str,
        period_days: int = 30
    ) -> str:
        """
        Export metrics history to CSV format.
        
        Args:
            strategy_name: Strategy name
            period_days: Days to include
            
        Returns:
            CSV formatted string
        """
        if not self.metrics_collector:
            return ""
        
        history = self.metrics_collector.get_performance_history(
            strategy_name, days=period_days
        )
        
        if not history:
            return ""
        
        # Header
        headers = [
            "timestamp",
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "total_pnl",
            "profit_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "expectancy"
        ]
        
        lines = [",".join(headers)]
        
        for m in history:
            row = [
                m.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                str(m.total_trades),
                str(m.winning_trades),
                str(m.losing_trades),
                f"{m.win_rate:.4f}",
                f"{m.total_pnl:.2f}",
                f"{m.profit_factor:.4f}",
                f"{m.sharpe_ratio:.4f}",
                f"{m.sortino_ratio:.4f}",
                f"{m.max_drawdown_pct:.4f}",
                f"{m.expectancy:.6f}"
            ]
            lines.append(",".join(row))
        
        return "\n".join(lines)
    
    # ==================== Telegram Integration ====================
    
    def send_daily_report(self, strategies: List[str]) -> bool:
        """
        Generate and send daily report via Telegram.
        
        Args:
            strategies: List of strategy names
            
        Returns:
            True if sent successfully
        """
        if not self.telegram:
            return False
        
        try:
            report = self.generate_daily_summary(strategies=strategies)
            return self.telegram.send_message(report)
        except Exception as e:
            print(f"Error sending daily report: {e}")
            return False
    
    def send_status_alert(
        self,
        title: str,
        message: str,
        level: str = "warning"
    ) -> bool:
        """
        Send a status alert via Telegram.
        
        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)
            
        Returns:
            True if sent successfully
        """
        if not self.telegram:
            return False
        
        level_map = {
            "info": "INFO",
            "warning": "WARNING",
            "error": "ERROR",
            "critical": "CRITICAL"
        }
        
        try:
            from .telegram_notifier import AlertLevel
            alert_level = AlertLevel.WARNING
            level_key = level_map.get(level, "WARNING")
            if hasattr(AlertLevel, level_key):
                alert_level = getattr(AlertLevel, level_key)
            return self.telegram.send_alert(title, message, level=alert_level)
        except Exception:
            return False
