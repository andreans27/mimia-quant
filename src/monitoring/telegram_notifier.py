"""
Telegram Notifier for Mimia Quant Trading System.

Sends alerts, notifications, and reports to Telegram chats.
Supports formatting, markdown, and different notification types.
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import requests
import threading

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    PROFIT = "profit"
    LOSS = "loss"


class NotificationType(Enum):
    """Types of notifications."""
    ALERT = "alert"
    REPORT = "report"
    SUMMARY = "summary"
    EDGE_DECAY = "edge_decay"
    REGIME_CHANGE = "regime_change"
    DRAWOWN_ALERT = "drawdown_alert"
    TRADE_EXECUTED = "trade_executed"
    DAILY_SUMMARY = "daily_summary"
    HEALTH_CHECK = "health_check"


class TelegramNotifier:
    """
    Telegram notification service.
    
    Sends formatted messages to Telegram chats using the Bot API.
    Supports markdown formatting, threading, and rate limiting.
    """
    
    API_URL = "https://api.telegram.org/bot{token}/sendMessage"
    MAX_MESSAGE_LENGTH = 4096
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        rate_limit_seconds: float = 1.0,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram bot token (or TELEGRAM_BOT_TOKEN env var)
            chat_id: Target chat ID (or TELEGRAM_CHAT_ID env var)
            rate_limit_seconds: Minimum time between messages
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        
        self._last_send_time = 0.0
        self._lock = threading.Lock()
        self._enabled = bool(self.bot_token and self.chat_id)
        
        if not self._enabled:
            logger.warning("Telegram notifier disabled: missing bot_token or chat_id")
    
    @property
    def enabled(self) -> bool:
        """Check if notifier is enabled."""
        return self._enabled
    
    def _rate_limit(self) -> None:
        """Apply rate limiting."""
        with self._lock:
            import time
            elapsed = time.time() - self._last_send_time
            if elapsed < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - elapsed)
            self._last_send_time = time.time()
    
    def _send_request(
        self,
        method: str,
        data: Dict[str, Any]
    ) -> Optional[Dict]:
        """Send HTTP request to Telegram API."""
        if not self._enabled:
            return None
        
        url = self.API_URL.format(token=self.bot_token)
        
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=data,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Telegram request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
                
        return None
    
    def send_message(
        self,
        text: str,
        parse_mode: str = "Markdown",
        disable_web_page_preview: bool = True,
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None
    ) -> bool:
        """
        Send a text message.
        
        Args:
            text: Message text (max 4096 characters)
            parse_mode: Parse mode (Markdown, HTML)
            disable_web_page_preview: Disable link previews
            disable_notification: Send silently
            reply_to_message_id: Reply to specific message
            
        Returns:
            True if sent successfully
        """
        if not self._enabled:
            return False
        
        self._rate_limit()
        
        # Truncate if needed
        if len(text) > self.MAX_MESSAGE_LENGTH:
            text = text[:self.MAX_MESSAGE_LENGTH - 3] + "..."
        
        data = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
            "disable_notification": disable_notification
        }
        
        if reply_to_message_id:
            data["reply_to_message_id"] = reply_to_message_id
        
        result = self._send_request("sendMessage", data)
        if result and result.get("ok"):
            logger.debug(f"Telegram message sent successfully")
            return True
        
        return False
    
    def send_alert(
        self,
        title: str,
        message: str,
        level: AlertLevel = AlertLevel.INFO,
        include_timestamp: bool = True
    ) -> bool:
        """
        Send an alert notification.
        
        Args:
            title: Alert title
            message: Alert body
            level: Alert severity level
            include_timestamp: Include timestamp in message
            
        Returns:
            True if sent successfully
        """
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
            AlertLevel.PROFIT: "💰",
            AlertLevel.LOSS: "📉"
        }
        
        emoji = level_emoji.get(level, "ℹ️")
        
        text = f"{emoji} *{title}*\n\n{message}"
        
        if include_timestamp:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            text += f"\n\n⏰ {timestamp}"
        
        return self.send_message(text)
    
    def send_trade_notification(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        strategy_name: str,
        holding_period: Optional[str] = None
    ) -> bool:
        """
        Send trade execution notification.
        
        Args:
            symbol: Trading symbol
            side: BUY or SELL
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position size
            pnl: Profit/loss amount
            pnl_pct: Profit/loss percentage
            strategy_name: Strategy name
            holding_period: Optional holding period string
            
        Returns:
            True if sent successfully
        """
        side_emoji = "🟢" if side.upper() == "BUY" else "🔴"
        pnl_emoji = "💰" if pnl >= 0 else "📉"
        
        text = (
            f"{side_emoji} *Trade Executed*\n\n"
            f"📊 Strategy: `{strategy_name}`\n"
            f"🪙 Symbol: *{symbol}*\n"
            f"📈 Side: {side.upper()}\n"
            f"💵 Entry: `${entry_price:,.4f}`\n"
            f"💵 Exit: `${exit_price:,.4f}`\n"
            f"📊 Qty: `{quantity:,.4f}`\n"
            f"{pnl_emoji} PnL: `${pnl:,.2f}` ({pnl_pct:+.2f}%)"
        )
        
        if holding_period:
            text += f"\n⏱️ Held: {holding_period}"
        
        return self.send_message(text)
    
    def send_performance_report(
        self,
        strategy_name: str,
        metrics: Dict[str, Any],
        period: str = "daily"
    ) -> bool:
        """
        Send performance report.
        
        Args:
            strategy_name: Strategy name
            metrics: Dictionary of performance metrics
            period: Report period (daily, weekly, monthly)
            
        Returns:
            True if sent successfully
        """
        period_emoji = {"daily": "📅", "weekly": "📆", "monthly": "🗓️"}.get(
            period, "📊"
        )
        
        total_trades = metrics.get("total_trades", 0)
        win_rate = metrics.get("win_rate", 0) * 100
        profit_factor = metrics.get("profit_factor", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown_pct", 0)
        total_pnl = metrics.get("total_pnl", 0)
        
        win_emoji = "🟢" if win_rate >= 50 else "🔴"
        pf_emoji = "✅" if profit_factor >= 1.5 else "⚠️"
        dd_emoji = "🚨" if max_dd > 10 else "✅"
        
        text = (
            f"{period_emoji} *{period.title()} Performance Report*\n\n"
            f"📊 Strategy: `{strategy_name}`\n\n"
            f"📈 Total Trades: `{total_trades}`\n"
            f"{win_emoji} Win Rate: `{win_rate:.1f}%`\n"
            f"{pf_emoji} Profit Factor: `{profit_factor:.2f}`\n"
            f"📉 Sharpe Ratio: `{sharpe:.2f}`\n"
            f"{dd_emoji} Max Drawdown: `{max_dd:.1f}%`\n\n"
            f"💵 Total PnL: `${total_pnl:,.2f}`"
        )
        
        return self.send_message(text)
    
    def send_edge_decay_alert(
        self,
        strategy_name: str,
        decay_type: str,
        current_value: float,
        threshold_value: float,
        drop_percentage: float,
        additional_info: Optional[str] = None
    ) -> bool:
        """
        Send edge decay alert.
        
        Args:
            strategy_name: Strategy name
            decay_type: Type of decay (win_rate, profit_factor, sharpe)
            current_value: Current metric value
            threshold_value: Threshold that was crossed
            drop_percentage: Percentage drop from baseline
            additional_info: Optional additional context
            
        Returns:
            True if sent successfully
        """
        decay_emoji = "⚠️"
        
        text = (
            f"{decay_emoji} *Edge Decay Detected*\n\n"
            f"📊 Strategy: `{strategy_name}`\n"
            f"🔍 Decay Type: *{decay_type}*\n"
            f"📉 Current: `{current_value:.4f}`\n"
            f"⚠️ Threshold: `{threshold_value:.4f}`\n"
            f"📊 Drop: `{drop_percentage:.1f}%`"
        )
        
        if additional_info:
            text += f"\n\n📝 Info: {additional_info}"
        
        return self.send_message(text)
    
    def send_regime_change_alert(
        self,
        symbol: str,
        old_regime: str,
        new_regime: str,
        confidence: float,
        recommendations: Optional[List[str]] = None
    ) -> bool:
        """
        Send regime change notification.
        
        Args:
            symbol: Trading symbol
            old_regime: Previous regime
            new_regime: New regime
            confidence: Confidence in detection (0-1)
            recommendations: Optional action recommendations
            
        Returns:
            True if sent successfully
        """
        text = (
            f"🔄 *Regime Change Detected*\n\n"
            f"🪙 Symbol: *{symbol}*\n"
            f"📍 Old Regime: `{old_regime}`\n"
            f"📍 New Regime: *{new_regime}*\n"
            f"🎯 Confidence: `{confidence:.0%}`"
        )
        
        if recommendations:
            text += "\n\n📋 *Recommendations:*"
            for rec in recommendations:
                text += f"\n• {rec}"
        
        return self.send_message(text)
    
    def send_drawdown_alert(
        self,
        strategy_name: str,
        current_drawdown: float,
        max_allowed_drawdown: float,
        current_equity: float,
        period: str = "today"
    ) -> bool:
        """
        Send drawdown alert.
        
        Args:
            strategy_name: Strategy name
            current_drawdown: Current drawdown percentage
            max_allowed_drawdown: Maximum allowed drawdown
            current_equity: Current equity value
            period: Time period
            
        Returns:
            True if sent successfully
        """
        severity = ""
        if current_drawdown >= max_allowed_drawdown:
            severity = "🚨 *CRITICAL*"
        elif current_drawdown >= max_allowed_drawdown * 0.8:
            severity = "⚠️ *WARNING*"
        
        text = (
            f"{severity}\n"
            f"📉 *Drawdown Alert*\n\n"
            f"📊 Strategy: `{strategy_name}`\n"
            f"📉 Current DD: `{current_drawdown:.2f}%`\n"
            f"⚠️ Max Allowed: `{max_allowed_drawdown:.2f}%`\n"
            f"💵 Equity: `${current_equity:,.2f}`\n"
            f"📅 Period: {period}"
        )
        
        return self.send_message(text)
    
    def send_health_check(
        self,
        status: str,
        checks: Dict[str, bool],
        latency_ms: Optional[float] = None
    ) -> bool:
        """
        Send system health check notification.
        
        Args:
            status: Overall status (healthy, degraded, unhealthy)
            checks: Dictionary of check results
            latency_ms: Optional API latency
            
        Returns:
            True if sent successfully
        """
        status_emoji = {
            "healthy": "✅",
            "degraded": "⚠️",
            "unhealthy": "❌"
        }
        emoji = status_emoji.get(status.lower(), "❓")
        
        text = (
            f"{emoji} *System Health Check*\n\n"
            f"Status: *{status.upper()}*"
        )
        
        for check_name, passed in checks.items():
            check_emoji = "✅" if passed else "❌"
            text += f"\n{check_emoji} {check_name}"
        
        if latency_ms is not None:
            text += f"\n\n⏱️ Latency: `{latency_ms:.0f}ms`"
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        text += f"\n\n⏰ {timestamp}"
        
        return self.send_message(text)
    
    def send_daily_summary(
        self,
        total_pnl: float,
        total_trades: int,
        win_rate: float,
        best_strategy: str,
        best_pnl: float,
        worst_strategy: str,
        worst_pnl: float,
        regime_overview: Dict[str, Any]
    ) -> bool:
        """
        Send daily summary report.
        
        Args:
            total_pnl: Total daily PnL
            total_trades: Total trades executed
            win_rate: Overall win rate
            best_strategy: Best performing strategy name
            best_pnl: Best strategy PnL
            worst_strategy: Worst performing strategy name
            worst_pnl: Worst strategy PnL
            regime_overview: Overview of market regimes
            
        Returns:
            True if sent successfully
        """
        pnl_emoji = "💰" if total_pnl >= 0 else "📉"
        
        text = (
            f"📊 *Daily Summary Report*\n"
            f"{'='*30}\n\n"
            f"{pnl_emoji} *Total PnL:* `${total_pnl:,.2f}`\n"
            f"📈 *Total Trades:* `{total_trades}`\n"
            f"🟢 *Win Rate:* `{win_rate:.1f}%`\n\n"
            f"🏆 *Best Strategy:* `{best_strategy}`\n"
            f"   PnL: `${best_pnl:,.2f}`\n\n"
            f"📉 *Worst Strategy:* `{worst_strategy}`\n"
            f"   PnL: `${worst_pnl:,.2f}`"
        )
        
        if regime_overview:
            text += "\n\n📍 *Market Regimes:*"
            for regime, count in regime_overview.items():
                text += f"\n• {regime}: {count} symbols"
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        text += f"\n\n⏰ Report Time: {timestamp} 08:00 UTC"
        
        return self.send_message(text)
    
    def send_risk_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send risk management alert.
        
        Args:
            alert_type: Type of risk alert
            message: Alert message
            details: Optional additional details
            
        Returns:
            True if sent successfully
        """
        text = (
            f"🚨 *Risk Alert: {alert_type}*\n\n"
            f"📝 {message}"
        )
        
        if details:
            text += "\n\n📋 *Details:*"
            for key, value in details.items():
                text += f"\n• {key}: `{value}`"
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        text += f"\n\n⏰ {timestamp}"
        
        return self.send_message(text)
    
    def send_bulk_messages(
        self,
        messages: List[Dict[str, str]]
    ) -> int:
        """
        Send multiple messages in sequence.
        
        Args:
            messages: List of message dicts with 'text' and optional 'parse_mode'
            
        Returns:
            Number of successfully sent messages
        """
        success_count = 0
        
        for msg in messages:
            text = msg.get("text", "")
            parse_mode = msg.get("parse_mode", "Markdown")
            
            if self.send_message(text, parse_mode=parse_mode):
                success_count += 1
        
        return success_count


class TelegramFormatter:
    """Helper class for formatting Telegram messages."""
    
    @staticmethod
    def format_metrics_table(
        metrics: Dict[str, tuple],
        title: Optional[str] = None
    ) -> str:
        """
        Format metrics as a table.
        
        Args:
            metrics: Dict of {label: (value, unit)}
            title: Optional table title
            
        Returns:
            Formatted table string
        """
        lines = []
        
        if title:
            lines.append(f"*{title}*")
            lines.append("")
        
        for label, (value, unit) in metrics.items():
            if unit:
                lines.append(f"• {label}: `{value} {unit}`")
            else:
                lines.append(f"• {label}: `{value}`")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_position_status(
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        unrealized_pnl: float,
        unrealized_pnl_pct: float
    ) -> str:
        """Format position status message."""
        side_emoji = "🟢" if side.upper() == "LONG" else "🔴"
        pnl_emoji = "💰" if unrealized_pnl >= 0 else "📉"
        
        return (
            f"{side_emoji} *Position Status*\n\n"
            f"🪙 Symbol: *{symbol}*\n"
            f"📊 Side: {side}\n"
            f"💵 Entry: `${entry_price:,.4f}`\n"
            f"💵 Current: `${current_price:,.4f}`\n"
            f"📊 Size: `{quantity:,.4f}`\n"
            f"{pnl_emoji} Unrealized: `${unrealized_pnl:,.2f}` ({unrealized_pnl_pct:+.2f}%)"
        )
    
    @staticmethod
    def format_signal(
        symbol: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        strategy_name: str
    ) -> str:
        """Format trading signal message."""
        action_emoji = "🟢" if action.upper() == "BUY" else "🔴"
        
        return (
            f"{action_emoji} *Trading Signal*\n\n"
            f"📊 Strategy: `{strategy_name}`\n"
            f"🪙 Symbol: *{symbol}*\n"
            f"📈 Action: *{action.upper()}*\n"
            f"💵 Entry: `${entry_price:,.4f}`\n"
            f"🛑 Stop Loss: `${stop_loss:,.4f}`\n"
            f"🎯 Take Profit: `${take_profit:,.4f}`\n"
            f"🎯 Confidence: `{confidence:.0%}`"
        )
