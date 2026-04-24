"""
Logging configuration for Mimia Quant Trading System.

Provides structured logging with file and console handlers,
log rotation, and custom formatters for different components.
"""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from .constants import (
    DEFAULT_LOG_LEVEL,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    SYSTEM_NAME,
)


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with color."""
        if hasattr(record, "levelname") and record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )
        return super().format(record)


class TradingFormatter(logging.Formatter):
    """
    Specialized formatter for trading logs.
    Includes additional context like strategy name, symbol, and order ID.
    """
    
    TRADE_FORMAT = "%(asctime)s | %(levelname)-8s | [%(strategy)s] %(symbol)s | %(message)s"
    STANDARD_FORMAT = LOG_FORMAT
    
    def __init__(self, include_context: bool = True):
        super().__init__(fmt=self.TRADE_FORMAT if include_context else self.STANDARD_FORMAT)
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        # Add default values for trading context
        if not hasattr(record, "strategy"):
            record.strategy = "SYSTEM"
        if not hasattr(record, "symbol"):
            record.symbol = "-"
        return super().format(record)


def setup_logging(
    log_level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
) -> logging.Logger:
    """
    Configure logging for the Mimia Quant system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Created if it doesn't exist.
        log_to_file: Whether to write logs to file.
        log_to_console: Whether to output logs to console.
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of backup log files to keep.
    
    Returns:
        The configured root logger.
    """
    # Determine log level
    level = getattr(logging, (log_level or DEFAULT_LOG_LEVEL).upper(), logging.INFO)
    
    # Get or create root logger
    logger = logging.getLogger(SYSTEM_NAME)
    logger.setLevel(level)
    
    # Avoid adding multiple handlers if already configured
    if logger.handlers:
        return logger
    
    # Create log directory if needed
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # File handler with rotation
    if log_to_file:
        file_handler = RotatingFileHandler(
            filename=log_dir / f"{SYSTEM_NAME}.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Separate error log
        error_handler = RotatingFileHandler(
            filename=log_dir / f"{SYSTEM_NAME}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
    
    # Console handler with colors
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Use colored formatter for TTY, regular for non-TTY
        if sys.stdout.isatty():
            console_formatter = ColoredFormatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        else:
            console_formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
        
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module (usually __name__).
    
    Returns:
        A configured logger instance.
    
    Example:
        logger = get_logger(__name__)
        logger.info("Processing order", extra={"strategy": "momentum", "symbol": "BTC/USDT"})
    """
    return logging.getLogger(f"{SYSTEM_NAME}.{name}")


class TradingLogger:
    """
    Specialized logger for trading operations.
    Attaches additional context like strategy name and symbol.
    """
    
    def __init__(self, strategy_name: str, symbol: str = "-"):
        """
        Initialize trading logger.
        
        Args:
            strategy_name: Name of the trading strategy.
            symbol: Trading symbol (e.g., 'BTC/USDT').
        """
        self.logger = get_logger(f"trading.{strategy_name}")
        self.strategy_name = strategy_name
        self.symbol = symbol
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal log method with extra context."""
        extra = {
            "strategy": self.strategy_name,
            "symbol": self.symbol,
            **kwargs,
        }
        self.logger.log(level, message, extra=extra)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def order_submitted(self, order_id: str, side: str, quantity: float, price: Optional[float] = None):
        """Log order submission."""
        price_str = f"@ {price}" if price else "at MARKET"
        self.info(f"Order submitted: {order_id} | {side} {quantity} {price_str}")
    
    def order_filled(self, order_id: str, filled_qty: float, avg_price: float):
        """Log order fill."""
        self.info(f"Order filled: {order_id} | Filled: {filled_qty} @ {avg_price}")
    
    def order_cancelled(self, order_id: str, reason: str = ""):
        """Log order cancellation."""
        reason_str = f" | Reason: {reason}" if reason else ""
        self.warning(f"Order cancelled: {order_id}{reason_str}")
    
    def position_opened(self, side: str, entry_price: float, size: float):
        """Log position opening."""
        self.info(f"Position opened: {side} | Entry: {entry_price} | Size: {size}")
    
    def position_closed(self, pnl: float, exit_price: float):
        """Log position closing."""
        pnl_str = f"+{pnl:.2f}" if pnl >= 0 else f"{pnl:.2f}"
        self.info(f"Position closed: PnL: {pnl_str} | Exit: {exit_price}")
    
    def signal_generated(self, signal_type: str, strength: float, indicators: dict):
        """Log trading signal."""
        self.info(f"Signal generated: {signal_type} | Strength: {strength:.2%} | {indicators}")


# Convenience function to initialize logging from config
def init_logging_from_config(config) -> logging.Logger:
    """
    Initialize logging system using configuration object.
    
    Args:
        config: Config object with logging settings.
    
    Returns:
        Configured logger instance.
    """
    log_dir = Path("logs") / datetime.now().strftime("%Y%m%d")
    
    return setup_logging(
        log_level=config.get("system.log_level", DEFAULT_LOG_LEVEL),
        log_dir=log_dir,
        log_to_file=config.get("monitoring.logging.file_enabled", True),
        log_to_console=True,
    )
