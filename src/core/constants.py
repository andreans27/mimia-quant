"""
Constants for the Mimia Quant Trading System.

This module contains all system-wide constants including enums,
status values, timeframes, and other fixed configuration values.
"""

from enum import Enum
from typing import Final


# =============================================================================
# VERSION INFORMATION
# =============================================================================
VERSION: Final[str] = "1.0.0"
SYSTEM_NAME: Final[str] = "mimia-quant"


# =============================================================================
# TRADING ENUMS
# =============================================================================

class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(Enum):
    """Position side enumeration."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TimeFrame(Enum):
    """Trading timeframe enumeration."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    DAY_1_WEEK_1 = "1w"
    MONTH_1 = "1M"


class Exchange(Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    ALPACA = "alpaca"
    COINBASE = "coinbase"
    KRACKEN = "kraken"


# =============================================================================
# RISK LIMITS
# =============================================================================

MAX_POSITION_SIZE: Final[float] = 0.1  # 10% of portfolio
MAX_TOTAL_EXPOSURE: Final[float] = 0.8  # 80% max exposure
MAX_LEVERAGE: Final[int] = 3
MAX_DAILY_LOSS: Final[float] = 0.05  # 5%
MAX_DRAWDOWN: Final[float] = 0.15  # 15%
MIN_RISK_REWARD_RATIO: Final[float] = 2.0

# =============================================================================
# ORDER LIMITS
# =============================================================================

DEFAULT_STOP_LOSS_PCT: Final[float] = 2.0
DEFAULT_TAKE_PROFIT_PCT: Final[float] = 5.0
DEFAULT_SLIPPAGE_TOLERANCE: Final[float] = 0.001  # 0.1%
ORDER_TIMEOUT_SECONDS: Final[int] = 30
RATE_LIMIT_RPS: Final[int] = 10


# =============================================================================
# STRATEGY CONSTANTS
# =============================================================================

KELLY_FRACTION: Final[float] = 0.25  # Kelly criterion fraction to use
MIN_CONFIDENCE_SCORE: Final[float] = 0.6
MOMENTUM_LOOKBACK_PERIOD: Final[int] = 20
MEAN_REVERSION_LOOKBACK_PERIOD: Final[int] = 50
GRID_MIN_LEVELS: Final[int] = 5
GRID_MAX_LEVELS: Final[int] = 50


# =============================================================================
# TIME CONSTANTS
# =============================================================================

SECONDS_PER_MINUTE: Final[int] = 60
SECONDS_PER_HOUR: Final[int] = 3600
SECONDS_PER_DAY: Final[int] = 86400
MILLISECONDS_PER_SECOND: Final[int] = 1000


# =============================================================================
# API CONSTANTS
# =============================================================================

MAX_RETRIES: Final[int] = 3
RETRY_BACKOFF_FACTOR: Final[float] = 2.0
CONNECTION_TIMEOUT: Final[int] = 10
READ_TIMEOUT: Final[int] = 30


# =============================================================================
# DATABASE CONSTANTS
# =============================================================================

REDIS_KEY_PREFIX: Final[str] = "mimia:"
POSITION_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}position:"
ORDER_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}order:"
METRIC_PREFIX: Final[str] = f"{REDIS_KEY_PREFIX}metric:"
CACHE_TTL_SECONDS: Final[int] = 300  # 5 minutes


# =============================================================================
# LOGGING CONSTANTS
# =============================================================================

LOG_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
LOG_DATE_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
