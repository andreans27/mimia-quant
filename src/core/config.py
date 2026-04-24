"""
Configuration module for Mimia Quant Trading System.

Handles loading and managing configuration from YAML files and environment variables.
Supports multiple configuration sources with proper precedence: env vars > config files.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv

from .constants import (
    DEFAULT_LOG_LEVEL,
    DEFAULT_SLIPPAGE_TOLERANCE,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    MAX_DAILY_LOSS,
    MAX_DRAWDOWN,
    MAX_LEVERAGE,
    MAX_POSITION_SIZE,
    MAX_TOTAL_EXPOSURE,
    MIN_RISK_REWARD_RATIO,
    VERSION,
)


class Config:
    """
    Main configuration class for the trading system.
    
    Loads configuration from multiple sources with the following precedence:
    1. Environment variables (highest priority)
    2. YAML configuration files
    3. Default values (lowest priority)
    
    Example:
        config = Config()
        config.load()
        log_level = config.get("system.log_level")
    """
    
    DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    STRATEGIES_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "strategies.yaml"
    
    def __init__(self, config_path: Optional[Path] = None, load_env: bool = True):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Optional path to main config.yaml file.
            load_env: Whether to load .env file into environment variables.
        """
        self._config: Dict[str, Any] = {}
        self._strategies_config: Dict[str, Any] = {}
        self._env_loaded: bool = False
        
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        
        if load_env:
            self._load_env()
    
    def _load_env(self) -> None:
        """Load environment variables from .env file."""
        env_path = Path(__file__).parent.parent.parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            self._env_loaded = True
    
    def load(self, config_path: Optional[Path] = None) -> None:
        """
        Load configuration from YAML file(s).
        
        Args:
            config_path: Optional path to config file. Uses default if not provided.
        """
        if config_path:
            self.config_path = config_path
        
        self._load_main_config()
        self._load_strategies_config()
        self._apply_env_overrides()
    
    def _load_main_config(self) -> None:
        """Load the main configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}
    
    def _load_strategies_config(self) -> None:
        """Load strategies configuration file."""
        if self.STRATEGIES_CONFIG_PATH.exists():
            with open(self.STRATEGIES_CONFIG_PATH, "r") as f:
                self._strategies_config = yaml.safe_load(f) or {}
        else:
            self._strategies_config = {}
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # System settings
        if os.getenv("LOG_LEVEL"):
            self._config.setdefault("system", {})["log_level"] = os.getenv("LOG_LEVEL")
        if os.getenv("DEBUG"):
            self._config.setdefault("system", {})["debug"] = os.getenv("DEBUG").lower() == "true"
        if os.getenv("ENVIRONMENT"):
            self._config.setdefault("system", {})["environment"] = os.getenv("ENVIRONMENT")
        
        # Trading settings overrides
        if os.getenv("MAX_POSITION_SIZE"):
            self._config.setdefault("trading", {})["max_position_size"] = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
        if os.getenv("MAX_DAILY_LOSS"):
            self._config.setdefault("risk", {})["max_daily_loss"] = float(os.getenv("MAX_DAILY_LOSS", "0.05"))
        
        # Redis configuration
        if os.getenv("REDIS_HOST"):
            self._config.setdefault("redis", {})["host"] = os.getenv("REDIS_HOST")
        if os.getenv("REDIS_PORT"):
            self._config.setdefault("redis", {})["port"] = int(os.getenv("REDIS_PORT", "6379"))
        if os.getenv("REDIS_PASSWORD"):
            self._config.setdefault("redis", {})["password"] = os.getenv("REDIS_PASSWORD")
        
        # Exchange API keys
        if os.getenv("BINANCE_API_KEY"):
            self._config.setdefault("exchange", {}).setdefault("api_keys", {})["binance"] = {
                "api_key": os.getenv("BINANCE_API_KEY"),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "testnet": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to the config value (e.g., 'system.log_level').
            default: Default value if key is not found.
        
        Returns:
            The configuration value or default.
        
        Example:
            log_level = config.get("system.log_level", "INFO")
        """
        keys = key_path.split(".")
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy.
        
        Args:
            strategy_name: Name of the strategy.
        
        Returns:
            Strategy configuration dictionary.
        """
        return self._strategies_config.get("strategies", {}).get(strategy_name, {})
    
    def get_all_strategies(self) -> Dict[str, Any]:
        """
        Get all strategy configurations.
        
        Returns:
            Dictionary of all strategy configurations.
        """
        return self._strategies_config.get("strategies", {})
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.get("system.environment", "development") == "production"
    
    def is_sandbox(self) -> bool:
        """Check if running in sandbox/testnet mode."""
        return self.get("exchange.sandbox_mode", True)
    
    # Convenience properties for common settings
    @property
    def log_level(self) -> str:
        return self.get("system.log_level", DEFAULT_LOG_LEVEL)
    
    @property
    def debug(self) -> bool:
        return self.get("system.debug", False)
    
    @property
    def max_position_size(self) -> float:
        return self.get("trading.max_position_size", MAX_POSITION_SIZE)
    
    @property
    def max_daily_loss(self) -> float:
        return self.get("risk.max_daily_loss", MAX_DAILY_LOSS)
    
    @property
    def max_drawdown(self) -> float:
        return self.get("risk.max_drawdown", MAX_DRAWDOWN)
    
    @property
    def max_leverage(self) -> int:
        return self.get("trading.max_leverage", MAX_LEVERAGE)
    
    @property
    def redis_config(self) -> Dict[str, Any]:
        return self.get("redis", {})
    
    @property
    def database_config(self) -> Dict[str, Any]:
        return self.get("database", {})
    
    @property
    def exchange_config(self) -> Dict[str, Any]:
        return self.get("exchange", {})
    
    @property
    def risk_config(self) -> Dict[str, Any]:
        return self.get("risk", {})
    
    @property
    def monitoring_config(self) -> Dict[str, Any]:
        return self.get("monitoring", {})
    
    def __repr__(self) -> str:
        return f"<Config(version={VERSION}, env={self.get('system.environment', 'unknown')})>"
    
    def __str__(self) -> str:
        return f"Mimia Quant Config v{VERSION}"


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.
    
    Returns:
        The global Config instance.
    """
    global _config
    if _config is None:
        _config = Config()
        _config.load()
    return _config


def reload_config(config_path: Optional[Path] = None) -> Config:
    """
    Reload the global configuration.
    
    Args:
        config_path: Optional path to configuration file.
    
    Returns:
        The reloaded Config instance.
    """
    global _config
    _config = Config(config_path=config_path)
    _config.load()
    return _config
