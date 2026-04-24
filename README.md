# Mimia Quant Trading System

A modular, event-driven quantitative trading system with support for multiple exchanges, strategies, and risk management.

## Features

- **Multi-Exchange Support**: Binance, Alpaca, Coinbase, and more
- **Strategy Framework**: Easy-to-extend strategy base classes
- **Risk Management**: Built-in position sizing, stop-loss, and drawdown controls
- **Real-time Monitoring**: Prometheus metrics and Grafana dashboards
- **Async Execution**: Fully asynchronous order execution
- **Configurable**: YAML-based configuration with environment variable overrides

## Project Structure

```
mimia-quant/
├── config/              # Main configuration files
│   ├── config.yaml     # System configuration
│   └── ...
├── configs/            # Additional configurations
│   └── strategies.yaml # Strategy-specific settings
├── src/                # Source code
│   ├── core/          # Core components (base classes, config, logging)
│   ├── strategies/    # Trading strategies
│   ├── execution/     # Order execution handlers
│   ├── monitoring/    # Monitoring and metrics
│   └── utils/         # Utility functions
├── data/              # Data storage
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── tests/             # Test suite
└── logs/              # Log files
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/mimia-quant.git
cd mimia-quant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment:
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Configuration

Configuration is loaded from multiple sources with the following precedence:
1. Environment variables (highest priority)
2. YAML configuration files
3. Default values (lowest priority)

### Main Configuration (`config/config.yaml`)

```yaml
system:
  name: "mimia-quant"
  version: "1.0.0"
  environment: "development"

trading:
  max_position_size: 0.1  # 10% of portfolio
  max_daily_loss: 0.05    # 5% max daily loss

risk:
  max_drawdown: 0.15      # 15% max drawdown
  max_leverage: 3
```

### Strategy Configuration (`configs/strategies.yaml`)

Configure individual strategy parameters including indicators, position sizing, and filters.

## Usage

### Basic Example

```python
from src.core import (
    Config,
    setup_logging,
    get_logger,
    BaseStrategy,
)

# Initialize configuration and logging
config = Config()
config.load()
logger = setup_logging(log_level=config.log_level)

# Create a custom strategy
class MyStrategy(BaseStrategy):
    def analyze(self, symbol, data):
        # Implement strategy logic
        return signal
    
    def calculate_position_size(self, signal, portfolio_value):
        return 0.05  # 5% of portfolio

# Run strategy
strategy = MyStrategy("my_strategy", config.get_strategy("momentum"))
```

### Running in Production

```bash
# Set environment
export ENVIRONMENT=production
export LOG_LEVEL=INFO

# Run the trading system
python -m src.main
```

## Available Strategies

- **Momentum**: Trend-following strategy using moving averages
- **Mean Reversion**: Returns to average price strategy
- **Grid**: Grid trading with automatic order placement
- **Breakout**: Support/resistance breakout strategy

## Risk Management

The system includes built-in risk management features:

- Position size limits
- Daily loss limits
- Maximum drawdown protection
- Correlation-based position filtering
- Kelly criterion for position sizing

## Monitoring

- **Prometheus Metrics**: Port 9090
- **Health Check**: Port 8080
- **Grafana Dashboard**: Configurable

## Development

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
black src/
flake8 src/
mypy src/
```

## License

MIT License - see LICENSE file for details

## Support

For issues and feature requests, please use the GitHub issue tracker.
