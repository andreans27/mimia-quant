# Mimia Quant Trading System

Systematic crypto quant trading system for Binance Futures — built for mid-frequency execution on both long and short sides.

## Architecture

```
├── src/
│   ├── core/          # Config, database, Redis, logging
│   ├── strategies/    # 5 trading strategies + backtesting
│   ├── execution/     # Order execution, position sizing, risk
│   ├── monitoring/    # Daily reports, Telegram, edge decay
│   └── utils/         # Binance REST + WebSocket clients
├── configs/           # Strategy & risk parameter configs
├── scripts/           # init_db, run_bot, test_api
├── data/              # SQLite DB + market data storage
├── tests/             # 69 passing tests
└── .github/workflows/ # CI/CD with GitHub Actions
```

## Quick Start

```bash
# Install dependencies
make install

# Initialize database
python scripts/init_db.py

# Run tests
make test

# Start trading bot
make run
```

## Strategies

| Strategy | Timeframe | Edge |
|----------|-----------|------|
| **Momentum** | 1m | Trend continuation after micro-breakouts |
| **Mean Reversion** | 1m | RSI/BB-based reversal at extremes |
| **Grid** | 5m | Range-bound price action exploitation |
| **Breakout** | 15m | Liquidity grab + volatility expansion |
| **Multi-Timeframe** | 1h+4h | Macro trend + micro entry alignment |

## Trading Parameters

- **Initial Capital**: 5,000 USDT (testnet)
- **Leverage**: 3x default, max 5x
- **Max Position**: 1.5% risk per trade
- **Max Daily DD**: 3% → reduce sizing 50%
- **Max Monthly DD**: 8% → full system halt

## CI/CD Pipeline

- **GitHub Actions** on every push: lint, typecheck, tests
- **Branch protection**: `main` requires PR + passing CI
- **Auto-deploy**: not enabled (manual trigger for safety)

## Telegram Reporting (08:00 UTC daily)

- Open positions & unrealized P&L
- Daily P&L & equity curve
- Strategy vote distribution
- Edge decay alerts
- Kill switch triggers

## Credentials

Configure in `.env` (copy from `.env.example`):

```env
BINANCE_TESTNET_API_KEY=your_key
BINANCE_TESTNET_API_SECRET=your_secret
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
```

## Current Status

🟡 **Testnet Mode** — Paper trading with live market data
- 69 tests passing
- All 5 strategies implemented
- Database initialized
- Telegram reporting configured
- API market data verified ✓
- API auth endpoints — requires valid testnet key

---

*Build: 2025-04-24 | Version: 2.0*
