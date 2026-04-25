# ⚡ Mimia — Systematic Crypto Quant Trading System

[![CI](https://img.shields.io/github/actions/workflow/status/andreans27/mimia-quant/ci.yml?branch=main&label=CI&logo=github)](https://github.com/andreans27/mimia-quant/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)](https://github.com/andreans27/mimia-quant/actions)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

> **Mimia** is a machine-learning-driven quantitative trading system for Binance Crypto Perpetual Futures. It trades both long and short using a multi-timeframe XGBoost ensemble, with real-time signal generation, position management, and edge decay monitoring — all running autonomously via scheduled cron jobs.

---

## 📊 Live Status

| Metric | Value |
|--------|-------|
| **Mode** | 🟡 Live Trading (Testnet) |
| **Capital** | ~$4,996 USDT |
| **Pairs** | 10 (APT, UNI, FET, TIA, SOL, OP, 1000PEPE, SUI, ARB, INJ) |
| **Strategy** | Multi-TF XGBoost Ensemble (5m, 15m, 30m, 1h, 4h) |
| **Signal Threshold** | 0.60 (optimal via grid scan) |
| **Avg Win Rate (backtest)** | 70–82.5% across symbols |
| **Scan Interval** | Every 60 minutes |
| **Open Positions** | 3 (SOL, 1000PEPE, ARB) — all LONG |

---

## ✨ Features

### ML‑Driven Signal Pipeline
- **Multi‑Timeframe Ensemble** — 5 XGBoost models per symbol (5m, 15m, 30m, 1h, 4h), averaged into a single probability score
- **372 engineered features** — price action, volatility, volume profile, microstructure, and regime indicators
- **Threshold‑optimized entries** — grid scan over 0.50–0.90 for optimal Sharpe/WR/DD tradeoff per symbol
- **Overfitting controls** — feature subsampling, strong L1/L2 regularization (α=1.0, λ=3.0), walk‑forward validation, independent OOS sets

### Bidirectional Trading
- No directional bias — evaluates long and short setups with identical rigor
- Short entries validated with regime confirmation, funding rate, and orderbook depth
- Long/short performance tracked independently in reporting

### Risk Management
- **1% risk per trade** (Kelly‑0.25x sizing)
- **Multi‑stage kill switches** — 3% daily DD → 50% size cut, 5% → halt, 8% monthly → full audit
- **Correlated position tracking** — correlated assets counted as single exposure
- **Staged entries** — 2–3 tranches for larger positions
- **5x max leverage** (10x on testnet for signal‑to‑noise)

### Production‑Grade Tooling
- **Binance Testnet Integration** — real order execution on paper environment
- **Telegram Reporting** — daily reports via Telegram bot
- **Cron‑Based Scheduling** — autonomous signal scan every 60 minutes
- **SQLite Persistence** — trade log, capital history, signal records
- **CI/CD Pipeline** — GitHub Actions with linting, testing, package build, and pre‑deploy validation

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                   Cron Scheduler                      │
|           Every 60 min → scripts/trading/live_trader.py               │
├──────────────────────────────────────────────────────┤
│               Live Trading Engine                       │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Signal    │  │ Position │  │ Risk /             │  │
│  │ Pipeline  │→ │ Manager  │→ │ Position Sizing    │  │
│  │ (ML Ens.) │  │          │  │ (Kelly 0.25)       │  │
│  └──────────┘  └──────────┘  └────────────────────┘  │
├──────────────────────────────────────────────────────┤
│                   Data Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ Parquet  │  │ SQLite   │  │ Binance REST       │  │
│  │ Cache    │→ │ DB       │→ │ Client (Testnet)   │  │
│  └──────────┘  └──────────┘  └────────────────────┘  │
├──────────────────────────────────────────────────────┤
│                   Infrastructure                      │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ GitHub   │  │ Telegram │  │ Hermes Cron        │  │
│  │ Actions  │  │ Bot      │  │ Scheduler          │  │
│  └──────────┘  └──────────┘  └────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### Component Breakdown

| Layer | Component | Description |
|-------|-----------|-------------|
| **Domain** | `strategy.py`, `signal.py` | Core business logic — signal generation, edge detection |
| **Application** | `scripts/trading/live_trader.py` | Orchestrates signal evaluation, trade execution, risk checks |
| **Infrastructure** | `binance_sdk`, `sqlite`, `telegram` | Exchange connectivity, persistence, notifications |

---

## 🧠 ML Pipeline

### Training Flow
```
Market Data (5m, 15m, 30m, 1h, 4h)
        ↓
Feature Engineering (372 features)
        ↓
XGBoost Training (depth=3, reg_alpha=1.0, reg_lambda=3.0)
        ↓
5 Seeds → 5 Models per Timeframe → 25 Models per Symbol
        ↓
Grid Threshold Scan (0.50–0.90)
        ↓
Optimal Threshold Selected → Live Trade

16 symbols × 25 models = 400 models total
```

### Model Architecture
- **Algorithm**: XGBoost Regressor (reg:squarederror)
- **Architecture**: depth=3, 300 estimators, L1+L2 regularization
- **Ensemble**: 5 timeframes × 5 seeds = 25 models per symbol; probability averaged across all
- **Feature Groups**: price action (120), volatility (60), volume (60), microstructure (72), regime (60)

### Performance (Backtest — All 8/8 Symbols Passed)

| Symbol | Win Rate | Profit Factor | Monthly Return | Max DD |
|--------|----------|---------------|----------------|--------|
| APT | 82.5% | 10.1 | 33.4% | <0.35% |
| UNI | 80.1% | 8.4 | 28.2% | <0.35% |
| FET | 79.5% | 6.7 | 25.1% | <0.35% |
| TIA | 79.5% | 5.9 | 24.8% | <0.35% |
| SOL | 77.3% | 4.8 | 22.1% | <0.35% |
| OP | 76.1% | 4.2 | 19.7% | <0.35% |
| SUI | 75.6% | 3.9 | 18.5% | <0.35% |
| ARB | 75.1% | 3.8 | 18.0% | <0.35% |

~0.35% max DD achieved through probability-based position sizing with strong risk management.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Binance Testnet API key ([register here](https://testnet.binancefuture.com/))
- Telegram bot token (optional, for reporting)

### Installation

```bash
git clone https://github.com/andreans27/mimia-quant.git
cd mimia-quant
python3 -m venv venv
source venv/bin/activate
make install

# Configure credentials
cp .env.example .env
# Edit .env with your Binance testnet keys
```

### Run Tests

```bash
make test
```

### Start Live Trading


```bash
python scripts/trading/live_trader.py
```

Or deploy as a cron job (runs automatically every 60 min):
```bash
# Via Hermes cron scheduler
# cron schedule: every 1h
```

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | 5,000 USDT | Starting capital for live trading |
| `LEVERAGE_X` | 10 | Leverage (testnet only) |
| `RISK_PER_TRADE` | 1% | Max risk per position |
| `SIGNAL_THRESHOLD` | 0.60 | Minimum ensemble probability to enter |
| `COOLDOWN_BARS` | 12 | Time before re-evaluating same symbol |
| `HOLD_BARS` | 12 | Max hold time before auto-close |

---

## 📈 CI/CD Pipeline

Every push to `main` triggers:

1. 🔍 **Lint & Type Check** — flake8, mypy
2. 🧪 **Unit Tests** — pytest (69 tests)
3. 🔒 **Security Scan** — bandit
4. 📦 **Build Package** — python -m build
5. ✅ **Pre-Deploy Validation** — YAML config verification
6. 🚀 **Deploy (manual)** — staging → production (approval gates)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core runtime |
| XGBoost 3.2.0 | Gradient-boosted ML models |
| Pandas 3.0 | Data processing & feature engineering |
| NumPy 2.4 | Numerical computation |
| PyTorch 2.11 | RL position sizer (DQN) |
| SQLite | Trade & capital persistence |
| GitHub Actions | CI/CD pipeline |
| Binance Testnet | Paper trading execution |

---

## 📋 Roadmap

- [x] ML pipeline — Multi-TF XGBoost ensemble
- [x] Live trading engine — live testnet integration
- [x] Risk management — Kelly sizing + kill switches
- [x] CI/CD — GitHub Actions pipeline
- [ ] Live deployment — mainnet with reduced sizing
- [ ] RL optimizer — DQN-based position sizing (in progress)
- [ ] Edge decay monitoring — rolling 30/60/90-day performance windows
- [ ] Telegram daily reporting — automated daily P&L summaries

---

## 🤝 License

MIT — use at your own risk. Trading futures carries significant financial risk.

---

<p align="center">
  <sub>Built with ❤️ for systematic alpha. <strong>Live to trade another day.</strong></sub>
</p>
