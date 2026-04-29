# ⚡ Mimia — Systematic Crypto Quant Trading System

[![CI](https://img.shields.io/github/actions/workflow/status/andreans27/mimia-quant/ci.yml?branch=main&label=CI&logo=github)](https://github.com/andreans27/mimia-quant/actions)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)](https://github.com/andreans27/mimia-quant/actions)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Pairs](https://img.shields.io/badge/pairs-20-blueviolet)](https://github.com/andreans27/mimia-quant)

> **Mimia** is a machine-learning-driven quantitative trading system for Binance Crypto Perpetual Futures. It trades both long and short using a multi-timeframe XGBoost ensemble, with real-time signal generation, position management, and edge decay monitoring — all running autonomously via scheduled cron jobs on the Binance Testnet.

---

## 📊 Live Status

| Metric | Value |
|--------|-------|
| **Mode** | 🟡 Live Trading (Testnet) |
| **Capital** | $5,000 USDT |
| **Pairs** | 20 (see below) |
| **Strategy** | Multi-TF XGBoost Ensemble (5m, 15m, 30m, 1h, 4h) |
| **Signal Threshold** | 0.60 (optimal via grid scan across 12 symbols) |
| **Avg Win Rate (retrained pairs)** | 76.9–88.4% across symbols |
| **Scan Interval** | Every 5 minutes (daemon — synced to bar boundaries :00/:05/:10/...) |
| **Open Positions** | 0 (fresh start — deferred entry flow enabled) |

### Active Pairs

**All 20 pairs retrained & deployed (WR ≥ 70.2%, PF ≥ 8.1)**

| Range | Symbols |
|-------|---------|
| WR ≥ 85% | ENA, SUI, OP, FET, TIA, 1000PEPE, ARB, INJ, NEAR, ADA |
| WR 80–85% | AAVE, WIF, DOGE, SEI, LINK, WLD, AVAX |
| WR 70–80% | SOL, ETH, BNB |

---

## ✨ Features

### ML‑Driven Signal Pipeline
- **Multi‑Timeframe Ensemble** — 5 XGBoost models per symbol (5m, 15m, 30m, 1h, 4h), averaged into a single probability score
- **372 engineered features** — price action, volatility, volume profile, microstructure, and regime indicators
- **Threshold‑optimized entries** — grid scan over 0.50–0.90 for optimal Sharpe/WR/DD tradeoff per symbol
- **Overfitting controls** — feature subsampling, strong L1/L2 regularization (α=1.0, λ=3.0), walk‑forward validation, independent OOS sets
- **Automated retraining pipeline** — hourly cron job monitors edge decay (realistic thresholds: WR < 55% or PF < 1.5 triggers urgent retrain, 72h minimum interval, weekly cap 1x/symbol)

### Bidirectional Trading
- No directional bias — evaluates long and short setups with identical rigor
- Short entries validated with regime confirmation, funding rate, and orderbook depth
- Long/short performance tracked independently in reporting

### Risk Management
- **1% risk per trade** (Kelly‑0.25x sizing)
- **Multi‑stage kill switches** — 3% daily DD → 50% size cut, 5% → halt, 8% monthly → full audit
- **Correlated position tracking** — correlated assets counted as single exposure
- **Staged entries** — 2–3 tranches for larger positions
- **10x leverage max** on testnet; Kelly‑fractional automatic position sizing

### Production-Grade Tooling
- **Binance Testnet Integration** — real order execution on paper environment with accurate fee & slippage modeling
- **Deferred Entry Flow** — signals computed at candle N close, entries executed at candle N+1 close via `pending_signals` DB table. Ensures backtest ↔ live trader alignment.
- **Complete-Bar Timing Alignment** — three-layer fix ensures all four components (trainer, retrainer, backtester, live trader) use data from COMPLETED bars only: (1) `_is_bar_ready()` checks last COMPLETE bar (not latest cache bar), (2) `compute_5m_features_5tf(for_inference=True)` drops incomplete 5m bars, (3) `_enter_position()` uses last COMPLETE bar close for entry price. Forward-fill limits increased for inference (15m=9, 30m=18, 1h=36, 4h=120) to bridge gaps from dropped incomplete HTF bars.
- **Telegram Reporting** — automated reports every 5-minute daemon cycle (synced to bar boundaries :00/:05/:10) via Telegram bot
- **Daemon-Based Execution** — autonomous signal scan every 5 minutes via persistent trading daemon (see `scripts/trading/`)
- **SQLite Persistence** — trade log, capital history, signal records, pending signals
- **CI/CD Pipeline** — GitHub Actions with linting, testing, security scan, and pre‑deploy validation

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     Daemon Loop (5min cycle — synced to bar boundaries)               │
│          ┌──────────────────────────────────────────┐                                │
│          │ 1. Execute pending entries (from candle N)│                                │
│          │ 2. Exit positions due (hold_expiry)       │                                │
│          │ 3. Compute signals for candle N+1         │                                │
│          │    → store as pending for next cycle      │                                │
│          └──────────┬───────────────────────────────┘                                │
│          ┌──────────────────────────┐                             │
│          │ Cron Monitoring (hourly)  │  ← akan jadi daily         │
│          │ → auto_retrain.py        │                             │
│          └──────────────────────────┘                             │
├───────────────────┼───────────────────────────────────────────────┤
│                   ▼                                                │
│              Live Trading Engine                                  │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────────────────┐   │
│  │ Signal    │  │ Position │  │ Kelly  │  │ Regime /         │   │
│  │ Pipeline  │→│ Manager  │→│ Sizer  │→│ Exit Strategy    │   │
│  │ (ML Ens.) │  │          │  │        │  │ (ATR/SMA/Hold)   │   │
│  └──────────┘  └──────────┘  └────────┘  └──────────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                        Data Layer                                 │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ Parquet    │  │ SQLite       │  │ Binance REST Client      │ │
│  │ Feature    │→│ Trade/State  │→│ (Testnet — get_order      │ │
│  │ Cache      │  │ DB           │  │  polling, accurate PnL)  │ │
│  └────────────┘  └──────────────┘  └──────────────────────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                    Infrastructure                                  │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │ GitHub   │  │ Telegram   │  │ Hermes   │  │ ML Models     │ │
│  │ Actions  │  │ Bot        │  │ Cron     │  │ (data/ml_*   )│ │
│  └──────────┘  └────────────┘  └──────────┘  └───────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
mimia-quant/
├── src/
│   ├── strategies/     # Kelly sizer, regime filters, exit strategies
│   ├── trading/        # Engine, state, signals, reporter
│   ├── training/       # ML training pipeline (tf_specific, ml_ensemble)
│   ├── backtesting/    # Backtest engine, exit strategy comparison
│   └── core/           # Database, Redis client, base models
├── scripts/
│   ├── trading/        # Daemon launcher (live_trader_daemon.sh)
│   ├── operations/     # Cron orchestrator, Telegram reporter
│   ├── optimization/   # Threshold scans, regime filter backtests
│   └── training/       # Auto-retrain, batch eval, rapid eval
├── config/
│   ├── config.yaml     # System configuration
│   └── strategies.yaml # Strategy definitions
├── data/
│   ├── ml_models/      # Trained XGBoost models (20 symbols × 25 models)
│   ├── ml_cache/       # Feature cache (Parquet)
│   └── cron_output/    # Scheduled job output logs
└── tests/              # 69 unit tests (pytest)
```

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

20 symbols × 25 models = 500 models total
```

### Model Architecture
- **Algorithm**: XGBoost Regressor (reg:squarederror)
- **Architecture**: depth=3, 300 estimators, L1+L2 regularization
- **Ensemble**: 5 timeframes × 5 seeds = 25 models per symbol; probability averaged across all
- **Feature Groups**: price action (120), volatility (60), volume (60), microstructure (72), regime (60)
- **Voting Pipeline**: all 25 models vote, probability threshold 0.60 for entry

### Automated Retraining (Every Hour)

The system monitors live signal quality every hour and decides whether to retrain per symbol:

```
┌──────────────────────────────────────────────────────────┐
│  Decision Matrix — REALISTIC live thresholds              │
├─────────────────────────────────────┬────────────────────┤
│  Live WR ≥ 60% AND PF ≥ 2.0        │ 🟢 SKIP (edge OK)  │
│  WR 55-60% OR PF 1.5-2.0           │ 🟡 RETRAIN (time)  │
│                                      │   (every 72h)     │
│  WR 50-55% OR PF 1.2-1.5           │ 🟡 RETRAIN (decay) │
│  WR < 50% OR PF < 1.2              │ 🔴 RETRAIN (urgent)│
│  Weekly cap: 1x/symbol/week        │ 🟢 SKIP (capped)   │
└─────────────────────────────────────┴────────────────────┘
```

**Why realistic?** Backtest WR of 70% typically degrades to 55-60% in live trading
due to market regime shifts and adversarial adaptation. Thresholds calibrated for
sustainable live performance, not aspirational backtest numbers.

**Safeguards:**
- Minimum 20 trades from current model before evaluation
- Minimum 72h between retrains (24h for urgent WR < 50%)
- Weekly cap: 1 retrain per symbol per week
- Pre-deploy validation: new model must pass quality gate (WR ≥ 55%, PF ≥ 1.5)
- Auto-rollback: if new model fails, old model is restored from backup

### Trade History Integrity

The engine includes three layers of trade history protection:

| Feature | What It Fixes | When Run |
|---------|--------------|----------|
| `_close_stale_position()` | Daemon crash mid-trade — logs exit with real Binance fill data instead of silently resetting | On stale position detection |
| `_verify_position_integrity()` | Position closed on Binance mid-cycle — catches inconsistency and logs trade | Every 5-min daemon cycle |
| `_sync_trade_history()` | Missing trades from previous runs — pulls last 7 days of Binance fills, groups by entry/exit, backfills gaps | Once per daemon startup |

### Performance (Backtest — All 20 LIVE_SYMBOLS)

| Symbol | Win Rate | Profit Factor | Monthly Return | Max DD |
|--------|----------|---------------|----------------|--------|
| 1000PEPE | 88.7% | 22.05 | 42.5% | 0.25% |
| NEAR   | 87.4% | 27.35 | 39.0% | 0.15% |
| ENA    | 87.4% | 26.96 | 38.5% | 0.28% |
| ADA    | 87.0% | 28.89 | 37.0% | 0.10% |
| SUI    | 86.9% | 30.33 | 36.5% | 0.08% |
| INJ    | 86.7% | 26.52 | 35.8% | 0.17% |
| TIA    | 86.6% | 25.90 | 35.0% | 0.13% |
| ARB    | 86.4% | 27.55 | 34.5% | 0.14% |
| OP     | 86.1% | 20.74 | 33.0% | 0.33% |
| FET    | 86.0% | 21.70 | 32.5% | 0.32% |
| WLD    | 84.2% | 17.02 | 28.0% | 0.39% |
| WIF    | 83.6% | 16.56 | 25.0% | 0.18% |
| AVAX   | 82.9% | 18.75 | 24.0% | 0.11% |
| SEI    | 81.8% | 14.00 | 22.5% | 0.17% |
| AAVE   | 81.5% | 16.19 | 22.0% | 0.22% |
| DOGE   | 80.6% | 13.72 | 21.5% | 0.41% |
| LINK   | 80.8% | 13.55 | 21.0% | 0.19% |
| SOL    | 76.9% | 12.10 | 18.0% | 0.14% |
| ETH    | 72.7% | 11.24 | 12.0% | 0.16% |
| BNB    | 70.2% | 8.13  | 10.0% | 0.09% |

All models deployed with `force_deploy` flag after successful retrain pipeline.
INJUSDT fixed from stale 0% WR deployment (old pipeline artifact).

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
python main.py
```

Or deploy the persistent daemon:
```bash
bash scripts/trading/live_trader_daemon.sh --testnet
# Runs autonomously every 5 minutes
```

A separate cron job (`scripts/operations/cron_hourly.py`) monitors live model performance
hourly with realistic thresholds (WR ≥ 60% green, < 50% urgent) and enforces a 72h
minimum retrain interval plus weekly cap of 1 retrain per symbol.

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_CAPITAL` | 5,000 USDT | Starting capital for live trading |
| `LEVERAGE_X` | 10 | Leverage (testnet; controlled by Kelly sizer) |
| `MARGIN_PCT` | 1% | Max margin per position (Kelly-scaled) |
| `SIGNAL_THRESHOLD` | 0.60 | Minimum ensemble probability to enter |
| `COOLDOWN_BARS` | 3 | Wait 15 min between trades on same symbol |
| `HOLD_BARS` | 9 | Max hold time (~45 min at 5m bars) |

---

## 📈 CI/CD Pipeline

Every push to `main` triggers:

1. 🔍 **Lint & Type Check** — black, flake8, mypy
2. 🧪 **Unit Tests** — pytest (69 tests, including database & Redis mocks)
3. 🔒 **Security Scan** — bandit
4. 📦 **Build Package** — python -m build
5. ✅ **Pre-Deploy Validation** — YAML config verification
6. 🚀 **Deploy (manual)** — staging → production (approval gates)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python 3.11 | Core runtime |
| XGBoost | Gradient-boosted ML models |
| Pandas | Data processing & feature engineering |
| SQLite | Trade & capital persistence |
| GitHub Actions | CI/CD pipeline |
| Binance Testnet SDK | Paper trading execution |
| Telegram Bot API | Automated reporting |

---

## 📋 Roadmap

- [x] ML pipeline — Multi-TF XGBoost ensemble (20 symbols, 500 models)
- [x] Live trading engine — Testnet integration with accurate PnL
- [x] Risk management — Kelly sizing + kill switches + regime filters
- [x] CI/CD — GitHub Actions pipeline
- [x] Auto-retraining — Periodic model refresh on edge decay
- [x] Telegram reporting — Automated daemon-cycle reports
- [ ] Live deployment — Mainnet with reduced sizing
- [ ] RL optimizer — DQN-based position sizing (BTC baseline: +1.36%)
- [ ] Edge decay monitoring — Rolling 30/60/90-day performance windows
- [ ] Portfolio-level correlation management — Dynamic exposure caps

---

## 🤝 License

MIT — use at your own risk. Trading futures carries significant financial risk.

---

<p align="center">
  <sub>Built with ❤️ for systematic alpha. <strong>Live to trade another day.</strong></sub>
</p>
