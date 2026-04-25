# Mimia Quant — Full Project Audit Report

**Date:** 2025-04-25  
**Auditor:** Hermes Agent  
**Scope:** `/root/projects/mimia-quant/` — complete file inventory and code smell analysis

---

## 1. FILE INVENTORY BY CATEGORY

### ✅ KEEP — Actively Used Files

| File | Size | Reason |
|------|------|--------|
| `scripts/live_trader.py` | 1,200 lines | Active trading engine |
| `scripts/live_trader_daemon.sh` | 2,552 chars | Active daemon script |
| `scripts/run_bot.py` | — | Referenced in Makefile (`make run`) |
| `scripts/run_backtest.py` | — | Referenced in Makefile (`make backtest`) |
| `scripts/init_db.py` | — | Referenced in Makefile (`make init-db`) |
| `scripts/oos_validation.py` | — | Has compiled .pyc in `__pycache__` |
| `src/` (all 7 subpackages) | ~290KB total | Core library: `core/`, `strategies/`, `execution/`, `monitoring/`, `utils/` |
| `config/config.yaml` | 1,319 chars | Main system configuration |
| `configs/strategies.yaml` | 1,790 chars | Strategy parameter configuration |
| `tests/` | — | 69 passing tests |
| `pyproject.toml` | — | Build configuration |
| `requirements.txt` | — | Python dependencies |
| `Makefile` | — | Build/test/run targets |
| `.gitignore` | 2,362 chars | Git exclusion rules |
| `README.md` | — | Project documentation (needs update) |
| `data/live_trading.db` | 36KB | Active live trading database |
| `data/backtest_results/` (3 files) | 1.7KB | Backtest result JSON files |
| `data/ml_cache/` (22 parquet + 3 pkl + 1 json) | ~784MB | ML feature cache (precomputed) |
| `data/rl_models/BTCUSDT_dqn.pt` | 94KB | RL model checkpoint |
| `data/market_data/` | — | Empty, ready for use |
| `.env` | — | Environment variables (API keys) |

### 🗑️ DELETE — Clearly Obsolete / Redundant

| File | Size | Reason |
|------|------|--------|
| `debug_breakout.py` | ~616 chars | One-off debug script, not part of system |
| `debug_momentum.py` | ~592 chars | One-off debug script |
| `debug_momentum2.py` | ~610 chars | One-off debug script |
| `debug_sizing.py` | ~680 chars | One-off debug script |
| `debug_sizing2.py` | ~776 chars | One-off debug script |
| `debug_sizing3.py` | ~765 chars | One-off debug script |
| `debug_trace.py` | ~650 chars | One-off debug script |
| `debug_verify_fix.py` | ~503 chars | One-off debug script |
| `scripts/paper_trader.py` | 1,185 lines | **Replaced by `live_trader.py`** — no imports reference it |
| `scripts/paper_trader_daemon.sh` | 2,275 chars | **Replaced by `live_trader_daemon.sh`** — no usage |
| `data/paper_trading.db` | 48KB | **Stale** — replaced by `live_trading.db` |
| `reports/paper_trade_performance_2025-04-25.md` | 3,888 chars | **Stale** — paper trade report, no longer relevant |
| `scripts/__pycache__/_test_binance_order2.cpython-311.pyc` | 4.9KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/_test_full_integration.cpython-311.pyc` | 3.1KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/compare_exit_strategies.cpython-311.pyc` | 25.7KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/threshold_scan.cpython-311.pyc` | 21.7KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/train_ml_ensemble.cpython-311.pyc` | 17.3KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/train_tf_specific.cpython-311.pyc` | 20.6KB | **Orphaned** — no corresponding `.py` source file |
| `scripts/__pycache__/ml_voting_backtest.cpython-311.pyc` | 28.2KB | **Orphaned** — no corresponding `.py` source file |

**Total DELETE candidates: 20 files** (~1.4MB reclaimable, excluding the 784MB ml_cache which is KEEP)

### ❓ MAYBE — Needs User Review

| File | Size | Reason |
|------|------|--------|
| `data/mimia_quant.db` | 580KB | Shared DB — could still hold historical/backtest data. Check if live_trader.py uses it or just paper_trading.db |
| `dist/mimia_quant-1.0.0-py3-none-any.whl` | 108KB | Build artifact (can be regenerated via `python3 -m build`) |
| `dist/mimia_quant-1.0.0.tar.gz` | 97KB | Build artifact (can be regenerated via `python3 -m build`) |
| `src/core/redis_client.py` | 16KB | Redis config defined in config.yaml, but actual trading uses SQLite. Dead code or future? |

---

## 2. CODE SMELLS

### 🐛 Stale Makefile References
**`Makefile` references scripts that DO NOT EXIST:**
- `make run` → `python3 scripts/run_bot.py` ✅ exists
- `make backtest` → `python3 scripts/run_backtest.py` ✅ exists
- `make report` → `python3 scripts/run_monitoring.py` ❌ **MISSING**
- `make cron-setup` → `python3 scripts/setup_cron.py` ❌ **MISSING**

### 📄 Stale Documentation
- **`README.md`** still references `paper_trader.py` — should be updated to `live_trader.py`

### 🗂️ Dual Config Directories
- Both `config/` and `configs/` exist — inconsistent naming. `config/config.yaml` is the main config, `configs/strategies.yaml` holds strategy params. Could be merged.

### 🔧 Misleading Database Config
- `config.yaml` section `database:` and `redis:` describe Redis connection params and `host: localhost, port: 6379`
- **But** actual trading code uses **SQLite** (`data/live_trading.db`), not Redis
- `redis_client.py` exists in `src/core/` (used for caching/pub-sub) but the `database:` config section is confusing/misleading

### 🧹 .gitignore Already Excludes debug_*.py
- Line 18: `debug_*.py` is already in `.gitignore` — so these 8 debug files **should never have been tracked**. If they're present, they need `git rm --cached`.

### 🔁 Duplicate Script Pattern
- `paper_trader.py` (1,185 lines) and `live_trader.py` (1,200 lines) are structurally parallel — paper_trader.py has been fully superseded but not removed

### 🧪 Orphaned .pyc Artifacts
- 7 `.pyc` files in `scripts/__pycache__/` have **no corresponding `.py` source file** in `scripts/`. These are stale compiled bytecode from scripts that were deleted but left behind.

---

## 3. SUMMARY

| Metric | Value |
|--------|-------|
| **Total files audited** | ~85 files (project code) |
| **🗑️ DELETE (obsolete)** | 20 files (8 debug scripts, 2 replaced scripts, 1 stale DB, 1 stale report, 7 orphaned .pyc, 1 stale daemon sh) |
| **❓ MAYBE (needs review)** | 4 files (1 DB, 2 build artifacts, 1 Redis module) |
| **🐛 Code smells found** | 7 issues (2 missing Makefile targets, stale docs, dual config dirs, misleading config, duplicate scripts, orphaned pycs) |
| **💾 Disk reclaimable** | ~1.4 MB (excluding 784 MB ml_cache which is KEEP) |
