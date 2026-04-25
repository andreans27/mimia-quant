## 📊 Paper Trade Performance Report

**Time:** 25 Apr 2026, 06:30 UTC
**Status:** 🟢 Running — 27 runs completed over 2.3 hours

---

### 💰 Capital

| Metric | Value |
|--------|-------|
| Starting Capital | $5,000.00 |
| Current Capital | $4,995.80 |
| Realized P&L | **-$4.20 (-0.08%)** |
| Max DD | 0.08% |
| Capital Stability | ✅ Excellent |

Capital is near-flat because all 3 open positions are **still in hold** (hold_remaining=8, hold_bars=12). No trades have closed yet, so the -$4.20 reflects only entry fees.

---

### 📡 Open Positions

| Symbol | Side | Entry Price | Qty | Hold Remaining |
|--------|------|-------------|-----|----------------|
| SOL | LONG | $86.35 | 5.78 | 8 bars |
| 1000PEPE | LONG | $0.00388 | 128,625 | 8 bars |
| ARB | LONG | $0.1323 | 3,788 | 8 bars |

All entered at ~04:26 UTC. ~$500 notional each (1% margin × 10x leverage).

---

### ⚠️ Critical Findings — Flat Probability Issue

**Major concern detected:** All XGBoost model predictions are **constant** per symbol — every run produces the exact same probability for each pair.

| Symbol | Probability | Repeated N times |
|--------|------------|-----------------|
| UNI | 0.5520 | 28× identical |
| APT | 0.5448 | 28× identical |
| TIA | 0.5617 | 28× identical |
| FET | 0.5122 | 28× identical |
| OP | 0.5362 | 28× identical |
| SUI | 0.4970 | 25× identical |
| INJ | 0.4324 | 25× identical |
| SOL | 0.6284 | 13× identical |
| 1000PEPE | 0.6006 | 13× identical |
| ARB | 0.6585 | 10× identical |

**Possible root causes:**
1. **Stale features** — The parquet cache may contain frozen data; if the cache isn't refreshing between runs, the model gets the same input every time
2. **Model overfit** — XGBoost depth=3 with strong regularization could be too simple to capture changing market regimes, outputting the same prediction for all similar inputs
3. **Feature drift** — Some features may be derived from the same fixed window (e.g., 120-day lookback) so short-term changes don't move the needle

**Impact:** The system will never generate new entry/exit signals until probabilities change. Only the initial batch of entries will execute, then the system becomes inert.

---

### 📈 Signal Distribution

- **226 total signals** generated across 27 runs
- **Only 3/10 symbols** have probabilities above threshold (0.60):
  - SOL (62.8%) ✅
  - 1000PEPE (60.1%) ✅
  - ARB (65.9%) ✅
- **7/10 symbols** have probabilities below threshold (0.43–0.56)
- **0 trades closed** so far

---

### 🩺 Recommendations

**Priority — Fix the flat probability issue:**

1. **Audit feature pipeline** — Verify that `ml_cache` parquet files are being refreshed with latest market data on each run. If stale, the model sees the same inputs → same outputs.

2. **Add model prediction diversity** — Consider:
   - Re-training periodically (weekly) with latest 120 days
   - Adding online features (e.g., rolling z-score, short-term momentum that changes every bar)
   - Relaxing regularization slightly if the model is too simple

3. **Implement exit logic debugging** — 0 trades closed in 27 runs is either:
   - Positions haven't aged out yet (hold_bars=12, only ~4 bars elapsed)
   - The exit condition (`proba < exit_threshold`) never triggers because probabilities never change

4. **Until fixed, consider lower threshold** — If most symbols cluster around 0.50–0.56, a threshold of 0.55 would activate more signals (at the cost of lower win rate)

---

### ⚙️ Cron Job Update ✅

- **Changed:** Every 5 min → **Every 60 min** (next run: ~07:31 UTC)
- **Reason:** 5-minute intervals generate no new signals (probability is flat), so running every hour is sufficient for monitoring until the flat-proba issue is fixed

**Next report:** After the first hourly run at ~07:31 UTC, I'll check if probabilities change and report findings.
