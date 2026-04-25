#!/usr/bin/env bash
# ============================================================================
# Mimia Live Trader Daemon
# Runs live_trader.py every 5 minutes in a persistent loop (not a cron job).
# Logs to data/live_trader_daemon.log
# Use: --testnet (default) or --mainnet (⚠️ real funds!)
# ============================================================================

set -euo pipefail

cd /root/projects/mimia-quant
source venv/bin/activate
export PYTHONPATH="/root/projects/mimia-quant:${PYTHONPATH:-}"

LOG_FILE="data/live_trader_daemon.log"
PID_FILE="data/live_trader_daemon.pid"
INTERVAL=300  # 5 minutes in seconds

# Default mode: testnet. Pass --mainnet as first arg to use mainnet.
MODE="${1:---testnet}"

# Write PID
echo $$ > "$PID_FILE"

# Trap for graceful shutdown
cleanup() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daemon stopping..." | tee -a "$LOG_FILE"
    rm -f "$PID_FILE"
    exit 0
}
trap cleanup SIGTERM SIGINT

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Daemon started (PID: $$) — Mode: $MODE" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Interval: ${INTERVAL}s (5min)" | tee -a "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log: ${LOG_FILE}" | tee -a "$LOG_FILE"

RUN_COUNT=0
while true; do
    RUN_COUNT=$((RUN_COUNT + 1))
    START_TS=$(date +%s)

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Run #${RUN_COUNT} (${MODE}) ===" | tee -a "$LOG_FILE"

    python main.py ${MODE} >> "$LOG_FILE" 2>&1 || true

    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run #${RUN_COUNT} completed (${ELAPSED}s)" | tee -a "$LOG_FILE"

    # Sleep the remaining time to maintain 5-min interval
    SLEEP_TIME=$((INTERVAL - ELAPSED))
    if [ "$SLEEP_TIME" -gt 0 ]; then
        sleep "$SLEEP_TIME"
    else
        echo "[WARN] Run took ${ELAPSED}s — longer than 5min interval!" | tee -a "$LOG_FILE"
        # Still sleep at least 10s to avoid busy-loop
        sleep 10
    fi
done
