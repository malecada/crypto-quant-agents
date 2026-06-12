#!/usr/bin/env bash
# Invoke generate_agent_signals.py in a loop, restarting on any non-zero exit
# (transient OpenAI outage, connection drop, crash, anything). The runner is
# idempotent — resumes from partial CSVs and refills ERROR rows. Safe to run
# for hours unattended.
#
# Usage:
#   scripts/run_until_done.sh --coins bitcoin ethereum --start 2026-01-16 \
#       --end 2026-04-15 --analysts market onchain prediction crypto_sentiment \
#       --deep-think gpt-4o-mini --quick-think gpt-4o-mini \
#       --sentiment-mode pit --output-dir data/agent_signals_pit_p3
#
# Every arg after the script name is forwarded verbatim to the underlying
# generate_agent_signals.py.

set -u

PYBIN="${PYBIN:-/usr/bin/python3}"
MAX_RESTARTS="${MAX_RESTARTS:-50}"
RESTART_SLEEP="${RESTART_SLEEP:-15}"
# Watchdog: if log file mtime hasn't advanced in N seconds, gen is hung
# (typically a blocked socket inside a httpx/openai retry loop that
# can't be killed by Python's threading timeout). Kill it; the outer
# loop then restarts and resumes from the last checkpoint.
WATCHDOG_STALL_SEC="${WATCHDOG_STALL_SEC:-900}"  # 15 min
WATCHDOG_POLL_SEC="${WATCHDOG_POLL_SEC:-60}"

cd "$(dirname "$0")/.."
LOG_DIR="${LOG_DIR:-/tmp/agent_signals_gen}"
mkdir -p "$LOG_DIR"

# Watchdog: monitors the gen log; if mtime is stale, escalates SIGTERM → SIGKILL.
watchdog() {
    local pid="$1"
    local logfile="$2"
    while kill -0 "$pid" 2>/dev/null; do
        sleep "$WATCHDOG_POLL_SEC"
        if ! kill -0 "$pid" 2>/dev/null; then return; fi
        if [ -f "$logfile" ]; then
            local now mtime age
            now=$(date +%s)
            mtime=$(stat -c %Y "$logfile" 2>/dev/null || echo "$now")
            age=$((now - mtime))
            if [ "$age" -gt "$WATCHDOG_STALL_SEC" ]; then
                echo "[watchdog] log stale ${age}s > ${WATCHDOG_STALL_SEC}s — killing gen pid=$pid" >&2
                kill -TERM "$pid" 2>/dev/null
                sleep 10
                kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
                return
            fi
        fi
    done
}

attempt=0
while [ "$attempt" -lt "$MAX_RESTARTS" ]; do
    attempt=$((attempt + 1))
    ts=$(date -u +"%Y%m%dT%H%M%SZ")
    log="$LOG_DIR/run_${ts}_attempt${attempt}.log"
    echo "[run_until_done] attempt=${attempt} log=${log}"
    # Start gen in background so we can attach a watchdog
    $PYBIN scripts/generate_agent_signals.py "$@" >"$log" 2>&1 &
    gen_pid=$!
    watchdog "$gen_pid" "$log" &
    wd_pid=$!
    wait "$gen_pid"
    rc=$?
    # Watchdog auto-exits when gen dies, but make sure
    kill "$wd_pid" 2>/dev/null
    if [ "$rc" -eq 0 ]; then
        echo "[run_until_done] generate_agent_signals exited 0 — done"
        exit 0
    fi
    echo "[run_until_done] exit code ${rc} — sleeping ${RESTART_SLEEP}s before retry"
    sleep "$RESTART_SLEEP"
done
echo "[run_until_done] gave up after ${MAX_RESTARTS} attempts"
exit 1
