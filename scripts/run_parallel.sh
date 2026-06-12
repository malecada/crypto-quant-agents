#!/usr/bin/env bash
# Run generate_agent_signals.py for multiple coins in parallel — one
# subprocess per coin, each wrapped in run_until_done.sh for crash recovery.
# Atomic CSV writes + per-coin output paths prevent collisions; the
# replay-cache SQLite is opened in WAL mode (see replay_cache.py) so
# concurrent reads/writes work.
#
# Usage:
#   scripts/run_parallel.sh \
#       --coins bitcoin ethereum binancecoin \
#       --start 2026-01-16 --end 2026-04-15 \
#       --analysts market onchain prediction crypto_sentiment \
#       --deep-think gpt-4o-mini --quick-think gpt-4o-mini \
#       --sentiment-mode pit --output-dir data/agent_signals_pit_p4
#
# All flags after --coins/before any other flag are interpreted as coin
# names. Everything else is forwarded verbatim to each child run_until_done
# call (with --coins replaced by a single coin per child).

set -u

cd "$(dirname "$0")/.."

# --- Parse out --coins values ---
coins=()
forward=()
parsing_coins=0
i=0
args=("$@")
while [ "$i" -lt "${#args[@]}" ]; do
    arg="${args[$i]}"
    if [ "$arg" = "--coins" ]; then
        parsing_coins=1
        i=$((i + 1))
        continue
    fi
    if [ "$parsing_coins" -eq 1 ]; then
        # Coins continue until the next flag
        case "$arg" in
            --*) parsing_coins=0; forward+=("$arg") ;;
            *)   coins+=("$arg") ;;
        esac
    else
        forward+=("$arg")
    fi
    i=$((i + 1))
done

if [ "${#coins[@]}" -eq 0 ]; then
    echo "[run_parallel] error: --coins is required" >&2
    exit 2
fi

echo "[run_parallel] coins  : ${coins[*]}"
echo "[run_parallel] forward: ${forward[*]}"

PIDFILE_DIR="${PIDFILE_DIR:-/tmp/agent_signals_parallel}"
mkdir -p "$PIDFILE_DIR"

pids=()
for coin in "${coins[@]}"; do
    echo "[run_parallel] launching $coin"
    nohup ./scripts/run_until_done.sh --coins "$coin" "${forward[@]}" \
        > "$PIDFILE_DIR/${coin}.log" 2>&1 &
    pid=$!
    echo "$pid" > "$PIDFILE_DIR/${coin}.pid"
    pids+=("$pid")
    echo "[run_parallel]   $coin pid=$pid log=$PIDFILE_DIR/${coin}.log"
done

echo "[run_parallel] waiting for ${#pids[@]} jobs..."
fail=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        fail=$((fail + 1))
        echo "[run_parallel] pid=$pid exited non-zero"
    fi
done

if [ "$fail" -gt 0 ]; then
    echo "[run_parallel] done with $fail failures"
    exit 1
fi
echo "[run_parallel] all coins complete"
