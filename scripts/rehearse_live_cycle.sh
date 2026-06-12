#!/usr/bin/env bash
set -euo pipefail

# Run 7 dry-run cycles back-to-back locally to validate the pipeline end-to-end.

export DATA_DIR=$(mktemp -d)
export LOG_DIR=$(mktemp -d)
export LIVE_MODE=false
# Pass through credentials if present, else error early.
: "${BINANCE_API_KEY:?BINANCE_API_KEY must be set}"
: "${BINANCE_API_SECRET:?BINANCE_API_SECRET must be set}"
: "${TELEGRAM_BOT_TOKEN:=}"  # optional
: "${TELEGRAM_CHAT_ID:=}"    # optional

for i in 1 2 3 4 5 6 7; do
    echo "── rehearsal cycle $i ──"
    python -m tradingagents.execution.live.runner --once --dry-run --cycle-id "rehearse-$i"
done

echo
echo "Journal contents:"
sqlite3 "$DATA_DIR/trade_journal.db" "SELECT cycle_id, status FROM cycles;"
sqlite3 "$DATA_DIR/trade_journal.db" "SELECT cycle_id, COUNT(*) FROM trades GROUP BY cycle_id;"
sqlite3 "$DATA_DIR/trade_journal.db" "SELECT cycle_id, AVG(agree) FROM shadow_decisions GROUP BY cycle_id;"

echo
echo "Cleanup: rm -rf $DATA_DIR $LOG_DIR"

# === V5 invariant assertions (run once after the 7 cycles) ===
set -e

DB="${DB:-$DATA_DIR/trade_journal.db}"
echo "=== V5 invariants ==="

# 1. After cycle 1, composite must have 4 routes
ROUTE_COUNT=$(sqlite3 "$DB" "SELECT routes FROM retrains ORDER BY cycle_id LIMIT 1" \
                | python -c "import sys, json; print(len(json.loads(sys.stdin.read())))")
if [ "$ROUTE_COUNT" -ne 4 ]; then
    echo "FAIL: cycle 1 composite has $ROUTE_COUNT routes, expected 4"
    exit 1
fi

# 2. Per cycle: 4 coins × 2 horizons = 8 predictions
for cycle in $(sqlite3 "$DB" "SELECT DISTINCT cycle_id FROM predictions ORDER BY cycle_id"); do
    n=$(sqlite3 "$DB" "SELECT COUNT(*) FROM predictions WHERE cycle_id='$cycle' AND bundle_route IS NOT NULL")
    if [ "$n" -ne 8 ]; then
        echo "FAIL: cycle $cycle has $n V5 predictions (bundle_route NOT NULL), expected 8"
        exit 1
    fi
done

# 3. Atomicity: every retrain row's routes JSON has exactly 4 entries
for cycle in $(sqlite3 "$DB" "SELECT cycle_id FROM retrains"); do
    n=$(sqlite3 "$DB" "SELECT routes FROM retrains WHERE cycle_id='$cycle'" \
          | python -c "import sys, json; print(len(json.loads(sys.stdin.read())))")
    if [ "$n" -ne 4 ]; then
        echo "FAIL: retrain $cycle has $n routes, expected 4"
        exit 1
    fi
done

echo "V5 invariants PASS — 4 routes per composite, 8 preds per cycle, atomic."
