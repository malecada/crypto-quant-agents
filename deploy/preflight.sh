#!/usr/bin/env bash
set -euo pipefail

# Disk free check (>10% free on /opt)
disk_pct=$(df --output=pcent /opt | tail -1 | tr -d ' %')
if [ "$disk_pct" -gt 90 ]; then
    echo "preflight: disk usage $disk_pct% > 90% — aborting" >&2
    exit 1
fi

# Network reachability
if ! curl -sSf --max-time 5 https://api.binance.com/api/v3/ping >/dev/null; then
    echo "preflight: cannot reach Binance — aborting" >&2
    exit 1
fi

# Secrets file present + locked
secrets="/opt/tradingagents/secrets/.env.trading"
if [ ! -f "$secrets" ]; then
    echo "preflight: secrets file missing — aborting" >&2
    exit 1
fi
mode=$(stat -c "%a" "$secrets")
if [ "$mode" != "600" ]; then
    echo "preflight: secrets file mode $mode (expected 600) — aborting" >&2
    exit 1
fi

# === V5 preflight additions ===
set -e

echo "=== V5 preflight ==="

# 1. COINGLASS_API_KEY present
if [ -z "${COINGLASS_API_KEY:-}" ]; then
    echo "FAIL: COINGLASS_API_KEY not set"
    exit 1
fi
echo "  COINGLASS_API_KEY: set"

# 2. Coin universe sane size (1..8). Correctness — every coin has a routing
# entry — is enforced in the python import check below (step 5), which is the
# genuinely critical condition and supports both the 4-coin and 8-coin universes.
N_COINS=$(echo "${COIN_UNIVERSE:-bitcoin,ethereum,binancecoin,solana,ripple,dogecoin,cardano,tron}" | tr ',' '\n' | grep -c .)
if [ "$N_COINS" -lt 1 ] || [ "$N_COINS" -gt 8 ]; then
    echo "FAIL: COIN_UNIVERSE size $N_COINS out of [1, 8]"
    exit 1
fi
echo "  COIN_UNIVERSE: $N_COINS coins"

# 3. Kelly is set + reasonable (0.10 to 0.29 band)
KELLY="${KELLY_FRACTION:-0.25}"
case "$KELLY" in
    0.[12][0-9]|0.[12]) ;;
    *) echo "FAIL: KELLY_FRACTION=$KELLY out of [0.10, 0.29] band"; exit 1 ;;
esac
echo "  KELLY_FRACTION: $KELLY"

# 3b. Signal config must match the canonical backtest (P2/P3 parity). The
# published SR +3.18 run used confidence_ref=0.05 + asymmetric (SYMMETRIC=false);
# any other value silently trades a different signal/size than was validated.
CONF_REF="${CONFIDENCE_REF_RETURN:-0.05}"
if [ "$CONF_REF" != "0.05" ]; then
    echo "FAIL: CONFIDENCE_REF_RETURN=$CONF_REF != canonical 0.05 (backtest parity)"
    exit 1
fi
echo "  CONFIDENCE_REF_RETURN: $CONF_REF"
SYM_LC=$(echo "${SYMMETRIC:-false}" | tr '[:upper:]' '[:lower:]')
case "$SYM_LC" in
    false|0|no) ;;
    *) echo "FAIL: SYMMETRIC=$SYM_LC must be false (canonical V5 MIX is asymmetric)"; exit 1 ;;
esac
echo "  SYMMETRIC: $SYM_LC"

# 4. Derivatives + options dirs writable
# P5: mirror config.load_config precedence — explicit root, else DATA_DIR.
DATA_ROOT="${TRADINGAGENTS_DATA_ROOT:-${DATA_DIR:-/opt/tradingagents/data}}"
for sub in derivatives derivatives_raw options onchain cache; do
    DIR="$DATA_ROOT/$sub"
    if [ ! -d "$DIR" ]; then
        mkdir -p "$DIR" || { echo "FAIL: cannot create $DIR"; exit 1; }
    fi
    if [ ! -w "$DIR" ]; then
        echo "FAIL: $DIR not writable"
        exit 1
    fi
done
echo "  data subdirs: writable"

# 5. Can import V5 live modules
# Use $PYTHON if set (deploy passes /opt/tradingagents/venv/bin/python via systemd
# Environment=), else fall back to bare `python`. Service user has no venv on PATH
# unless explicitly added.
PYTHON_BIN="${PYTHON:-/opt/tradingagents/venv/bin/python}"
[ -x "$PYTHON_BIN" ] || PYTHON_BIN="python"
"$PYTHON_BIN" -c "
import os
from tradingagents.execution.live.config import LiveConfig, _V5_DEFAULT_ROUTING
from tradingagents.execution.live.data_refresh import refresh_all, CriticalDataRefreshError
from tradingagents.execution.live.retrain import run_retrain_with_fallback
from tradingagents.execution.live.predict import run_predict, PredictMajorityFail
# Routing correctness: every COIN_UNIVERSE coin must have a routing entry,
# else predict.py KeyErrors mid-cycle. This is the genuinely critical universe
# check (replaces the old fixed coin-count), and supports 4 or 8 coins.
_u = [c.strip() for c in os.environ.get(
    'COIN_UNIVERSE',
    'bitcoin,ethereum,binancecoin,solana,ripple,dogecoin,cardano,tron',
).split(',') if c.strip()]
_missing = [c for c in _u if c not in _V5_DEFAULT_ROUTING]
assert not _missing, f'COIN_UNIVERSE coins missing routing: {_missing}'
print('  V5 imports + routing: OK')
" || { echo "FAIL: V5 import / routing error"; exit 1; }

# 6. Sample Coinglass auth — SUPPLEMENTARY (PF1). Coinglass is not in
# data_refresh_critical {ohlcv, coinmetrics}; the runtime tiers it for graceful
# degradation. A Coinglass outage must NOT abort a full trading day, so warn
# rather than exit. (The key-present check at step 1 stays hard-fail.)
if curl -s --max-time 8 -H "CG-API-KEY: $COINGLASS_API_KEY" \
    "https://open-api-v4.coinglass.com/api/futures/supported-coins" \
    | grep -q '"code":"0"'; then
    echo "  Coinglass auth: OK"
else
    echo "  WARN: Coinglass auth check failed — supplementary source, runtime will tier it; continuing." >&2
fi

echo "V5 preflight: ALL OK"

# --- hybrid preflight (only when HYBRID_DATA_DIR is set) ---
if [ -n "${HYBRID_DATA_DIR:-}" ]; then
  : "${HYBRID_BINANCE_API_KEY:?HYBRID_BINANCE_API_KEY missing}"
  : "${HYBRID_BINANCE_API_SECRET:?HYBRID_BINANCE_API_SECRET missing}"
  : "${OPENAI_API_KEY:?OPENAI_API_KEY missing for modulator}"
  echo "  hybrid secrets: present"
  # All 8 live-coin regime HMMs must be provisioned. Checkpoints are gitignored
  # out-of-band artifacts (trained on / scp'd to the VPS), so guard their
  # presence here before the hybrid cycle relies on them.
  ckpt_dir="${CHECKPOINT_DIR:-data/checkpoints}"
  for c in bitcoin ethereum binancecoin solana ripple dogecoin cardano tron; do
    [ -f "$ckpt_dir/regime_hmm_$c.pkl" ] || {
      echo "MISSING regime HMM: $ckpt_dir/regime_hmm_$c.pkl" >&2
      echo "  provision via: python scripts/train_regime_hmm.py --coins $c --through <date>" >&2
      exit 1
    }
  done
  echo "  hybrid regime HMMs: 8/8 present"
fi
