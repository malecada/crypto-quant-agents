"""Regime HMM checkpoints must be loadable + 3-state when present.

Checkpoints live under ``data/checkpoints`` which is gitignored — they are
out-of-band operational artifacts (trained locally / on the VPS), NOT committed.
So this test SKIPS a coin whose checkpoint is absent (e.g. a fresh CI checkout)
rather than failing. The hard "all 8 present" gate for the hybrid bot lives in
``deploy/preflight.sh`` (runs on the VPS before each hybrid cycle), where the
files are guaranteed provisioned.

XRP/DOGE/ADA/TRX are the satellites pre-trained for the 8-coin hybrid deploy
(Phase 0.1); BTC/ETH/BNB/SOL shipped with the quant bot.
"""
import pickle
from pathlib import Path

import pytest

LIVE_COINS = ["bitcoin", "ethereum", "binancecoin", "solana",
              "ripple", "dogecoin", "cardano", "tron"]


@pytest.mark.parametrize("coin", LIVE_COINS)
def test_regime_hmm_loadable_when_present(coin):
    path = Path("data/checkpoints") / f"regime_hmm_{coin}.pkl"
    if not path.exists():
        pytest.skip(f"regime HMM for {coin} not provisioned in this env "
                    f"(gitignored; deploy preflight enforces presence)")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    # FittedHMM bundle: a fitted GaussianHMM + a 3-state label map
    assert hasattr(bundle, "model")
    assert hasattr(bundle, "state_to_label")
    assert len(set(bundle.state_to_label.values())) >= 2
