"""Train per-coin NH-HMM bundles for V3.

Usage:
    python scripts/v3_train_regime.py --coin bitcoin --through 2024-12-31

Pickles to data/checkpoints/regime_hmm_v3_{coin}.pkl.

Phase-4 v0 note: produces a degenerate NH-HMM (zero covariate coefficients)
that reduces to a standard GaussianHMM. The bundle is structurally compatible
with the full NH-HMM interface so future L-BFGS coefficient training can slot
in without breaking consumers.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train per-coin NH-HMM bundle (V3 regime detector)."
    )
    parser.add_argument("--coin", required=True, help="CoinGecko coin id, e.g. bitcoin")
    parser.add_argument("--through", required=True, help="ISO date YYYY-MM-DD (train through this date)")
    parser.add_argument("--out-dir", default="data/checkpoints", help="Output directory for pickle")
    parser.add_argument("--n-iter", type=int, default=200, help="Max EM iterations (default 200)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pull OHLCV via existing dataflow
    from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv

    ohlcv = _load_crypto_ohlcv(coingecko_id=args.coin, curr_date=args.through)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    ohlcv = ohlcv[ohlcv["Date"] <= pd.Timestamp(args.through)].copy()
    ohlcv = ohlcv.set_index("Date").sort_index()

    if len(ohlcv) < 100:
        raise SystemExit(f"Not enough OHLCV rows for {args.coin}: {len(ohlcv)}")

    logger.info("Training NH-HMM for %s through %s (%d bars)", args.coin, args.through, len(ohlcv))

    bundle = train_nh_hmm(
        prices=ohlcv["Close"],
        covariates_df=None,
        n_states=3,
        n_iter=args.n_iter,
    )

    out_file = out_dir / f"regime_hmm_v3_{args.coin}.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(bundle, f)
    logger.info(
        "Wrote %s (%d states, %d features)",
        out_file,
        bundle.n_states,
        len(bundle.feature_names),
    )


if __name__ == "__main__":
    main()
