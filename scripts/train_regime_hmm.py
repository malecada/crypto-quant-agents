"""Train per-coin 3-state Gaussian HMM regime detectors.

Fits on price-only features (log_return, realized_vol, abs_return) up to
``--through`` and pickles the bundle to
``data/checkpoints/regime_hmm_{coin}.pkl``.

Walk-forward fitting is NOT done here — Phase 1 commits to a single
training window per coin (the plan defers walk-forward HMM to V1
placebo). If the regime label flips more than ~2x/month on the inspect
output, retune ``vol_lookback`` or feature set before Phase 4.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Make tradingagents importable when run from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.regime import (  # noqa: E402
    FittedHMM,
    assign_labels,
    build_regime_features,
    smooth_label_sequence,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fit_and_save(coin: str, through: str, out_dir: str, n_iter: int = 200) -> str:
    df = _load_crypto_ohlcv(coin, through)
    df = df[df["Date"] <= pd.to_datetime(through).tz_localize(None)]
    if len(df) < 200:
        raise RuntimeError(f"insufficient OHLCV history for {coin}: {len(df)} bars")
    prices = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    feats = build_regime_features(prices)
    feature_names = list(feats.columns)
    X = feats.values
    logger.info(f"{coin}: fitting HMM on {len(X)} samples × {X.shape[1]} features")
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42,
        tol=1e-4,
    )
    model.fit(X)
    state_to_label = assign_labels(model, feats)
    bundle = FittedHMM(
        model=model, state_to_label=state_to_label, feature_names=feature_names
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"regime_hmm_{coin}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(bundle, f)
    states = model.predict(X)
    raw_labels = np.array([state_to_label[s] for s in states])
    smoothed = smooth_label_sequence(raw_labels, window=3)
    raw = pd.Series(raw_labels, index=feats.index)
    smooth = pd.Series(smoothed, index=feats.index)
    raw_flips = (raw != raw.shift(1)).sum()
    smooth_flips = (smooth != smooth.shift(1)).sum()
    months = (feats.index[-1] - feats.index[0]).days / 30.0 or 1.0
    logger.info(
        f"{coin}: saved {out_path}; state→label={state_to_label}; "
        f"raw_flips/month={raw_flips / months:.2f}; "
        f"smoothed_flips/month={smooth_flips / months:.2f}; "
        f"smoothed_dist={smooth.value_counts().to_dict()}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--through", required=True, help="YYYY-MM-DD upper bound")
    p.add_argument("--out-dir", default="data/checkpoints")
    p.add_argument("--n-iter", type=int, default=200)
    args = p.parse_args()

    for coin in args.coins:
        try:
            fit_and_save(coin, args.through, args.out_dir, args.n_iter)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"failed for {coin}: {exc}")


if __name__ == "__main__":
    main()
