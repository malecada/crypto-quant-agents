"""Retrain V3 multi-horizon ensembles using real VPIN microstructure features.

Step 4 of the Binance Vision VPIN rerun experiment.

Usage:
    python scripts/train_v3_vision.py \\
        --coins bitcoin ethereum \\
        --microstructure-dir data/microstructure_vpin \\
        --out-suffix _vision \\
        --train-end 2025-12-31
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v3.models.multi_horizon import MultiHorizonEnsemble  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    parser.add_argument("--microstructure-dir", default="data/microstructure_vpin")
    parser.add_argument("--derivatives-dir", default="data/derivatives")
    parser.add_argument("--ckpt-dir", default="data/checkpoints")
    parser.add_argument("--out-suffix", default="_vision",
                        help="Suffix appended to checkpoint filename: v3_models{suffix}_{coin}.pkl")
    parser.add_argument("--train-end", default="2025-12-31")
    args = parser.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    micro_dir = Path(args.microstructure_dir)
    deriv_dir = Path(args.derivatives_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_end = pd.Timestamp(args.train_end)
    logger.info("Training V3 models with real VPIN, train window → %s", args.train_end)
    logger.info("Microstructure dir: %s", micro_dir)

    for coin in args.coins:
        logger.info("=== %s ===", coin)
        try:
            # Load OHLCV
            df_raw = _load_crypto_ohlcv(coingecko_id=coin, curr_date="2026-04-15")
            df_raw["Date"] = pd.to_datetime(df_raw["Date"])
            df_raw = df_raw.set_index("Date").sort_index()
            df_raw.columns = [c.lower() for c in df_raw.columns]
            prices = df_raw["close"]
            returns = prices.pct_change().fillna(0.0)

            # Normalize timezone
            if prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
                returns.index = returns.index.tz_localize(None)

            # Load real VPIN microstructure
            micro_file = micro_dir / f"{coin}.parquet"
            if not micro_file.exists():
                logger.error("Missing microstructure: %s — skipping", micro_file)
                continue
            micro = pd.read_parquet(micro_file)
            if micro.index.tz is not None:
                micro.index = micro.index.tz_localize(None)
            logger.info(
                "%s micro rows=%d, cols=%s, date_range=[%s, %s]",
                coin, len(micro), list(micro.columns),
                micro.index.min().date(), micro.index.max().date(),
            )

            # Load derivatives (optional)
            deriv_file = deriv_dir / f"{coin}.parquet"
            deriv = pd.DataFrame()
            if deriv_file.exists():
                deriv = pd.read_parquet(deriv_file)
                if deriv.index.tz is not None:
                    deriv.index = deriv.index.tz_localize(None)
                logger.info("%s deriv rows=%d, cols=%s", coin, len(deriv), list(deriv.columns))

            # Build feature matrix
            features = pd.DataFrame(index=prices.index)
            features["ret_1d"] = prices.pct_change().fillna(0.0)
            features["ret_5d"] = prices.pct_change(5).fillna(0.0)
            features["vol_5d"] = prices.pct_change().rolling(5).std().fillna(0.0)
            features["vol_21d"] = prices.pct_change().rolling(21).std().fillna(0.0)

            for col in micro.columns:
                features[col] = micro[col].reindex(features.index).fillna(0.0)

            if not deriv.empty:
                for col in deriv.columns:
                    features[col] = deriv[col].reindex(features.index).fillna(0.0)

            features = features.fillna(0.0)

            # Train on data up to train_end
            train_mask = features.index <= train_end
            train_feats = features.loc[train_mask]
            train_rets = returns.loc[train_mask]
            logger.info(
                "%s: train rows=%d, non-zero VPIN rows=%d, features=%s",
                coin, len(train_feats),
                int((features.loc[train_mask, "vpin_50"] != 0.0).sum())
                if "vpin_50" in features.columns else 0,
                list(train_feats.columns),
            )

            # Fit MultiHorizonEnsemble (LGB only — xgb+catboost adds noise per BT11)
            mhe = MultiHorizonEnsemble(horizons=(3, 7, 14, 21), holdout_fraction=0.20)
            mhe.fit(train_feats, train_rets, members=("lgb",))

            out_file = ckpt_dir / f"v3_models{args.out_suffix}_{coin}.pkl"
            with open(out_file, "wb") as f:
                pickle.dump(mhe, f)
            logger.info("Saved %s", out_file)

        except Exception:
            logger.exception("Failed %s", coin)
            continue

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
