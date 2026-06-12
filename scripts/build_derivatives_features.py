"""Pull Binance Futures funding/OI + Coinglass liquidations + spot/perp price
for a date range, compute daily derivative features, write to
``data/derivatives/{coin}.parquet``.

Usage:
    python scripts/build_derivatives_features.py \\
        --coins bitcoin ethereum \\
        --start 2024-05-01 --end 2026-04-15
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.strategies.v3.features.derivatives import (  # noqa: E402
    build_daily_derivatives_features,
    fetch_funding_rate,
    fetch_liquidations,
    fetch_open_interest_history,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_COIN_TO_SYMBOL = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cache-dir", default="data/derivatives_raw")
    parser.add_argument("--out-dir", default="data/derivatives")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    for coin in args.coins:
        symbol = _COIN_TO_SYMBOL[coin]
        logger.info("Fetching derivatives for %s", symbol)
        try:
            funding = fetch_funding_rate(symbol=symbol, cache_dir=cache_dir, start=start, end=end)
            oi = fetch_open_interest_history(symbol=symbol, cache_dir=cache_dir, start=start, end=end)
            liq = fetch_liquidations(symbol=symbol, cache_dir=cache_dir, start=start, end=end)
        except Exception:
            logger.exception("Fetch failed for %s", symbol)
            continue

        # Spot/perp price series — use OI value / OI quantity as a proxy for perp price;
        # for spot, use Binance spot klines via existing dataflow if available.
        # For now, leave spot/perp empty — basis_annual will be NaN if not provided.
        spot_series = pd.Series(dtype="float64")
        perp_series = pd.Series(dtype="float64")
        if not oi.empty and "open_interest_value" in oi.columns:
            perp_series = (oi["open_interest_value"] / oi["open_interest"]).rename("perp_price")

        if perp_series.empty:
            logger.warning("No perp price for %s — basis_annual will be NaN", symbol)

        idx_max = max(
            funding.index.max() if not funding.empty else pd.Timestamp("1970-01-01", tz="UTC"),
            oi.index.max() if not oi.empty else pd.Timestamp("1970-01-01", tz="UTC"),
            liq.index.max() if not liq.empty else pd.Timestamp("1970-01-01", tz="UTC"),
        )
        features = build_daily_derivatives_features(
            funding_df=funding,
            oi_df=oi,
            liq_df=liq,
            spot_price_series=spot_series,
            perp_price_series=perp_series,
            as_of=idx_max,
        )

        out_file = out_dir / f"{coin}.parquet"
        features.to_parquet(out_file)
        logger.info("Wrote %s (%d rows)", out_file, len(features))


if __name__ == "__main__":
    main()
