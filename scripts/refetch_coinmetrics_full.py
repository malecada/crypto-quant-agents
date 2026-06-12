"""Refetch CoinMetrics Community metrics into the bitemporal on-chain store.

Driven by ``tradingagents.dataflows.coinmetrics.SUPPORTED`` — adds any new
metrics introduced there to the existing parquet shards. Existing rows are
de-duped on (event_ts, coin, metric, source, as_of_ts).

Usage:
    python scripts/refetch_coinmetrics_full.py --coins btc eth --since 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.onchain import fetch_coinmetrics_incremental  # noqa: E402
from tradingagents.dataflows.onchain_store import upsert_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", nargs="+", default=["btc", "eth"])
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--root", default="data/onchain")
    args = p.parse_args()

    logger.info("Fetching CM Community for %s since %s", args.coins, args.since)
    df = fetch_coinmetrics_incremental(args.coins, args.since)
    logger.info("Fetched %d rows", len(df))
    if df.empty:
        logger.warning("No rows returned — exiting")
        return

    metric_counts = df.groupby(["coin", "metric"]).size().sort_index()
    logger.info("Rows per (coin, metric):\n%s", metric_counts.to_string())

    written = upsert_rows(df, root=Path(args.root))
    logger.info("Upserted; total rows in touched shards now: %d", written)


if __name__ == "__main__":
    main()
