"""Daily cron entrypoint to recompute per-coin rolling LLM edge.

Reads from ``data/trade_journal.db`` and writes
``data/rolling_edge.parquet``. Idempotent on ``(coin, as_of_date)``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.strategies.rolling_edge import update_rolling_edge  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        default=None,
        help="YYYY-MM-DD reference date. Default: today (UTC).",
    )
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--min-trades", type=int, default=10)
    args = p.parse_args()

    as_of = pd.Timestamp.utcnow() if args.as_of is None else pd.to_datetime(args.as_of, utc=True)
    n = update_rolling_edge(
        as_of_date=as_of,
        window_days=args.window_days,
        min_trades=args.min_trades,
    )
    logger.info(f"rolling_edge: wrote {n} (coin, as_of) rows")


if __name__ == "__main__":
    main()
