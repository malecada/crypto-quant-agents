"""Weekly CVRF reflection cron entrypoint (Tier B1).

Reads ``data/trade_journal.db`` for the past 7 days per coin, distills
into 3-5 sentence "investment beliefs" via an LLM, and appends to
``data/beliefs/weekly.parquet``. Idempotent on ``(week_end, coin)``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.agents.cvrf_reflector import CVRFReflector  # noqa: E402
from tradingagents.dataflows.beliefs_store import upsert_beliefs  # noqa: E402
from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402
from tradingagents.llm_clients.factory import LLMClientFactory  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--week-end",
        required=True,
        help="YYYY-MM-DD reflection date (the week ending on this date)",
    )
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--llm-model", default=None, help="Override config quick_think_llm")
    args = p.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.llm_model:
        cfg["quick_think_llm"] = args.llm_model
    cfg["replay_cache"] = True  # reflections are deterministic on rerun

    factory = LLMClientFactory(cfg)
    llm = factory.create_llm("quick")
    reflector = CVRFReflector(llm)

    week_end = pd.to_datetime(args.week_end, utc=True)
    rows = []
    for coin in args.coins:
        row = reflector.reflect_week(coin, week_end)
        if row is None:
            continue
        rows.append(row)
        logger.info(f"{coin} belief: {row['belief_text'][:120]}...")

    n = upsert_beliefs(rows)
    logger.info(f"upserted {n} belief rows for week ending {week_end.date()}")


if __name__ == "__main__":
    main()
