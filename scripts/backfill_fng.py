#!/usr/bin/env python
"""Backfill alternative.me Fear & Greed Index into the bitemporal sentiment store.

Unlike Alpaca News (rate-limited paginated), F&G is a single endpoint that
returns the full history. One call => ~3000 daily rows. Safe to re-run.

Usage:
    python scripts/backfill_fng.py
    python scripts/backfill_fng.py --limit 0 --out-dir data/sentiment/fng

Notes on PIT:
- `event_ts` is the UTC midnight timestamp for the F&G value's publication day
- `as_of_ts` is set to event_ts + 1h as a conservative lag (published on the day)
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows import fng_store  # noqa: E402

FNG_URL = "https://api.alternative.me/fng/"
INGEST_LAG = timedelta(hours=1)

log = logging.getLogger("backfill_fng")


def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill alternative.me Fear & Greed index.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0,
                   help="Number of rows (0 = full history).")
    p.add_argument("--out-dir", default="data/sentiment/fng")
    return p.parse_args()


def fetch_all(limit: int = 0) -> list[dict]:
    r = requests.get(FNG_URL, params={"limit": limit}, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def normalize(item: dict) -> dict:
    # API returns unix timestamp seconds as string
    ts = datetime.fromtimestamp(int(item["timestamp"]), tz=timezone.utc)
    return {
        "event_ts": ts,
        "as_of_ts": ts + INGEST_LAG,
        "value": int(item["value"]),
        "classification": item.get("value_classification", ""),
    }


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    items = fetch_all(args.limit)
    log.info("Fetched %d F&G rows", len(items))

    rows = [normalize(it) for it in items]
    n = fng_store.upsert_fng_rows(pd.DataFrame(rows), root=Path(args.out_dir))
    log.info("Store now has %d unique daily rows", n)

    df = pd.DataFrame(rows)
    if not df.empty:
        log.info("Date range: %s -> %s", df["event_ts"].min(), df["event_ts"].max())
        log.info("Value range: min=%d max=%d mean=%.1f",
                 df["value"].min(), df["value"].max(), df["value"].mean())


if __name__ == "__main__":
    main()
