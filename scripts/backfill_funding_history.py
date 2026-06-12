"""Backfill Binance Futures funding rates pre-existing cache window.

Existing cache (data/derivatives_raw/{SYMBOL}_funding.parquet) starts 2024-01-01.
This script extends it back to ``--start`` (default 2021-11-01) by paginated
fetch through /fapi/v1/fundingRate, merges, dedupes, sorts, and overwrites the
parquet. Backup with ``.bak.{YYYYMMDD}`` suffix is written before overwrite.

Usage:
    python scripts/backfill_funding_history.py \\
        --symbols BTCUSDT ETHUSDT \\
        --start 2021-11-01
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.strategies.v3.features._http import with_backoff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def _fetch_page(symbol: str, start_ms: int, end_ms: int, limit: int = 1000) -> list[dict]:
    resp = requests.get(
        _BINANCE_FUNDING_URL,
        params={"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_range(symbol: str, start: pd.Timestamp, end: pd.Timestamp, limit: int = 1000) -> pd.DataFrame:
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict] = []
    while cursor_ms < end_ms:
        page = with_backoff(lambda: _fetch_page(symbol, cursor_ms, end_ms, limit))
        if not page:
            break
        rows.extend(page)
        last_time = page[-1]["fundingTime"]
        if last_time <= cursor_ms:
            break
        cursor_ms = last_time + 1
        time.sleep(0.15)
    if not rows:
        return pd.DataFrame(columns=["funding_rate"])
    df = pd.DataFrame(
        {"funding_rate": [float(r["fundingRate"]) for r in rows]},
        index=pd.to_datetime([r["fundingTime"] for r in rows], unit="ms", utc=True),
    )
    df.index.name = "ts"
    return df.sort_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--start", default="2021-11-01")
    parser.add_argument("--cache-dir", default="data/derivatives_raw")
    parser.add_argument("--daily-out-dir", default="data/derivatives", help="Where to write daily aggregates consumed by V3 runner")
    args = parser.parse_args()
    symbol_to_coin = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "BNBUSDT": "binancecoin", "SOLUSDT": "solana"}
    daily_dir = Path(args.daily_out_dir)
    daily_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.utcnow().tz_convert("UTC").normalize()
    bak_suffix = pd.Timestamp.utcnow().strftime(".bak.%Y%m%d")

    for symbol in args.symbols:
        cache_file = cache_dir / f"{symbol}_funding.parquet"
        existing = pd.DataFrame(columns=["funding_rate"])
        if cache_file.exists():
            existing = pd.read_parquet(cache_file)
            backup = cache_file.with_suffix(cache_file.suffix + bak_suffix)
            shutil.copy2(cache_file, backup)
            logger.info("%s: existing %d rows %s → %s; backup → %s", symbol, len(existing), existing.index.min(), existing.index.max(), backup.name)
            gap_end = existing.index.min()
        else:
            gap_end = today

        if gap_end <= start:
            logger.info("%s: existing cache already covers start=%s, skipping", symbol, args.start)
            continue

        logger.info("%s: backfilling %s → %s", symbol, args.start, gap_end)
        new_rows = fetch_range(symbol, start, gap_end)
        logger.info("%s: fetched %d new rows", symbol, len(new_rows))

        merged = pd.concat([existing, new_rows]) if not existing.empty else new_rows
        merged = merged[~merged.index.duplicated(keep="first")].sort_index()
        merged.to_parquet(cache_file)
        logger.info(
            "%s: wrote %d rows %s → %s",
            symbol, len(merged), merged.index.min(), merged.index.max(),
        )

        coin = symbol_to_coin.get(symbol)
        if coin is None:
            logger.warning("%s: no coin mapping — skipping daily aggregate regen", symbol)
            continue
        daily = merged["funding_rate"].resample("D").mean().to_frame("funding_rate")
        daily["funding_rate_ma7"] = daily["funding_rate"].rolling(7).mean().fillna(0.0)
        daily_file = daily_dir / f"{coin}.parquet"
        if daily_file.exists():
            daily_bak = daily_file.with_suffix(daily_file.suffix + bak_suffix)
            shutil.copy2(daily_file, daily_bak)
        daily.to_parquet(daily_file)
        logger.info(
            "%s → %s: %d daily rows %s → %s",
            symbol, daily_file.name, len(daily), daily.index.min().date(), daily.index.max().date(),
        )


if __name__ == "__main__":
    main()
