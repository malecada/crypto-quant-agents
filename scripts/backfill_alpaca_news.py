#!/usr/bin/env python
"""Backfill Alpaca News (Benzinga-sourced) into the bitemporal sentiment store.

Usage:
    python scripts/backfill_alpaca_news.py \
        --start 2023-10-01 --end 2026-04-15 \
        --symbols BTCUSD ETHUSD

Environment:
    ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY must be set in .env
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows import sentiment_store  # noqa: E402

ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
INGEST_LAG_SECONDS = 60
HTML_TAG_RE = re.compile(r"<[^>]+>")

log = logging.getLogger("backfill_alpaca_news")


def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill Alpaca News into the PIT sentiment store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--symbols", nargs="+", default=["BTCUSD", "ETHUSD"])
    p.add_argument("--out-dir", default="data/sentiment/alpaca")
    p.add_argument("--batch-days", type=int, default=7,
                   help="Fetch window in days per Alpaca request")
    p.add_argument("--limit", type=int, default=50,
                   help="Alpaca API page size (max 50)")
    return p.parse_args()


def strip_html(text: str | None) -> str:
    if not text:
        return ""
    return HTML_TAG_RE.sub("", text).strip()


def _headers() -> dict:
    key = os.environ.get("ALPACA_API_KEY_ID")
    sec = os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        raise RuntimeError(
            "ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY missing. "
            "Add them to .env (see prerequisites in the plan)."
        )
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def fetch_window(symbols: list[str], start: datetime, end: datetime,
                 limit: int = 50) -> Iterable[dict]:
    """Yield raw Alpaca news items within [start, end), paginating via next_page_token."""
    params = {
        "symbols": ",".join(symbols),
        "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "limit": limit,
        "include_content": "true",
        "sort": "asc",
    }
    backoff = 1.0
    while True:
        resp = requests.get(ALPACA_NEWS_URL, headers=_headers(),
                            params=params, timeout=30)
        if resp.status_code == 429:
            log.warning("429 rate limit — sleeping %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        resp.raise_for_status()
        backoff = 1.0
        payload = resp.json()
        for item in payload.get("news", []) or []:
            yield item
        tok = payload.get("next_page_token")
        if not tok:
            return
        params["page_token"] = tok
        time.sleep(0.35)  # stay well under 200 req/min


def normalize(item: dict) -> dict:
    created = pd.to_datetime(item["created_at"], utc=True).to_pydatetime()
    symbols = ",".join(item.get("symbols") or [])
    return {
        "event_ts": created,
        "as_of_ts": created + timedelta(seconds=INGEST_LAG_SECONDS),
        "id": int(item["id"]),
        "headline": item.get("headline") or "",
        "content": strip_html(item.get("content")),
        "summary": strip_html(item.get("summary")),
        "symbols": symbols,
        "source": item.get("source") or "",
        "author": item.get("author") or "",
        "url": item.get("url") or "",
    }


def daterange(start: datetime, end: datetime, step_days: int):
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=step_days), end)
        yield cur, nxt
        cur = nxt


def main():
    load_dotenv()
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_dir = Path(args.out_dir)

    total = 0
    by_month: dict[tuple[int, int], list[dict]] = {}
    for window_start, window_end in daterange(start, end, args.batch_days):
        log.info("Fetching %s → %s", window_start.date(), window_end.date())
        for item in fetch_window(args.symbols, window_start, window_end, args.limit):
            row = normalize(item)
            month_key = (row["event_ts"].year, row["event_ts"].month)
            by_month.setdefault(month_key, []).append(row)
            total += 1
        for (y, m), rows in list(by_month.items()):
            if len(rows) >= 500:
                sentiment_store.upsert_alpaca_rows(
                    pd.DataFrame(rows), year=y, month=m, root=out_dir)
                log.info("Flushed %d rows to %d-%02d.parquet", len(rows), y, m)
                by_month.pop((y, m), None)

    for (y, m), rows in by_month.items():
        if rows:
            sentiment_store.upsert_alpaca_rows(
                pd.DataFrame(rows), year=y, month=m, root=out_dir)
            log.info("Flushed %d rows to %d-%02d.parquet", len(rows), y, m)

    log.info("Backfill complete: %d articles", total)


if __name__ == "__main__":
    main()
