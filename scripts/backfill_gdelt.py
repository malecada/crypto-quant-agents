#!/usr/bin/env python
"""Backfill GDELT 2.0 DOC Article Search API into the bitemporal sentiment store.

GDELT monitors ~100k news sites globally. The DOC 2.0 API
(https://api.gdeltproject.org/api/v2/doc/doc) supports server-side
keyword + date filtering so we don't pull the raw 86 GB GKG feed.

Coverage: full article text is not provided — only title + URL + seendate
(observation timestamp) + domain/language/country. Adequate for LLM
sentiment analysis: the LLM gets a stream of headlines to reason over.

Storage: data/sentiment/gdelt/{year}/{month:02d}.parquet — same
bitemporal schema as Alpaca (sentiment_store.SCHEMA_COLS), so existing
query_news() can UNION across both sources.

Usage:
    python scripts/backfill_gdelt.py \\
        --start 2026-01-16 --end 2026-04-15 \\
        --query '(bitcoin OR ethereum OR cryptocurrency)' \\
        --out-dir data/sentiment/gdelt
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows import sentiment_store  # noqa: E402

GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_QUERY = "(bitcoin OR ethereum OR cryptocurrency OR BNB OR solana) sourcelang:english"
REQUEST_SLEEP = 6.0  # stay under GDELT's ~200 req/hour soft limit

log = logging.getLogger("backfill_gdelt")


def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill GDELT DOC 2.0 crypto news into the sentiment store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--out-dir", default="data/sentiment/gdelt")
    p.add_argument("--maxrecords", type=int, default=250,
                   help="Max articles per day per request (GDELT caps at 250).")
    return p.parse_args()


def stable_id(url: str, seendate: str) -> int:
    h = hashlib.sha1(f"{url}|{seendate}".encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def fetch_day(query: str, day: datetime, maxrecords: int = 250) -> list[dict]:
    """Pull up to maxrecords crypto articles for a single UTC day."""
    start = day.strftime("%Y%m%d") + "000000"
    end = (day + timedelta(days=1)).strftime("%Y%m%d") + "000000"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": maxrecords,
        "startdatetime": start,
        "enddatetime": end,
        "sort": "datedesc",
    }
    backoff = REQUEST_SLEEP
    attempts = 0
    max_attempts = 6
    while attempts < max_attempts:
        attempts += 1
        try:
            resp = requests.get(GDELT_URL, params=params, timeout=60)
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError) as e:
            log.warning("Network error on %s (attempt %d/%d): %s — retry in %.1fs",
                        day.date(), attempts, max_attempts, e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue
        if resp.status_code == 429:
            log.warning("429 rate limit — sleeping %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 120.0)
            continue
        if resp.status_code != 200:
            log.error("GDELT %d on %s: %s", resp.status_code, day.date(),
                      resp.text[:200])
            return []
        try:
            payload = resp.json()
        except ValueError:
            log.error("Non-JSON response on %s: %s", day.date(), resp.text[:200])
            return []
        return payload.get("articles", []) or []
    log.error("Gave up on %s after %d attempts", day.date(), max_attempts)
    return []


def normalize(item: dict) -> dict:
    # seendate format: YYYYMMDDTHHMMSSZ
    sd = item.get("seendate", "")
    try:
        event_ts = datetime.strptime(sd, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return {}
    url = item.get("url", "") or ""
    return {
        "event_ts": event_ts,
        "as_of_ts": event_ts,  # GDELT observes at publication time
        "id": stable_id(url, sd),
        "headline": item.get("title", "") or "",
        "content": "",  # GDELT DOC API doesn't return article body
        "summary": "",
        "symbols": "",  # will be mapped to coin via keyword match at query time
        "source": item.get("domain", "") or "",
        "author": "",
        "url": url,
    }


def daterange(start: datetime, end: datetime) -> Iterable[datetime]:
    cur = start
    while cur < end:
        yield cur
        cur += timedelta(days=1)


def tag_symbols(row: dict) -> dict:
    """Tag the article's symbols field based on keyword matches in the headline.

    Since GDELT doesn't return article body, we match the headline against
    known crypto terms and attach the corresponding Alpaca-style symbol list
    so the PIT tool's per-coin LIKE filter can later pick the row up."""
    text = row.get("headline", "").lower()
    tags: list[str] = []
    if "bitcoin" in text or "btc" in text:
        tags.append("BTCUSD")
    if "ethereum" in text or "ether" in text or " eth" in text:
        tags.append("ETHUSD")
    if "binance" in text or "bnb" in text:
        tags.append("BNBUSD")
    if "solana" in text or "sol" in text.split():
        tags.append("SOLUSD")
    if "cardano" in text or "ada" in text.split():
        tags.append("ADAUSD")
    if not tags:
        # Crypto-general news (matched the query but no specific coin keyword)
        tags.append("BTCUSD")  # default to BTC since BTC dominates crypto news
    row["symbols"] = ",".join(tags)
    return row


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_dir = Path(args.out_dir)

    total = 0
    for day in daterange(start, end):
        items = fetch_day(args.query, day, args.maxrecords)
        rows_by_month: dict[tuple[int, int], list[dict]] = {}
        for raw in items:
            row = normalize(raw)
            if not row:
                continue
            row = tag_symbols(row)
            key = (row["event_ts"].year, row["event_ts"].month)
            rows_by_month.setdefault(key, []).append(row)
            total += 1
        for (y, m), rows in rows_by_month.items():
            sentiment_store.upsert_alpaca_rows(
                pd.DataFrame(rows), year=y, month=m, root=out_dir
            )
        log.info("%s: %d articles (saved)", day.date(), len(items))
        time.sleep(REQUEST_SLEEP)

    log.info("Backfill complete: %d articles over %d days",
             total, (end - start).days)


if __name__ == "__main__":
    main()
