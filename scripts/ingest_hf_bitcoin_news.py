#!/usr/bin/env python
"""One-shot ingest of the edaschau/bitcoin_news HuggingFace dataset.

Dataset covers 2011-06 → 2024-01-24 — does NOT overlap the current 2026
backtest window. Included as a thesis artifact for future longer-window
backtests (2024-05 → 2024-12 post-cutoff validation, etc.).

Files pulled from HF:
- BTC_yahoo.csv       (80,806 rows — Yahoo Finance sourced)
- BTC_match_title.csv (30,626 rows — title-matched across sources)
- BTC_match_text.csv  (99,400 rows — text-matched)

After dedup by URL, stored in the same bitemporal Parquet layout as
Alpaca news (data/sentiment/hf_btc/{year}/{month:02d}.parquet) so
sentiment_store.query_news can union across both sources.

Usage:
    python scripts/ingest_hf_bitcoin_news.py
    python scripts/ingest_hf_bitcoin_news.py --out-dir data/sentiment/hf_btc
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows import sentiment_store  # noqa: E402

REPO = "edaschau/bitcoin_news"
FILES = ["BTC_yahoo.csv", "BTC_match_title.csv", "BTC_match_text.csv"]
INGEST_LAG = timedelta(minutes=30)  # plausible lag between article publish and corpus ingest

log = logging.getLogger("ingest_hf_btc")


def parse_args():
    p = argparse.ArgumentParser(
        description="Ingest edaschau/bitcoin_news into the sentiment store.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-dir", default="data/sentiment/hf_btc")
    p.add_argument("--cache-dir", default="/tmp/bitcoin_news_raw")
    p.add_argument("--files", nargs="+", default=FILES)
    return p.parse_args()


def stable_id(url: str, time_unix: int) -> int:
    """Derive a numeric id from url + timestamp since HF rows lack a numeric id."""
    payload = f"{url}|{time_unix}".encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()
    # Take 15 hex chars (60 bits) so we fit in int64
    return int(h[:15], 16)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map HF CSV schema → sentiment_store.SCHEMA_COLS."""
    event_ts = pd.to_datetime(df["time_unix"], unit="s", utc=True)
    return pd.DataFrame({
        "event_ts": event_ts,
        "as_of_ts": event_ts + INGEST_LAG,
        "id": [stable_id(str(u), int(t)) for u, t in zip(df["url"], df["time_unix"])],
        "headline": df["title"].fillna("").astype(str),
        "content": df["article_text"].fillna("").astype(str),
        "summary": "",
        "symbols": "BTCUSD",  # dataset is crypto/BTC-scoped
        "source": df["source"].fillna("").astype(str),
        "author": "",
        "url": df["url"].fillna("").astype(str),
    })


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    frames: list[pd.DataFrame] = []
    for fname in args.files:
        log.info("Downloading %s …", fname)
        path = hf_hub_download(repo_id=REPO, repo_type="dataset",
                               filename=fname, local_dir=args.cache_dir)
        df = pd.read_csv(path)
        log.info("  %s: %d rows", fname, len(df))
        frames.append(normalize(df))

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["url"], keep="first")
    combined = combined.drop_duplicates(subset=["id"], keep="first")
    log.info("Combined: %d rows (%d after dedup)", before, len(combined))

    # Partition by year/month and upsert into sentiment_store
    combined["_year"] = combined["event_ts"].dt.year
    combined["_month"] = combined["event_ts"].dt.month
    total = 0
    for (y, m), group in combined.groupby(["_year", "_month"]):
        group = group.drop(columns=["_year", "_month"])
        n = sentiment_store.upsert_alpaca_rows(
            group, year=int(y), month=int(m), root=Path(args.out_dir)
        )
        log.info("  %d-%02d: %d rows after upsert", y, m, n)
        total += n

    log.info("Ingest complete: %d total rows across %s", total,
             f"{combined['event_ts'].min()} → {combined['event_ts'].max()}")


if __name__ == "__main__":
    main()
