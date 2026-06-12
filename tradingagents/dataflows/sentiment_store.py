"""Bitemporal sentiment store backed by Parquet + DuckDB.

Layout: data/sentiment/alpaca/{year}/{month:02d}.parquet.
Every row has (event_ts, as_of_ts) so backtests can enforce
as_of_ts <= trade_date to avoid look-ahead bias.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

DEFAULT_ROOT = Path("data/sentiment/alpaca")

COIN_TO_SYMBOL: dict[str, str] = {
    "bitcoin": "BTCUSD",
    "ethereum": "ETHUSD",
    "binancecoin": "BNBUSD",
    "solana": "SOLUSD",
    "dogecoin": "DOGEUSD",
    "cardano": "ADAUSD",
}

SCHEMA_COLS = [
    "event_ts", "as_of_ts", "id", "headline", "content",
    "summary", "symbols", "source", "author", "url",
]


def _month_path(root: Path, year: int, month: int) -> Path:
    return Path(root) / str(year) / f"{month:02d}.parquet"


def upsert_alpaca_rows(df: pd.DataFrame, year: int, month: int,
                       root: Path = DEFAULT_ROOT) -> int:
    """Merge rows into the month Parquet, deduping by `id`. Returns rows written."""
    if df.empty:
        return 0
    missing = set(SCHEMA_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"upsert missing columns: {sorted(missing)}")
    target = _month_path(root, year, month)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_parquet(target)
        combined = pd.concat([existing, df[SCHEMA_COLS]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["id", "as_of_ts"], keep="last")
    else:
        combined = df[SCHEMA_COLS].drop_duplicates(subset=["id", "as_of_ts"], keep="last")
    combined.to_parquet(target, index=False)
    return len(combined)


def query_news(coin: str, ts_start: datetime, ts_end: datetime,
               as_of: datetime, limit: int = 50,
               root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Return rows where event_ts in [ts_start, ts_end] AND as_of_ts <= as_of,
    filtered to the coin's symbol. Enforces the PIT rule."""
    symbol = COIN_TO_SYMBOL.get(coin.lower())
    if symbol is None:
        raise ValueError(f"Unsupported coin for sentiment store: {coin!r}")
    glob = f"{root}/*/*.parquet"
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"CREATE VIEW news AS SELECT * FROM read_parquet('{glob}')")
        except duckdb.IOException:
            return pd.DataFrame(columns=SCHEMA_COLS)
        sql = """
        SELECT event_ts, as_of_ts, id, headline, content, summary,
               symbols, source, author, url
        FROM news
        WHERE event_ts BETWEEN ? AND ?
          AND as_of_ts <= ?
          AND list_contains(string_split(symbols, ','), ?)
        ORDER BY event_ts DESC
        LIMIT ?
        """
        return con.execute(
            sql,
            [ts_start, ts_end, as_of, symbol, limit],
        ).fetchdf()
    finally:
        con.close()
