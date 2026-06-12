"""Bitemporal Fear & Greed store backed by single Parquet file.

alternative.me publishes one F&G integer (0-100) per day. We store
(event_ts, as_of_ts) on every row so backtests can enforce
as_of_ts <= trade_date, same PIT rule as the Alpaca news store.

Layout: data/sentiment/fng/fng.parquet
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

DEFAULT_ROOT = Path("data/sentiment/fng")
DEFAULT_FILE = "fng.parquet"

SCHEMA_COLS = ["event_ts", "as_of_ts", "value", "classification"]


def _path(root: Path) -> Path:
    return Path(root) / DEFAULT_FILE


def upsert_fng_rows(df: pd.DataFrame, root: Path = DEFAULT_ROOT) -> int:
    """Merge F&G rows into the single Parquet file, dedup by event_ts (UTC day)."""
    if df.empty:
        return 0
    missing = set(SCHEMA_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"upsert missing columns: {sorted(missing)}")
    target = _path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_parquet(target)
        combined = pd.concat([existing, df[SCHEMA_COLS]], ignore_index=True)
        combined = combined.drop_duplicates(subset=["event_ts"], keep="last")
    else:
        combined = df[SCHEMA_COLS].drop_duplicates(subset=["event_ts"], keep="last")
    combined = combined.sort_values("event_ts").reset_index(drop=True)
    combined.to_parquet(target, index=False)
    return len(combined)


def query_fng(trade_date: datetime, lookback_days: int = 7,
              root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """Return F&G rows in [trade_date - lookback_days, trade_date], PIT-filtered.

    Rows are returned ascending by event_ts so the caller can format
    a simple time-series snippet.
    """
    path = _path(root)
    if not path.exists():
        return pd.DataFrame(columns=SCHEMA_COLS)
    con = duckdb.connect(":memory:")
    try:
        con.execute(f"CREATE VIEW fng AS SELECT * FROM read_parquet('{path}')")
        sql = """
        SELECT event_ts, as_of_ts, value, classification
        FROM fng
        WHERE event_ts >= ? AND event_ts <= ?
          AND as_of_ts <= ?
        ORDER BY event_ts ASC
        """
        ts_end = trade_date
        ts_start = trade_date - pd.Timedelta(days=lookback_days)
        return con.execute(sql, [ts_start, ts_end, trade_date]).fetchdf()
    finally:
        con.close()
