"""Bitemporal token-unlock store (Parquet + DuckDB).

Schema mirrors ``onchain_store.py`` but with unlock-specific columns. We
store the *announced* unlock schedule as it was known at ``as_of_ts``,
so backtests evaluating an unlock T-30 to T+14 window get the exact
information that would have been available at decision time.

Layout: ``data/unlocks/{year}/{month:02d}.parquet``.

Columns:
  unlock_date              datetime64[ns, UTC]   Day of unlock (event)
  as_of_ts                 datetime64[ns, UTC]   When this row was ingested
  coin                     string                lowercase coingecko id
  amount_tokens            float64               raw token count
  pct_circulating_supply   float64               at time of ingest
  recipient_category       string                team|vc|treasury|community|airdrop|unknown
  source                   string                tokenomist|cryptorank|manual

Recipient categorisation is the actionable axis: the literature shows
team + VC + insider unlocks dominate negative post-unlock returns, while
community + airdrop are roughly neutral. ``query_unlocks`` accepts a
``recipient_categories`` filter so callers can scope to insider supply
events without re-deriving the rule downstream.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

DEFAULT_ROOT = Path("data/unlocks")

SCHEMA_COLS = [
    "unlock_date",
    "as_of_ts",
    "coin",
    "amount_tokens",
    "pct_circulating_supply",
    "recipient_category",
    "source",
]

DEDUPE_KEYS = ["unlock_date", "coin", "recipient_category", "as_of_ts"]

INSIDER_CATEGORIES = {"team", "vc", "treasury"}


def _month_path(root: Path, year: int, month: int) -> Path:
    return Path(root) / str(year) / f"{month:02d}.parquet"


def upsert_rows(df: pd.DataFrame, root: Path = DEFAULT_ROOT) -> int:
    """Write rows to monthly Parquet shards keyed by ``unlock_date`` month."""
    if df.empty:
        return 0
    missing = set(SCHEMA_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"upsert missing columns: {sorted(missing)}")
    df = df[SCHEMA_COLS].copy()
    df["unlock_date"] = pd.to_datetime(df["unlock_date"], utc=True)
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True)
    df["_year"] = df["unlock_date"].dt.year
    df["_month"] = df["unlock_date"].dt.month
    total_written = 0
    for (year, month), chunk in df.groupby(["_year", "_month"], sort=False):
        target = _month_path(root, int(year), int(month))
        target.parent.mkdir(parents=True, exist_ok=True)
        body = chunk[SCHEMA_COLS]
        if target.exists():
            existing = pd.read_parquet(target)
            combined = pd.concat([existing, body], ignore_index=True)
        else:
            combined = body
        combined = combined.drop_duplicates(subset=DEDUPE_KEYS, keep="last")
        combined.to_parquet(target, index=False)
        total_written += len(combined)
    return total_written


def query_unlocks(
    coin: str,
    ts_start: datetime,
    ts_end: datetime,
    as_of: datetime,
    recipient_categories: Optional[Iterable[str]] = None,
    root: Path = DEFAULT_ROOT,
) -> pd.DataFrame:
    """Return PIT-safe unlock rows for ``coin``.

    Filters to ``unlock_date ∈ [ts_start, ts_end]`` AND
    ``as_of_ts <= as_of``. Empty DataFrame on no data — graceful
    degradation per the plan's risk-flag for the small-vendor source.
    """
    glob = f"{root}/*/*.parquet"
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(
                f"CREATE VIEW unlocks AS SELECT * FROM read_parquet('{glob}')"
            )
        except duckdb.IOException:
            return pd.DataFrame(columns=SCHEMA_COLS)
        sql = """
        SELECT unlock_date, as_of_ts, coin, amount_tokens,
               pct_circulating_supply, recipient_category, source
        FROM unlocks
        WHERE coin = ?
          AND unlock_date BETWEEN ? AND ?
          AND as_of_ts <= ?
        """
        args: list = [coin.lower(), ts_start, ts_end, as_of]
        if recipient_categories is not None:
            cats = list(recipient_categories)
            placeholders = ",".join(["?"] * len(cats))
            sql += f" AND recipient_category IN ({placeholders})"
            args.extend(cats)
        sql += " ORDER BY unlock_date ASC"
        return con.execute(sql, args).fetchdf()
    finally:
        con.close()


def next_unlock(
    coin: str,
    as_of: datetime,
    max_days: int = 30,
    insider_only: bool = False,
    root: Path = DEFAULT_ROOT,
) -> Optional[dict]:
    """Return the soonest upcoming unlock or ``None`` if none in window."""
    end = as_of + pd.Timedelta(days=max_days)
    cats = INSIDER_CATEGORIES if insider_only else None
    rows = query_unlocks(coin, as_of, end, as_of, cats, root)
    if rows.empty:
        return None
    first = rows.iloc[0]
    return {
        "unlock_date": first["unlock_date"],
        "amount_tokens": float(first["amount_tokens"]),
        "pct_circulating_supply": float(first["pct_circulating_supply"]),
        "recipient_category": str(first["recipient_category"]),
        "days_until": int((first["unlock_date"] - as_of).days),
    }
