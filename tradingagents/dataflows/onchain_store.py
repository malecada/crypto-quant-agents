"""Bitemporal on-chain metric store (Parquet + DuckDB).

Long-format storage so heterogeneous sources (CoinMetrics, DefiLlama,
beaconcha.in, mempool.space) coexist in one layout. Every row carries
(event_ts, as_of_ts) so backtests can enforce as_of_ts <= trade_date
to avoid look-ahead bias.

Layout: data/onchain/{year}/{month:02d}.parquet.
Schema:
  event_ts      timestamp   UTC day or intraday event time
  as_of_ts      timestamp   when the value became available to a PIT caller
  coin          string      lowercase symbol (btc, eth, bnb)
  metric        string      source-namespaced metric name (e.g. cm.CapMVRVCur)
  value         double
  source        string      vendor tag (coinmetrics_community, defillama, ...)
  status        string      "final" | "flash" (CM flash = revisable)
"""
from __future__ import annotations

import os as _os
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import duckdb
import pandas as pd

_DATA_ROOT_ENV = _os.environ.get("TRADINGAGENTS_DATA_ROOT", "data")
DEFAULT_ROOT = Path(_DATA_ROOT_ENV) / "onchain"

SCHEMA_COLS = [
    "event_ts", "as_of_ts", "coin", "metric", "value", "source", "status",
]

DEDUPE_KEYS = ["event_ts", "coin", "metric", "source", "as_of_ts"]


def _month_path(root: Path, year: int, month: int) -> Path:
    return Path(root) / str(year) / f"{month:02d}.parquet"


def upsert_rows(df: pd.DataFrame, root: Path = DEFAULT_ROOT) -> int:
    """Write rows to monthly Parquet shards keyed by event_ts month.

    Dedupes on (event_ts, coin, metric, source, as_of_ts). Returns
    total rows written across all touched months.
    """
    if df.empty:
        return 0
    missing = set(SCHEMA_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"upsert missing columns: {sorted(missing)}")
    df = df[SCHEMA_COLS].copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True)
    df["_year"] = df["event_ts"].dt.year
    df["_month"] = df["event_ts"].dt.month
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


def query_metrics(
    coin: str,
    ts_start: datetime,
    ts_end: datetime,
    as_of: datetime,
    metrics: Optional[Iterable[str]] = None,
    root: Path = DEFAULT_ROOT,
) -> pd.DataFrame:
    """Return rows where event_ts in [ts_start, ts_end] AND as_of_ts <= as_of,
    filtered to the given coin. Enforces the PIT rule.

    If `metrics` is None, returns every metric available for the coin.
    Output is long-format. Pivot downstream if a wide DataFrame is needed.
    """
    glob = f"{root}/*/*.parquet"
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"CREATE VIEW onchain AS SELECT * FROM read_parquet('{glob}')")
        except duckdb.IOException:
            return pd.DataFrame(columns=SCHEMA_COLS)
        sql = """
        SELECT event_ts, as_of_ts, coin, metric, value, source, status
        FROM onchain
        WHERE coin = ?
          AND event_ts BETWEEN ? AND ?
          AND as_of_ts <= ?
        """
        args: list = [coin.lower(), ts_start, ts_end, as_of]
        if metrics is not None:
            metric_list = list(metrics)
            placeholders = ",".join(["?"] * len(metric_list))
            sql += f" AND metric IN ({placeholders})"
            args.extend(metric_list)
        sql += " ORDER BY event_ts ASC, metric ASC"
        return con.execute(sql, args).fetchdf()
    finally:
        con.close()


def latest_snapshot(
    coin: str,
    as_of: datetime,
    metrics: Optional[Iterable[str]] = None,
    root: Path = DEFAULT_ROOT,
) -> pd.DataFrame:
    """Return the most recent PIT-valid row per metric for `coin` as of `as_of`.

    One row per metric with its latest value, event time, source, and status.
    """
    glob = f"{root}/*/*.parquet"
    con = duckdb.connect(":memory:")
    try:
        try:
            con.execute(f"CREATE VIEW onchain AS SELECT * FROM read_parquet('{glob}')")
        except duckdb.IOException:
            return pd.DataFrame(columns=["metric", "event_ts", "value", "source", "status"])
        sql = """
        WITH filtered AS (
            SELECT *
            FROM onchain
            WHERE coin = ? AND as_of_ts <= ?
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY metric
                       ORDER BY event_ts DESC, as_of_ts DESC
                   ) AS rk
            FROM filtered
        )
        SELECT metric, event_ts, value, source, status
        FROM ranked
        WHERE rk = 1
        """
        args: list = [coin.lower(), as_of]
        if metrics is not None:
            metric_list = list(metrics)
            placeholders = ",".join(["?"] * len(metric_list))
            sql += f" AND metric IN ({placeholders})"
            args.extend(metric_list)
        sql += " ORDER BY metric ASC"
        return con.execute(sql, args).fetchdf()
    finally:
        con.close()
