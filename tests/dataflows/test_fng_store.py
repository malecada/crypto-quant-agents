"""Unit tests for tradingagents.dataflows.fng_store."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from tradingagents.dataflows import fng_store


def _row(ts: datetime, value: int, classification: str = "Fear",
         as_of: datetime | None = None) -> dict:
    return {
        "event_ts": ts,
        "as_of_ts": as_of if as_of is not None else ts,
        "value": value,
        "classification": classification,
    }


def test_roundtrip(tmp_path):
    base = datetime(2024, 1, 15, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame([
        _row(base.replace(day=10), 20, "Extreme Fear"),
        _row(base.replace(day=11), 35, "Fear"),
        _row(base.replace(day=12), 55, "Greed"),
    ])
    fng_store.upsert_fng_rows(df, root=tmp_path)

    out = fng_store.query_fng(
        trade_date=pd.Timestamp("2024-01-12", tz="UTC"),
        lookback_days=5,
        root=tmp_path,
    )
    assert len(out) == 3
    assert out["value"].tolist() == [20, 35, 55]
    assert out["classification"].iloc[0] == "Extreme Fear"


def test_dedup_by_event_ts_keeps_latest(tmp_path):
    ts = datetime(2024, 1, 10, tzinfo=timezone.utc)
    df1 = pd.DataFrame([_row(ts, 20)])
    fng_store.upsert_fng_rows(df1, root=tmp_path)
    # simulate correction the next day
    df2 = pd.DataFrame([_row(ts, 25, as_of=ts + pd.Timedelta(days=1))])
    fng_store.upsert_fng_rows(df2, root=tmp_path)

    out = fng_store.query_fng(
        trade_date=pd.Timestamp("2024-01-15", tz="UTC"),
        lookback_days=10,
        root=tmp_path,
    )
    assert len(out) == 1
    assert out["value"].iloc[0] == 25


def test_pit_filter_excludes_future_observations(tmp_path):
    """A row whose event_ts is before trade_date but whose as_of_ts is AFTER
    must be filtered out."""
    df = pd.DataFrame([
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 30,
             as_of=datetime(2024, 1, 10, 1, tzinfo=timezone.utc)),  # observed same day
        _row(datetime(2024, 1, 12, tzinfo=timezone.utc), 88,
             as_of=datetime(2024, 3, 1, tzinfo=timezone.utc)),  # observed months later
    ])
    fng_store.upsert_fng_rows(df, root=tmp_path)

    out = fng_store.query_fng(
        trade_date=pd.Timestamp("2024-01-15", tz="UTC"),
        lookback_days=10,
        root=tmp_path,
    )
    assert len(out) == 1
    assert out["value"].iloc[0] == 30


def test_empty_store_returns_empty_df(tmp_path):
    out = fng_store.query_fng(
        trade_date=pd.Timestamp("2024-01-15", tz="UTC"),
        lookback_days=7,
        root=tmp_path,
    )
    assert out.empty
    assert list(out.columns) == fng_store.SCHEMA_COLS
