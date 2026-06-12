"""Unit tests for tradingagents.dataflows.onchain_store."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingagents.dataflows import onchain_store


def _row(
    event_ts: datetime,
    metric: str,
    value: float,
    coin: str = "btc",
    source: str = "coinmetrics_community",
    as_of_ts: datetime | None = None,
    status: str = "final",
) -> dict:
    return {
        "event_ts": event_ts,
        "as_of_ts": as_of_ts if as_of_ts is not None else event_ts,
        "coin": coin,
        "metric": metric,
        "value": float(value),
        "source": source,
        "status": status,
    }


def test_roundtrip_single_month(tmp_path):
    base = datetime(2025, 12, 1, tzinfo=timezone.utc)
    rows = pd.DataFrame([
        _row(base.replace(day=d), "CapMVRVCur", 1.5 + d / 100) for d in (1, 2, 3)
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)
    out = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(out) == 3
    assert list(out["metric"].unique()) == ["CapMVRVCur"]
    assert sorted(out["value"].round(2).tolist()) == [1.51, 1.52, 1.53]


def test_pit_filter_excludes_future_observations(tmp_path):
    """as_of_ts > query as_of must not return row."""
    rows = pd.DataFrame([
        _row(
            datetime(2025, 12, 1, tzinfo=timezone.utc), "CapMVRVCur", 1.5,
            as_of_ts=datetime(2025, 12, 2, tzinfo=timezone.utc),
        ),
        _row(
            datetime(2025, 12, 2, tzinfo=timezone.utc), "FlowInExUSD", 1e9,
            as_of_ts=datetime(2026, 3, 1, tzinfo=timezone.utc),  # flash revision
        ),
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)
    out = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2025, 12, 10, tzinfo=timezone.utc),
        root=tmp_path,
    )
    metrics = out["metric"].tolist()
    assert "CapMVRVCur" in metrics
    assert "FlowInExUSD" not in metrics, "flash row ingested later must be hidden"


def test_coin_isolation(tmp_path):
    rows = pd.DataFrame([
        _row(datetime(2025, 12, 1, tzinfo=timezone.utc), "AdrActCnt", 700000, coin="btc"),
        _row(datetime(2025, 12, 1, tzinfo=timezone.utc), "AdrActCnt", 800000, coin="eth"),
        _row(datetime(2025, 12, 1, tzinfo=timezone.utc), "AdrActCnt", 42, coin="bnb"),
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)
    btc = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(btc) == 1
    assert btc["value"].iloc[0] == 700000


def test_dedupe_on_repeat_ingest(tmp_path):
    """Upserting the same (event_ts, coin, metric, source, as_of_ts) twice
    must not produce duplicates."""
    row = _row(datetime(2025, 12, 1, tzinfo=timezone.utc), "CapMVRVCur", 1.5)
    df = pd.DataFrame([row])
    onchain_store.upsert_rows(df, root=tmp_path)
    onchain_store.upsert_rows(df, root=tmp_path)
    out = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(out) == 1


def test_revision_history_preserved(tmp_path):
    """Flash → final revision: two as_of_ts values for same event_ts must both
    persist. Query at as_of between them returns only the earlier one."""
    event = datetime(2025, 12, 1, tzinfo=timezone.utc)
    flash = _row(event, "FlowInExUSD", 1.0e9,
                 as_of_ts=datetime(2025, 12, 2, tzinfo=timezone.utc), status="flash")
    final = _row(event, "FlowInExUSD", 1.2e9,
                 as_of_ts=datetime(2026, 1, 5, tzinfo=timezone.utc), status="final")
    onchain_store.upsert_rows(pd.DataFrame([flash]), root=tmp_path)
    onchain_store.upsert_rows(pd.DataFrame([final]), root=tmp_path)

    between = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2025, 12, 10, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(between) == 1
    assert between["value"].iloc[0] == 1.0e9
    assert between["status"].iloc[0] == "flash"

    later = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2026, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(later) == 2
    assert set(later["status"].tolist()) == {"flash", "final"}


def test_latest_snapshot_returns_most_recent_pit_valid(tmp_path):
    base = datetime(2025, 12, 1, tzinfo=timezone.utc)
    rows = pd.DataFrame([
        _row(base.replace(day=1), "CapMVRVCur", 1.5),
        _row(base.replace(day=2), "CapMVRVCur", 1.6),
        _row(base.replace(day=3), "CapMVRVCur", 1.7),
        _row(base.replace(day=1), "AdrActCnt", 700000),
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)
    snap = onchain_store.latest_snapshot(
        coin="btc",
        as_of=datetime(2025, 12, 15, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert set(snap["metric"].tolist()) == {"CapMVRVCur", "AdrActCnt"}
    mvrv = snap[snap["metric"] == "CapMVRVCur"].iloc[0]
    assert mvrv["value"] == 1.7
    assert mvrv["event_ts"] == pd.Timestamp("2025-12-03T00:00:00Z")


def test_latest_snapshot_respects_pit(tmp_path):
    """If asked for a snapshot before the latest value's as_of_ts, return the
    older final value."""
    rows = pd.DataFrame([
        _row(
            datetime(2025, 12, 1, tzinfo=timezone.utc), "CapMVRVCur", 1.5,
            as_of_ts=datetime(2025, 12, 2, tzinfo=timezone.utc),
        ),
        _row(
            datetime(2025, 12, 5, tzinfo=timezone.utc), "CapMVRVCur", 1.9,
            as_of_ts=datetime(2025, 12, 6, tzinfo=timezone.utc),
        ),
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)
    snap = onchain_store.latest_snapshot(
        coin="btc",
        as_of=datetime(2025, 12, 4, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(snap) == 1
    assert snap["value"].iloc[0] == 1.5


def test_missing_store_returns_empty(tmp_path):
    out = onchain_store.query_metrics(
        coin="btc",
        ts_start=datetime(2025, 12, 1, tzinfo=timezone.utc),
        ts_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
        root=tmp_path / "nonexistent",
    )
    assert out.empty
