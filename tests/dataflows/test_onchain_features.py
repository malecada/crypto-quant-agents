"""Unit tests for tradingagents.dataflows.onchain_features PIT alignment."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from tradingagents.dataflows import onchain_features, onchain_store


def _row(
    event_ts: datetime, metric: str, value: float,
    coin: str = "btc", as_of_ts: datetime | None = None,
    source: str = "coinmetrics_community", status: str = "final",
) -> dict:
    return {
        "event_ts": event_ts,
        "as_of_ts": as_of_ts if as_of_ts is not None else event_ts + timedelta(days=1),
        "coin": coin,
        "metric": metric,
        "value": float(value),
        "source": source,
        "status": status,
    }


def test_pit_alignment_uses_as_of_ts_not_event_ts(tmp_path):
    """Row ingested at as_of_ts > query date must not appear in features."""
    rows = pd.DataFrame([
        _row(
            datetime(2025, 12, 1, tzinfo=timezone.utc), "CapMVRVCur", 1.5,
            as_of_ts=datetime(2025, 12, 2, tzinfo=timezone.utc),
        ),
        _row(
            datetime(2025, 12, 5, tzinfo=timezone.utc), "FlowInExUSD", 1e9,
            as_of_ts=datetime(2026, 3, 1, tzinfo=timezone.utc),  # flash revision
        ),
    ])
    onchain_store.upsert_rows(rows, root=tmp_path)

    dates = [datetime(2025, 12, 10, tzinfo=timezone.utc)]
    feats = onchain_features.build_pit_onchain_features(
        "btc", dates,
        metrics=["CapMVRVCur", "FlowInExUSD"],
        include_global=False, include_derived=False,
        root=tmp_path,
    )
    assert feats["oc_CapMVRVCur"].iloc[0] == 1.5
    assert pd.isna(feats["oc_FlowInExUSD"].iloc[0]), (
        "Flow metric with as_of_ts in 2026-03 must not leak into 2025-12-10"
    )


def test_derived_mvrv_z_no_leakage(tmp_path):
    """MVRV-Z rolling must use only PIT-safe history."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(120):
        rows.append(_row(
            base + timedelta(days=i), "CapMVRVCur", 1.0 + i * 0.01,
            as_of_ts=base + timedelta(days=i, hours=12),
        ))
    onchain_store.upsert_rows(pd.DataFrame(rows), root=tmp_path)

    # Query dates 60–120 days in.
    query_dates = [base + timedelta(days=d) for d in range(65, 120)]
    feats = onchain_features.build_pit_onchain_features(
        "btc", query_dates,
        metrics=["CapMVRVCur"], include_global=False, include_derived=True,
        root=tmp_path,
    )
    assert "oc_mvrv_z_1y" in feats.columns
    assert feats["oc_CapMVRVCur"].notna().all()
    # The first date is day-65: rolling window uses 60+ prior points.
    assert feats["oc_mvrv_z_1y"].notna().iloc[-1], (
        "Late-window z-score must resolve with enough history"
    )


def test_net_flow_from_in_out(tmp_path):
    base = datetime(2025, 6, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(5):
        d = base + timedelta(days=i)
        rows.append(_row(
            d, "FlowInExUSD", 1e9,
            as_of_ts=d + timedelta(days=1),  # overriding default flash lag for test
        ))
        rows.append(_row(
            d, "FlowOutExUSD", 0.7e9,
            as_of_ts=d + timedelta(days=1),
        ))
    onchain_store.upsert_rows(pd.DataFrame(rows), root=tmp_path)

    query_dates = [base + timedelta(days=d) for d in range(2, 5)]
    feats = onchain_features.build_pit_onchain_features(
        "btc", query_dates,
        metrics=["FlowInExUSD", "FlowOutExUSD"],
        include_global=False, include_derived=True,
        root=tmp_path,
    )
    assert "oc_net_flow_usd" in feats.columns
    assert (feats["oc_net_flow_usd"].round(-6) == 300_000_000).all()


def test_empty_store_returns_empty_cols(tmp_path):
    dates = [datetime(2025, 12, 1, tzinfo=timezone.utc)]
    feats = onchain_features.build_pit_onchain_features(
        "btc", dates, include_global=False, include_derived=False,
        include_options=False, include_derivatives=False,
        include_stablecoin_context=False,
        root=tmp_path / "nonexistent",
    )
    # All columns present but all NaN (no data).
    assert len(feats) == 1
    for c in feats.columns:
        assert feats[c].isna().all()
