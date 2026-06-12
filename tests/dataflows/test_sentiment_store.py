"""Unit tests for tradingagents.dataflows.sentiment_store."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from tradingagents.dataflows import sentiment_store


def _row(ts: datetime, article_id: int, symbols: str = "BTCUSD",
         headline: str = "Example", content: str = "", source: str = "Benzinga") -> dict:
    return {
        "event_ts": ts,
        "as_of_ts": ts,
        "id": article_id,
        "headline": headline,
        "content": content,
        "summary": "",
        "symbols": symbols,
        "source": source,
        "author": "",
        "url": f"https://example.com/{article_id}",
    }


def test_roundtrip_single_month(tmp_path):
    """Ingest 3 rows, query the window containing them, get them back."""
    base = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
    rows = pd.DataFrame([
        _row(base.replace(day=10), 1, headline="Article 1"),
        _row(base.replace(day=15), 2, headline="Article 2"),
        _row(base.replace(day=20), 3, headline="Article 3"),
    ])

    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    out = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(out) == 3
    assert sorted(out["headline"].tolist()) == ["Article 1", "Article 2", "Article 3"]


def test_pit_filter_excludes_future_observations(tmp_path):
    """A row whose event_ts is before as_of but whose as_of_ts is AFTER
    must be excluded — this is the PIT rule."""
    rows = pd.DataFrame([
        # event ts in Jan; observed in Jan (visible at Feb 1)
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 1, headline="Known early"),
        # event ts in Jan but only entered the store in March (NOT visible at Feb 1)
        {
            "event_ts": datetime(2024, 1, 25, tzinfo=timezone.utc),
            "as_of_ts": datetime(2024, 3, 5, tzinfo=timezone.utc),
            "id": 2, "headline": "Late ingest", "content": "",
            "summary": "", "symbols": "BTCUSD", "source": "x", "author": "", "url": "",
        },
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    out = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    ids = out["id"].tolist()
    assert 1 in ids, "row observed before as_of should be visible"
    assert 2 not in ids, "row ingested after as_of must be filtered out"


def test_symbol_filter_isolates_coin(tmp_path):
    """bitcoin query must not return rows tagged only with ETHUSD."""
    rows = pd.DataFrame([
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 1, symbols="BTCUSD"),
        _row(datetime(2024, 1, 11, tzinfo=timezone.utc), 2, symbols="ETHUSD"),
        _row(datetime(2024, 1, 12, tzinfo=timezone.utc), 3, symbols="BTCUSD,ETHUSD"),
        _row(datetime(2024, 1, 13, tzinfo=timezone.utc), 4, symbols="SOLUSD"),
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    btc = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert sorted(btc["id"].tolist()) == [1, 3]

    eth = sentiment_store.query_news(
        coin="ethereum",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert sorted(eth["id"].tolist()) == [2, 3]


def test_wbtc_symbol_does_not_contaminate_btc_query(tmp_path):
    """A row tagged only with WBTCUSD (Wrapped Bitcoin) must not appear
    in a bitcoin query — previously the substring LIKE '%BTCUSD%' leaked it."""
    rows = pd.DataFrame([
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 1, symbols="BTCUSD"),
        _row(datetime(2024, 1, 11, tzinfo=timezone.utc), 2, symbols="WBTCUSD"),
        _row(datetime(2024, 1, 12, tzinfo=timezone.utc), 3, symbols="BTCUSD,WBTCUSD"),
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    btc = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert sorted(btc["id"].tolist()) == [1, 3], (
        "WBTCUSD-only row must be excluded from bitcoin query"
    )


def test_revision_history_preserved_across_upserts(tmp_path):
    """Two revisions of the same article id (same event_ts, different as_of_ts)
    must both survive upsert. A query at an as_of between the two must return
    the earlier revision, not the later one."""
    event_time = datetime(2024, 1, 10, tzinfo=timezone.utc)
    # v1: headline as published
    v1 = pd.DataFrame([{
        "event_ts": event_time,
        "as_of_ts": datetime(2024, 1, 10, 12, 0, tzinfo=timezone.utc),
        "id": 42, "headline": "v1 headline", "content": "",
        "summary": "", "symbols": "BTCUSD", "source": "x", "author": "", "url": "",
    }])
    # v2: corrected headline, observed 5 days later
    v2 = pd.DataFrame([{
        "event_ts": event_time,
        "as_of_ts": datetime(2024, 1, 15, 9, 0, tzinfo=timezone.utc),
        "id": 42, "headline": "v2 corrected", "content": "",
        "summary": "", "symbols": "BTCUSD", "source": "x", "author": "", "url": "",
    }])
    sentiment_store.upsert_alpaca_rows(v1, year=2024, month=1, root=tmp_path)
    sentiment_store.upsert_alpaca_rows(v2, year=2024, month=1, root=tmp_path)

    # At as_of between v1 and v2: only v1 should be visible
    out_between = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 1, 12, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(out_between) == 1
    assert out_between["headline"].iloc[0] == "v1 headline"

    # At as_of after v2: both revisions are visible; the later (v2) comes first due to ORDER BY event_ts DESC
    # (same event_ts → order between them is unspecified but both must be present)
    out_after = sentiment_store.query_news(
        coin="bitcoin",
        ts_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        ts_end=datetime(2024, 1, 31, tzinfo=timezone.utc),
        as_of=datetime(2024, 2, 1, tzinfo=timezone.utc),
        root=tmp_path,
    )
    assert len(out_after) == 2
    assert set(out_after["headline"].tolist()) == {"v1 headline", "v2 corrected"}
