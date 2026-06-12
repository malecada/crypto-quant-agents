"""Tests for the PIT crypto sentiment tool wrappers."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from tradingagents.dataflows import sentiment_store, crypto_sentiment_pit, fng_store


def _row(ts, article_id, symbols="BTCUSD", headline="H", content="body"):
    return {
        "event_ts": ts, "as_of_ts": ts, "id": article_id,
        "headline": headline, "content": content, "summary": "",
        "symbols": symbols, "source": "Benzinga", "author": "", "url": "",
    }


def test_get_crypto_news_pit_returns_formatted_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    rows = pd.DataFrame([
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 1,
             headline="BTC surges on ETF approval"),
        _row(datetime(2024, 1, 12, tzinfo=timezone.utc), 2,
             headline="SEC hints at stricter enforcement"),
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    out = crypto_sentiment_pit.get_crypto_news_pit(
        coin_name="bitcoin",
        start_date="2024-01-08",
        end_date="2024-01-15",
    )
    assert "Alpaca" in out or "Benzinga" in out
    assert "BTC surges on ETF approval" in out
    assert "SEC hints at stricter enforcement" in out


def test_get_crypto_news_pit_respects_pit_cutoff(tmp_path, monkeypatch):
    """An article whose event_ts falls inside the window but whose as_of_ts is AFTER
    the trade_date must not appear in the report."""
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    rows = pd.DataFrame([
        _row(datetime(2024, 1, 10, tzinfo=timezone.utc), 1, headline="visible"),
        {
            "event_ts": datetime(2024, 1, 12, tzinfo=timezone.utc),
            "as_of_ts": datetime(2024, 3, 1, tzinfo=timezone.utc),
            "id": 2, "headline": "leaked future", "content": "",
            "summary": "", "symbols": "BTCUSD", "source": "x", "author": "", "url": "",
        },
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    out = crypto_sentiment_pit.get_crypto_news_pit(
        coin_name="bitcoin", start_date="2024-01-08", end_date="2024-01-15",
    )
    assert "visible" in out
    assert "leaked future" not in out


def test_get_crypto_news_pit_empty_returns_notice(tmp_path, monkeypatch):
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(fng_store, "DEFAULT_ROOT", tmp_path / "fng")
    out = crypto_sentiment_pit.get_crypto_news_pit(
        coin_name="bitcoin", start_date="2024-01-08", end_date="2024-01-15",
    )
    assert "No" in out  # "No Alpaca articles found" or "No cached sentiment"


def test_get_reddit_posts_pit_stub_returns_disabled_message():
    """P1 does not implement Reddit PIT; stub must explicitly say so
    rather than silently fall back to live data."""
    out = crypto_sentiment_pit.get_reddit_posts_pit_stub(
        coin_name="bitcoin", start_date="2024-01-01", end_date="2024-01-10",
    )
    assert "not available" in out.lower() or "disabled" in out.lower()


def test_get_crypto_news_pit_respects_lookback_boundary(tmp_path, monkeypatch):
    """An article older than lookback_days must not appear."""
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    rows = pd.DataFrame([
        _row(datetime(2024, 1, 7, 23, 0, tzinfo=timezone.utc), 1, headline="too old"),
        _row(datetime(2024, 1, 9, 12, 0, tzinfo=timezone.utc), 2, headline="inside window"),
    ])
    sentiment_store.upsert_alpaca_rows(rows, year=2024, month=1, root=tmp_path)

    # Window = [2024-01-08, 2024-01-15]. The 2024-01-07 article is outside
    # the window and must NOT appear; the 2024-01-09 article must.
    out = crypto_sentiment_pit.get_crypto_news_pit(
        coin_name="bitcoin", start_date="2024-01-08", end_date="2024-01-15",
    )
    assert "inside window" in out
    assert "too old" not in out


def test_get_crypto_news_pit_unsupported_coin_returns_error_string(tmp_path, monkeypatch):
    """An unsupported coin name must return an error string, not raise."""
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    out = crypto_sentiment_pit.get_crypto_news_pit(
        coin_name="zzznotarealcoin", start_date="2024-01-08", end_date="2024-01-15",
    )
    assert "error" in out.lower() or "unsupported" in out.lower()
