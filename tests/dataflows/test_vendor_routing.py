"""Vendor routing sanity tests — catches positional-dispatch signature bugs.

The LangChain @tool wrappers call route_to_vendor positionally, so vendor
implementations must accept the same positional args that the tool exposes.
Unit tests that use keyword arguments can hide signature mismatches.
"""
from __future__ import annotations

import pytest

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.interface import route_to_vendor


def _configure_pit():
    cfg = DEFAULT_CONFIG.copy()
    cfg["data_vendors"] = dict(cfg["data_vendors"])
    cfg["data_vendors"]["crypto_sentiment"] = "crypto_sentiment_pit"
    cfg["data_vendors"]["news_data"] = "news_data_pit"
    set_config(cfg)


def test_route_get_crypto_google_news_positional_pit(tmp_path, monkeypatch):
    """Must accept positional (coin, start_date, end_date) like the LangChain tool."""
    from tradingagents.dataflows import sentiment_store, fng_store
    monkeypatch.setattr(sentiment_store, "DEFAULT_ROOT", tmp_path)
    monkeypatch.setattr(fng_store, "DEFAULT_ROOT", tmp_path / "fng")
    _configure_pit()
    out = route_to_vendor("get_crypto_google_news", "bitcoin", "2024-01-08", "2024-01-15")
    assert isinstance(out, str)
    # Empty store → notice rather than an exception
    assert "no" in out.lower() or "alpaca" in out.lower()


def test_route_get_reddit_posts_positional_pit():
    """Stub must return a positional-dispatch compatible 'not available' message."""
    _configure_pit()
    out = route_to_vendor("get_reddit_posts", "bitcoin", "2024-01-01", "2024-01-10")
    assert "not available" in out.lower()


def test_route_get_news_positional_pit():
    _configure_pit()
    out = route_to_vendor("get_news", "bitcoin", "2024-01-01", "2024-01-15")
    assert "not available" in out.lower() or "pit" in out.lower()


def test_route_get_global_news_positional_pit():
    _configure_pit()
    out = route_to_vendor("get_global_news", "2024-01-15")
    assert "not available" in out.lower() or "pit" in out.lower()
