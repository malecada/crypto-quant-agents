"""Unit tests for Binance -1003 IP-ban handling in ExchangeClient._retry."""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest


def _make_api_exc(message: str, status_code: int = 418, code: int = -1003):
    """Build a BinanceAPIException-shaped object without hitting the network.

    python-binance constructs BinanceAPIException from a Response; mock the
    fields _retry actually inspects (status_code, code, str(e)) instead.
    """
    from binance.exceptions import BinanceAPIException

    exc = BinanceAPIException.__new__(BinanceAPIException)
    exc.status_code = status_code
    exc.code = code
    exc.message = message
    exc.response = None
    exc.request = None
    return exc


def test_retry_raises_BinanceIPBan_on_1003_with_timestamp():
    from tradingagents.execution.exchange import BinanceIPBan, ExchangeClient

    future_ms = int(time.time() * 1000) + 60_000  # 60s in future
    err = _make_api_exc(
        f"APIError(code=-1003): Way too many requests; "
        f"IP(15.158.242.73) banned until {future_ms}. Use websocket."
    )

    def boom():
        raise err

    with pytest.raises(BinanceIPBan) as exc_info:
        ExchangeClient._retry(boom)

    assert exc_info.value.until_ms == future_ms
    assert exc_info.value.seconds_remaining > 0
    assert "-1003" in exc_info.value.raw_message or "banned" in exc_info.value.raw_message


def test_retry_raises_BinanceIPBan_on_1003_without_timestamp():
    """-1003 without parsable timestamp should still raise BinanceIPBan, not retry."""
    from tradingagents.execution.exchange import BinanceIPBan, ExchangeClient

    err = _make_api_exc("APIError(code=-1003): Too many requests, no timestamp here")

    with pytest.raises(BinanceIPBan) as exc_info:
        ExchangeClient._retry(lambda: (_ for _ in ()).throw(err))

    assert exc_info.value.until_ms == 0


def test_retry_does_not_retry_on_1003():
    """Ban must short-circuit immediately — retrying extends the ban."""
    from tradingagents.execution.exchange import BinanceIPBan, ExchangeClient

    future_ms = int(time.time() * 1000) + 60_000
    err = _make_api_exc(f"APIError(code=-1003): banned until {future_ms}")
    fn = MagicMock(side_effect=err)

    with pytest.raises(BinanceIPBan):
        ExchangeClient._retry(fn)

    assert fn.call_count == 1  # zero retries


def test_retry_still_retries_plain_429(monkeypatch):
    """Non-ban 429 errors keep the existing exponential-backoff retry behavior."""
    from tradingagents.execution.exchange import ExchangeClient

    err = _make_api_exc("APIError: rate limit", status_code=429, code=-1015)
    fn = MagicMock(side_effect=[err, err, "ok"])
    monkeypatch.setattr("tradingagents.execution.exchange.time.sleep", lambda *_: None)

    result = ExchangeClient._retry(fn)
    assert result == "ok"
    assert fn.call_count == 3
