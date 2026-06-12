"""ExchangeClient.get_open_positions — single-call live position snapshot."""
from __future__ import annotations

from unittest.mock import MagicMock

from tradingagents.execution.exchange import ExchangeClient


def _bare_client() -> ExchangeClient:
    """An ExchangeClient with no real Binance connection (bypasses __init__)."""
    ex = ExchangeClient.__new__(ExchangeClient)
    ex._client = MagicMock()
    ex._retry = lambda fn, **kw: fn(**kw)
    return ex


def test_get_open_positions_filters_flat_and_carries_signed_usd():
    ex = _bare_client()
    ex._client.futures_position_information.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "0.001",
         "markPrice": "63000.0", "notional": "63.0"},
        {"symbol": "SOLUSDT", "positionAmt": "-2.0",
         "markPrice": "66.0", "notional": "-132.0"},
        {"symbol": "ETHUSDT", "positionAmt": "0.0",
         "markPrice": "3000.0", "notional": "0.0"},
    ]
    out = ex.get_open_positions()

    by = {p["symbol"]: p for p in out}
    assert set(by) == {"BTCUSDT", "SOLUSDT"}  # flat ETH dropped
    assert by["BTCUSDT"]["qty"] == 0.001
    assert by["BTCUSDT"]["usd"] == 63.0
    assert by["SOLUSDT"]["qty"] == -2.0       # short preserved
    assert by["SOLUSDT"]["usd"] == -132.0     # signed


def test_get_open_positions_falls_back_to_mark_price_when_no_notional():
    ex = _bare_client()
    ex._client.futures_position_information.return_value = [
        {"symbol": "BTCUSDT", "positionAmt": "2.0", "markPrice": "100.0"},
    ]
    out = ex.get_open_positions()
    assert out[0]["usd"] == 200.0
