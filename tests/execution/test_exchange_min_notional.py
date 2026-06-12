"""Unit tests for `ExchangeClient.min_notional` — the MIN_NOTIONAL filter the
live runner uses to skip dust rebalance orders that Binance would reject."""
from __future__ import annotations

from unittest.mock import MagicMock


def _bare_client():  # type: ignore[name-defined]
    from tradingagents.execution.exchange import ExchangeClient
    ex = ExchangeClient.__new__(ExchangeClient)
    ex._client = MagicMock()
    ex._symbol_info_cache = {}
    return ex


def _exchange_info(symbol, filters):
    return {"symbols": [{"symbol": symbol, "filters": filters}]}


def test_min_notional_reads_filter():
    ex = _bare_client()
    ex._client.futures_exchange_info.return_value = _exchange_info(
        "BTCUSDT", [{"filterType": "MIN_NOTIONAL", "notional": "20"}],
    )
    assert ex.min_notional("BTCUSDT") == 20.0


def test_min_notional_defaults_when_filter_absent():
    ex = _bare_client()
    ex._client.futures_exchange_info.return_value = _exchange_info(
        "BTCUSDT", [{"filterType": "LOT_SIZE", "stepSize": "0.001"}],
    )
    assert ex.min_notional("BTCUSDT") == 5.0
