"""Read-only ExchangeClient additions for the monitor (mocked client)."""
from __future__ import annotations

from unittest.mock import MagicMock

from tradingagents.execution.exchange import ExchangeClient


def _client_with(positions=None, income=None):
    ex = ExchangeClient.__new__(ExchangeClient)  # skip __init__/network
    ex._client = MagicMock()
    ex._client.futures_position_information.return_value = positions or []
    ex._client.futures_income_history.return_value = income or []
    return ex


def test_get_position_details_maps_fields():
    ex = _client_with(positions=[
        {"symbol": "BTCUSDT", "positionAmt": "0.05", "entryPrice": "65000",
         "markPrice": "66000", "unRealizedProfit": "50.0",
         "liquidationPrice": "30000", "leverage": "3", "notional": "3300"},
        {"symbol": "ETHUSDT", "positionAmt": "0", "entryPrice": "0",
         "markPrice": "3000", "unRealizedProfit": "0",
         "liquidationPrice": "0", "leverage": "3", "notional": "0"},
    ])
    out = ex.get_position_details()
    assert len(out) == 1
    p = out[0]
    assert p["symbol"] == "BTCUSDT" and p["qty"] == 0.05
    assert p["entry_price"] == 65000.0 and p["mark_price"] == 66000.0
    assert p["upnl"] == 50.0 and p["leverage"] == 3.0
    assert p["liq_price"] == 30000.0 and p["notional"] == 3300.0


def test_get_position_details_notional_fallback():
    ex = _client_with(positions=[
        {"symbol": "BTCUSDT", "positionAmt": "-0.1", "entryPrice": "65000",
         "markPrice": "60000", "unRealizedProfit": "500",
         "liquidationPrice": "90000", "leverage": "2"},  # no notional key
    ])
    assert ex.get_position_details()[0]["notional"] == -6000.0


def test_income_history_passes_filters():
    ex = _client_with(income=[{"incomeType": "REALIZED_PNL", "income": "5"}])
    out = ex.income_history(start_time_ms=123, income_type="REALIZED_PNL")
    assert out == [{"incomeType": "REALIZED_PNL", "income": "5"}]
    ex._client.futures_income_history.assert_called_once_with(
        limit=1000, startTime=123, incomeType="REALIZED_PNL")
