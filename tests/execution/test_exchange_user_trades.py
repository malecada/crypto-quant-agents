"""Unit tests for `ExchangeClient.get_user_trades` — used by the live runner
after `place_market_order` to backfill `trades.fees` + `trades.pnl` in the
journal (V5 parity gap #2).
"""
from __future__ import annotations

from unittest.mock import MagicMock


def _bare_client() -> "ExchangeClient":  # type: ignore[name-defined]
    from tradingagents.execution.exchange import ExchangeClient
    ex = ExchangeClient.__new__(ExchangeClient)
    ex._client = MagicMock()
    return ex


def test_get_user_trades_queries_futures_account_trades_with_symbol_orderid():
    """Returns the raw fills list from Binance Futures /fapi/v1/userTrades."""
    ex = _bare_client()
    ex._client.futures_account_trades.return_value = [
        {"symbol": "BTCUSDT", "orderId": 12345, "commission": "0.078",
         "commissionAsset": "USDT", "realizedPnl": "1.23", "qty": "0.001",
         "price": "77000.0", "side": "BUY"},
        {"symbol": "BTCUSDT", "orderId": 12345, "commission": "0.039",
         "commissionAsset": "USDT", "realizedPnl": "0.62", "qty": "0.0005",
         "price": "77010.0", "side": "BUY"},
    ]

    fills = ex.get_user_trades("BTCUSDT", "12345")

    ex._client.futures_account_trades.assert_called_once_with(
        symbol="BTCUSDT", orderId=12345,
    )
    assert len(fills) == 2
    assert fills[0]["commission"] == "0.078"
    assert fills[1]["realizedPnl"] == "0.62"


def test_get_user_trades_returns_empty_list_when_no_fills():
    ex = _bare_client()
    ex._client.futures_account_trades.return_value = []

    fills = ex.get_user_trades("BTCUSDT", 99999)

    assert fills == []
    ex._client.futures_account_trades.assert_called_once_with(
        symbol="BTCUSDT", orderId=99999,
    )


def test_get_user_trades_accepts_string_or_int_order_id():
    """Runner stores order_id as str in the journal; the Binance API expects
    int. The wrapper must coerce so callers don't have to."""
    ex = _bare_client()
    ex._client.futures_account_trades.return_value = []

    ex.get_user_trades("ETHUSDT", "555")

    ex._client.futures_account_trades.assert_called_once_with(
        symbol="ETHUSDT", orderId=555,
    )
