"""Unit tests for unknown-execution reconciliation in place_market_order /
place_stop_loss.

Two unknown-execution events are handled identically: Binance -1007 ("timeout
waiting for backend; execution status unknown") and a network/request error
(ConnectionError / ReadTimeout / BinanceRequestException) — in both cases the
order may or may not have landed, so we must not blind-retry.

Reconciliation is DETERMINISTIC: every create_order is tagged with a unique
newClientOrderId, and after an unknown-execution event we query that exact
order via futures_get_order(origClientOrderId=...). This is partial-fill safe
(executedQty is cumulative on the order) and cannot mis-match an unrelated
same-side/same-qty order, unlike the old side+qty userTrades heuristic.

State machine:
- FILLED / PARTIALLY_FILLED (executedQty>0) -> success (synth envelope)
- NEW (resting): MARKET -> cancel + raise; STOP_MARKET -> that's the goal
- order-not-found / CANCELED / REJECTED -> not_placed -> retry once
- second unknown event -> raise BinanceOrderTimeoutUnknown
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def _make_api_exc(message: str, code: int = -1007, status_code: int = 504):
    from binance.exceptions import BinanceAPIException

    exc = BinanceAPIException.__new__(BinanceAPIException)
    exc.status_code = status_code
    exc.code = code
    exc.message = message
    exc.response = None
    exc.request = None
    return exc


def _order_not_found_exc():
    """python-binance get_order on a never-placed order -> -2013."""
    return _make_api_exc("APIError(code=-2013): Order does not exist",
                         code=-2013, status_code=400)


@pytest.fixture
def ex(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setattr("tradingagents.execution.exchange.time.sleep",
                        lambda *_: None)
    from tradingagents.execution.exchange import ExchangeClient

    client = ExchangeClient(api_key="k", api_secret="s", testnet=True)
    client._client = MagicMock()
    client._symbol_info_cache["BTCUSDT"] = {
        "symbol": "BTCUSDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01"},
        ],
    }
    return client


def _sent_client_order_id(ex):
    """The newClientOrderId passed to the first create_order call."""
    return ex._client.futures_create_order.call_args_list[0].kwargs["newClientOrderId"]


def test_market_order_tags_unique_client_order_id(ex):
    ex._client.futures_create_order.return_value = {
        "orderId": 1, "status": "FILLED", "symbol": "BTCUSDT",
    }
    ex.place_market_order("BTCUSDT", "BUY", 0.01)
    coid = ex._client.futures_create_order.call_args.kwargs.get("newClientOrderId")
    assert coid and isinstance(coid, str) and len(coid) <= 36


def test_timeout_then_fill_found_returns_synthesized_envelope(ex):
    """-1007 raised, but the tagged order actually FILLED -> treat as success."""
    timeout = _make_api_exc("APIError(code=-1007): Timeout waiting for backend")
    ex._client.futures_create_order.side_effect = timeout
    ex._client.futures_get_order.return_value = {
        "orderId": 999, "clientOrderId": "ignored", "side": "SELL",
        "status": "FILLED", "origQty": "0.0168", "executedQty": "0.0168",
        "avgPrice": "77000.0",
    }

    result = ex.place_market_order("BTCUSDT", "SELL", 0.0168)
    assert result["status"] == "FILLED"
    assert result["orderId"] == 999
    assert result["_reconciled"] is True
    assert ex._client.futures_create_order.call_count == 1
    # reconciliation queried the exact order we tagged
    assert ex._client.futures_get_order.call_args.kwargs["origClientOrderId"] == _sent_client_order_id(ex)


def test_timeout_then_partial_fill_is_recorded_not_double_traded(ex):
    """R2: a partial / multi-level fill (executedQty < origQty across several
    trades) must be recognized as a real position from the order's cumulative
    executedQty — NOT missed (which used to cause a retry -> double position)."""
    timeout = _make_api_exc("APIError(code=-1007): timeout")
    ex._client.futures_create_order.side_effect = timeout
    ex._client.futures_get_order.return_value = {
        "orderId": 321, "side": "BUY", "status": "PARTIALLY_FILLED",
        "origQty": "0.0168", "executedQty": "0.0100", "avgPrice": "77050.0",
    }

    result = ex.place_market_order("BTCUSDT", "BUY", 0.0168)
    assert result["status"] == "FILLED"
    assert result["executedQty"] == "0.01"
    # critical: did NOT retry — no second order on top of the partial position
    assert ex._client.futures_create_order.call_count == 1


def test_timeout_then_open_order_cancels_and_raises(ex):
    from tradingagents.execution.exchange import BinanceOrderTimeoutUnknown

    timeout = _make_api_exc("APIError(code=-1007): timeout")
    ex._client.futures_create_order.side_effect = timeout
    ex._client.futures_get_order.return_value = {
        "orderId": 1234, "side": "SELL", "type": "MARKET",
        "status": "NEW", "origQty": "0.0168", "executedQty": "0",
    }

    with pytest.raises(BinanceOrderTimeoutUnknown) as exc_info:
        ex.place_market_order("BTCUSDT", "SELL", 0.0168)

    assert exc_info.value.state == "open_order_canceled"
    ex._client.futures_cancel_order.assert_called_once()
    assert ex._client.futures_cancel_order.call_args.kwargs["orderId"] == 1234


def test_timeout_not_placed_retries_once_and_succeeds(ex):
    """-1007, order-not-found -> not_placed -> retry succeeds."""
    timeout = _make_api_exc("APIError(code=-1007): timeout")
    success = {"orderId": 5555, "symbol": "BTCUSDT", "status": "FILLED",
               "side": "SELL", "origQty": "0.0168", "avgPrice": "77100"}
    ex._client.futures_create_order.side_effect = [timeout, success]
    ex._client.futures_get_order.side_effect = _order_not_found_exc()

    result = ex.place_market_order("BTCUSDT", "SELL", 0.0168)
    assert result["orderId"] == 5555
    assert ex._client.futures_create_order.call_count == 2


def test_double_1007_raises_BinanceOrderTimeoutUnknown(ex):
    from tradingagents.execution.exchange import BinanceOrderTimeoutUnknown

    timeout = _make_api_exc("APIError(code=-1007): timeout")
    ex._client.futures_create_order.side_effect = [timeout, timeout]
    ex._client.futures_get_order.side_effect = _order_not_found_exc()

    with pytest.raises(BinanceOrderTimeoutUnknown) as exc_info:
        ex.place_market_order("BTCUSDT", "SELL", 0.0168)

    assert exc_info.value.state == "unknown"
    assert ex._client.futures_create_order.call_count == 2


def test_non_1007_error_does_not_trigger_reconcile(ex):
    """A regular Binance error (e.g. -2010 insufficient balance) must NOT
    reconcile — only unknown-execution events do."""
    err = _make_api_exc("APIError(code=-2010): insufficient balance",
                        code=-2010, status_code=400)
    ex._client.futures_create_order.side_effect = err

    from binance.exceptions import BinanceAPIException
    with pytest.raises(BinanceAPIException):
        ex.place_market_order("BTCUSDT", "SELL", 0.0168)

    ex._client.futures_get_order.assert_not_called()


def test_network_error_triggers_reconcile_and_finds_fill(ex):
    """R3: a network error (ConnectionError) on create is unknown-execution —
    must reconcile, not silently FAIL leaving a naked unrecorded position."""
    import requests
    ex._client.futures_create_order.side_effect = requests.exceptions.ConnectionError("conn reset")
    ex._client.futures_get_order.return_value = {
        "orderId": 888, "side": "BUY", "status": "FILLED",
        "origQty": "0.01", "executedQty": "0.01", "avgPrice": "77000",
    }

    result = ex.place_market_order("BTCUSDT", "BUY", 0.01)
    assert result["status"] == "FILLED"
    assert result["orderId"] == 888
    assert result["_reconciled"] is True


def test_network_error_not_placed_retries_once(ex):
    """R3: a ReadTimeout where reconcile finds nothing -> safe retry."""
    import requests
    success = {"orderId": 4242, "status": "FILLED", "symbol": "BTCUSDT",
               "side": "BUY", "origQty": "0.01"}
    ex._client.futures_create_order.side_effect = [
        requests.exceptions.ReadTimeout("read timed out"), success,
    ]
    ex._client.futures_get_order.side_effect = _order_not_found_exc()

    result = ex.place_market_order("BTCUSDT", "BUY", 0.01)
    assert result["orderId"] == 4242
    assert ex._client.futures_create_order.call_count == 2


def test_place_stop_loss_timeout_with_resting_order_is_success(ex):
    """STOP_MARKET: a resting (NEW) stop after -1007 IS the desired end state —
    return it, do NOT cancel (that would leave the position unprotected)."""
    timeout = _make_api_exc("APIError(code=-1007): timeout")
    ex._client.futures_create_order.side_effect = timeout
    ex._client.futures_get_order.return_value = {
        "orderId": 7777, "side": "SELL", "type": "STOP_MARKET",
        "status": "NEW", "origQty": "0.0168", "executedQty": "0",
    }

    result = ex.place_stop_loss("BTCUSDT", 0.0168, stop_price=75000, side="SELL")
    assert result["orderId"] == 7777
    ex._client.futures_cancel_order.assert_not_called()
