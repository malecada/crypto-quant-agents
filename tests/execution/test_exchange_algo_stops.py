"""Stops are Binance CONDITIONAL/algo orders (algoId, not orderId), invisible to
the regular open-orders/cancel endpoints. These tests pin the algo-aware
behavior of place_stop_loss / list_open_stops / cancel_order.
"""
import pytest
from binance.exceptions import BinanceAPIException

from tradingagents.execution.exchange import ExchangeClient


def _bare_client(monkeypatch_client):
    ex = ExchangeClient.__new__(ExchangeClient)
    ex._client = monkeypatch_client
    ex._retry = lambda fn, *a, **k: fn(*a, **k)
    ex.round_price = lambda s, p: float(p)
    ex.round_quantity = lambda s, q: float(q)
    return ex


class _Client:
    def __init__(self, **methods):
        for k, v in methods.items():
            setattr(self, k, v)


def test_place_stop_loss_accepts_algoid():
    ex = _bare_client(_Client(
        futures_create_order=lambda **kw: {"algoId": 1000000091736019, "algoStatus": "NEW"},
    ))
    r = ex.place_stop_loss("BTCUSDT", 0.002, 70000.0, "SELL")
    assert r["algoId"] == 1000000091736019


def test_place_stop_loss_raises_on_empty_response():
    """Silent rejection (no orderId AND no algoId) must raise -> UNPROTECTED."""
    ex = _bare_client(_Client(
        futures_create_order=lambda **kw: {"orderId": None, "status": None},
    ))
    with pytest.raises(RuntimeError, match="no orderId/algoId"):
        ex.place_stop_loss("BTCUSDT", 0.002, 70000.0, "SELL")


def test_list_open_stops_merges_regular_and_algo():
    ex = _bare_client(_Client(
        futures_get_open_orders=lambda symbol: [
            {"type": "STOP_MARKET", "orderId": 1, "stopPrice": "100", "origQty": "0.5"},
            {"type": "LIMIT", "orderId": 2},
        ],
        futures_get_open_algo_orders=lambda: [
            {"symbol": "BTCUSDT", "orderType": "STOP_MARKET", "algoId": 999,
             "triggerPrice": "95", "quantity": "0.5"},
            {"symbol": "ETHUSDT", "orderType": "STOP_MARKET", "algoId": 888,
             "triggerPrice": "1", "quantity": "1"},
        ],
    ))
    stops = ex.list_open_stops("BTCUSDT")
    ids = {s["orderId"] for s in stops}
    assert ids == {1, 999}  # regular + this-symbol algo only (ETH excluded)
    algo = next(s for s in stops if s["orderId"] == 999)
    assert algo["stopPrice"] == "95" and algo["origQty"] == "0.5" and algo["_algo"] is True


def test_cancel_order_falls_back_to_algo():
    calls = {}

    def _reg_cancel(symbol, orderId):
        raise BinanceAPIException(
            type("R", (), {"text": ""})(), 400,
            '{"code":-2013,"msg":"Order does not exist."}',
        )

    def _algo_cancel(algoId):
        calls["algo"] = algoId
        return {"algoId": algoId, "status": "CANCELED"}

    ex = _bare_client(_Client(
        futures_cancel_order=_reg_cancel,
        futures_cancel_algo_order=_algo_cancel,
    ))
    r = ex.cancel_order("BTCUSDT", 999)
    assert r["status"] == "CANCELED" and calls["algo"] == 999


def test_cancel_order_reraises_unexpected():
    def _reg_cancel(symbol, orderId):
        raise BinanceAPIException(
            type("R", (), {"text": ""})(), 400,
            '{"code":-1111,"msg":"Precision over the maximum."}',
        )
    ex = _bare_client(_Client(futures_cancel_order=_reg_cancel))
    with pytest.raises(BinanceAPIException):
        ex.cancel_order("BTCUSDT", 5)
