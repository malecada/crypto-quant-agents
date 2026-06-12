"""R1 + R5 — atomic, monotonic protective-stop replacement.

R1: the runner cancelled the existing stop BEFORE placing the new one, so a
transient placement failure left the position naked for ~24h. arm_stop_loss
places the new stop FIRST and only cancels the old one on success; if the new
placement fails it keeps the old stop (position stays protected).

R5: the stop was re-placed every cycle, re-anchored to the current price — on
a long that drifted down this ratcheted the stop *looser*. arm_stop_loss is
monotonic: if an existing stop of the same size is already at least as
protective, it is kept (no churn, no widening).
"""
import pytest


class FakeExchange:
    """Records place/cancel calls; place can be made to fail."""

    def __init__(self, existing=None, place_fails=False):
        self._existing = existing or []      # list of open stop dicts
        self.place_fails = place_fails
        self.calls = []                      # ordered ("place"|"cancel", payload)

    def list_open_stops(self, symbol):
        return list(self._existing)

    def place_stop_loss(self, symbol, quantity, stop_price, side="SELL"):
        self.calls.append(("place", {"qty": quantity, "stop_price": stop_price, "side": side}))
        if self.place_fails:
            raise RuntimeError("stop placement failed")
        return {"orderId": 9001}

    def cancel_order(self, symbol, order_id):
        self.calls.append(("cancel", {"order_id": order_id}))
        return {"orderId": order_id, "status": "CANCELED"}


def _stop(order_id, stop_price, qty, side="SELL"):
    return {"orderId": order_id, "stopPrice": str(stop_price),
            "origQty": str(qty), "side": side, "type": "STOP_MARKET"}


def test_no_existing_stop_places_new(monkeypatch):
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[])
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "EXECUTED"
    assert stop_id == "9001"
    assert [c[0] for c in ex.calls] == ["place"]  # nothing to cancel


def test_places_new_before_cancelling_old(monkeypatch):
    """Atomic ordering: the new stop is placed first, the old one cancelled
    only after — and only the OLD id is cancelled (not the new)."""
    from tradingagents.execution.live.stops import arm_stop_loss
    # long, old stop looser (lower) than new -> should replace
    ex = FakeExchange(existing=[_stop(1234, 57000.0, 0.5)])
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "EXECUTED"
    assert stop_id == "9001"
    assert [c[0] for c in ex.calls] == ["place", "cancel"]
    assert ex.calls[1][1]["order_id"] == 1234


def test_keep_old_stop_when_new_placement_fails(monkeypatch):
    """R1: placement fails but an old stop exists -> keep it, NOT naked."""
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[_stop(1234, 57000.0, 0.5)], place_fails=True)
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "EXECUTED"          # still protected by old stop
    assert stop_id == "1234"
    assert "cancel" not in [c[0] for c in ex.calls]  # old stop NOT cancelled


def test_unprotected_when_no_old_and_placement_fails(monkeypatch):
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[], place_fails=True)
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "UNPROTECTED"
    assert stop_id is None


def test_monotonic_keep_tighter_existing_long(monkeypatch):
    """R5: existing stop already tighter (higher) for a long, same qty -> keep,
    no churn, no loosening."""
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[_stop(1234, 59000.0, 0.5)])  # tighter than 58000
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "EXECUTED"
    assert stop_id == "1234"
    assert ex.calls == []  # neither placed nor cancelled


def test_monotonic_keep_tighter_existing_short(monkeypatch):
    """Short mirror: protective = lower stop; keep existing if already lower."""
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[_stop(1234, 41000.0, 0.5, side="BUY")])
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=-0.5,
        stop_price=42000.0, stop_side="BUY",
    )
    assert status == "EXECUTED"
    assert stop_id == "1234"
    assert ex.calls == []


def test_replace_when_position_size_changed(monkeypatch):
    """Different qty -> must replace even if price is similar (stop must cover
    the new position size)."""
    from tradingagents.execution.live.stops import arm_stop_loss
    ex = FakeExchange(existing=[_stop(1234, 59000.0, 0.3)])  # tighter but wrong qty
    stop_id, status = arm_stop_loss(
        ex, symbol="BTCUSDT", net_position=0.5,
        stop_price=58000.0, stop_side="SELL",
    )
    assert status == "EXECUTED"
    assert stop_id == "9001"
    assert [c[0] for c in ex.calls] == ["place", "cancel"]
