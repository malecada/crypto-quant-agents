"""Atomic, monotonic protective-stop replacement (R1 + R5).

The live cycle re-derives a protective stop for the net position every time it
trades a coin. Two failure modes the naive cancel-then-place sequence created:

R1 (naked window): cancelling the existing stop *before* placing the new one
means a transient placement failure leaves the position with no stop for up to
a full cycle (~24h). ``arm_stop_loss`` places the new stop FIRST and only
cancels the old one once the new one is confirmed; if the new placement fails
it keeps the old stop, so the position is never left naked by a failed swap.

R5 (loosening ratchet): re-placing the stop every cycle re-anchors it to the
current price. On a long that has drifted down, the fresh stop sits *below* the
previous one — a looser stop. ``arm_stop_loss`` is monotonic: if an existing
stop of the same size is already at least as protective (>= price for a long,
<= for a short), it is kept untouched.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_QTY_TOL = 1e-6


def arm_stop_loss(
    ex, *, symbol: str, net_position: float,
    stop_price: float, stop_side: str,
) -> tuple[str | None, str]:
    """Ensure ``symbol`` has a protective stop for ``net_position``.

    Returns ``(stop_id, status)``:
    - ``(id, "EXECUTED")``: a protective stop (new or kept) covers the position.
    - ``(None, "UNPROTECTED")``: no stop could be placed and none pre-existed.

    ``ex`` must provide ``list_open_stops(symbol)`` -> list of open STOP_MARKET
    dicts (``orderId`` / ``stopPrice`` / ``origQty``), ``place_stop_loss(...)``,
    and ``cancel_order(symbol, order_id)``.
    """
    qty = abs(net_position)
    is_long = net_position > 0

    try:
        existing = ex.list_open_stops(symbol)
    except Exception as e:  # noqa: BLE001 — never let a query failure block arming
        logger.warning("list_open_stops failed for %s: %s", symbol, e)
        existing = []

    # R5: keep an existing same-size stop that is already at least as protective.
    for o in existing:
        try:
            o_price = float(o.get("stopPrice", 0) or 0)
            o_qty = float(o.get("origQty", 0) or 0)
        except (TypeError, ValueError):
            continue
        if abs(o_qty - qty) > _QTY_TOL:
            continue
        tighter_or_equal = o_price >= stop_price if is_long else o_price <= stop_price
        if tighter_or_equal and o_price > 0:
            logger.info(
                "Keeping existing %s stop %s @ %.8f (>= candidate %.8f); no churn",
                symbol, o.get("orderId"), o_price, stop_price,
            )
            return (str(o.get("orderId")), "EXECUTED")

    old_ids = [o.get("orderId") for o in existing if o.get("orderId") is not None]

    # R1: place the NEW stop before touching the old one.
    try:
        stop = ex.place_stop_loss(symbol, qty, stop_price, stop_side)
    except Exception as e:  # noqa: BLE001
        if old_ids:
            logger.error(
                "New stop for %s failed (%s); keeping prior stop(s) %s — "
                "position remains protected (size may be stale)",
                symbol, e, old_ids,
            )
            return (str(old_ids[0]), "EXECUTED")
        logger.error("Stop placement for %s failed and no prior stop exists: %s",
                     symbol, e)
        return (None, "UNPROTECTED")

    new_id = str(stop.get("orderId") or stop.get("algoId", ""))

    # New stop confirmed — now cancel the superseded old stop(s) by id.
    for oid in old_ids:
        try:
            ex.cancel_order(symbol, oid)
        except Exception as e:  # noqa: BLE001 — a stale resting stop is reduceOnly, harmless
            logger.warning("Failed to cancel superseded stop %s for %s: %s",
                           oid, symbol, e)
    return (new_id, "EXECUTED")
