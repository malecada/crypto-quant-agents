"""Pure trade-analytics aggregation for the monitor UI.

Income records come from ExchangeClient.income_history() (Binance futures
income endpoint); slippage comes from journal trade rows. No I/O here.
"""
from __future__ import annotations


def _to_float(val, default: float = 0.0) -> float:
    """Parse Binance numeric strings defensively; malformed values count 0."""
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def income_summary(records: list[dict]) -> dict:
    """Aggregate Binance income records into the analytics strip payload.

    Win rate = share of profitable records among NONZERO REALIZED_PNL fills
    (zero-pnl rows are position-increase fills, not round trips). None when
    no closing fills exist yet.

    Args:
        records: List of raw income dicts from ExchangeClient.income_history().
            Each dict must contain ``incomeType``, ``income`` (string), and
            ``symbol`` keys as returned by the Binance futures income endpoint.

    Returns:
        Dict with keys:
            realized_pnl_per_coin: sorted mapping of symbol → total PnL.
            realized_pnl_total: sum across all coins.
            fees_total: total COMMISSION income (negative = cost).
            funding_total: total FUNDING_FEE income (negative = cost).
            win_rate: fraction of nonzero REALIZED_PNL records that are
                positive; None when no closing fills present.
            n_closing_fills: count of nonzero REALIZED_PNL records.
    """
    pnl_per_coin: dict[str, float] = {}
    fees = 0.0
    funding = 0.0
    wins = 0
    closing = 0
    for r in records:
        kind = r.get("incomeType")
        amount = _to_float(r.get("income"))
        symbol = r.get("symbol") or "?"
        if kind == "REALIZED_PNL":
            pnl_per_coin[symbol] = pnl_per_coin.get(symbol, 0.0) + amount
            if amount != 0.0:
                closing += 1
                if amount > 0:
                    wins += 1
        elif kind == "COMMISSION":
            fees += amount
        elif kind == "FUNDING_FEE":
            funding += amount
    return {
        "realized_pnl_per_coin": {k: round(v, 4) for k, v in sorted(pnl_per_coin.items())},
        "realized_pnl_total": round(sum(pnl_per_coin.values()), 4),
        "fees_total": round(fees, 4),
        "funding_total": round(funding, 4),
        "win_rate": (wins / closing) if closing else None,
        "n_closing_fills": closing,
    }


def slippage_stats(trades: list[dict]) -> dict:
    """Mean/max/count of journal slippage values (None entries skipped).

    Args:
        trades: List of journal trade row dicts. Each dict may contain a
            ``slippage`` key; entries where the value is ``None`` or the key
            is absent are excluded from the calculation.

    Returns:
        Dict with keys:
            mean: arithmetic mean of available slippage values, or None.
            max: maximum slippage value, or None.
            n: count of non-None slippage values.
    """
    vals = [t["slippage"] for t in trades if t.get("slippage") is not None]
    if not vals:
        return {"mean": None, "max": None, "n": 0}
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "max": round(max(vals), 4),
        "n": len(vals),
    }
