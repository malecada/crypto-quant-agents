"""Derived performance statistics for the monitor UI.

All functions are pure: they take plain rows (dicts/floats) and return
numbers or plain lists. No DB access here.
"""
from __future__ import annotations

import math

# Crypto trades every day -> annualize daily returns with sqrt(365).
_ANNUALIZATION = math.sqrt(365.0)


def max_drawdown(values: list[float]) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.25).
    Returns 0.0 for a series that never declines or is too short."""
    if len(values) < 2:
        return 0.0
    peak = values[0]
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, (v - peak) / peak)
    return worst


def sharpe(values: list[float]) -> float:
    """Annualized Sharpe ratio of an equity series (risk-free rate 0).
    Returns 0.0 if the series is too short or has zero variance."""
    if len(values) < 2:
        return 0.0
    returns = [
        (values[i] - values[i - 1]) / values[i - 1]
        for i in range(1, len(values))
        if values[i - 1] != 0
    ]
    if len(returns) < 2:
        return 0.0
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    return (mean / std) * _ANNUALIZATION


def cumulative_pnl(trades: list[dict]) -> float:
    """Sum of realized trade PnL, ignoring None values."""
    return sum(t["pnl"] for t in trades if t.get("pnl") is not None)


def drawdown_series(equity: list[dict]) -> list[dict]:
    """Running drawdown per equity point as {ts, value} (value <= 0)."""
    out: list[dict] = []
    peak = float("-inf")
    for pt in equity:
        v = pt["value"]
        peak = max(peak, v)
        dd = (v - peak) / peak if peak > 0 else 0.0
        out.append({"ts": pt["ts"], "value": round(dd, 6)})
    return out


def rolling_sharpe(equity: list[dict], window: int = 30) -> list[dict]:
    """Rolling annualized Sharpe over the trailing ``window`` returns.

    Emits one {ts, value} per point starting at index ``window`` (needs
    ``window`` returns => window+1 equity points). Empty when history is
    shorter — the UI hides the pane until enough cycles exist.
    """
    values = [p["value"] for p in equity]
    out: list[dict] = []
    for i in range(window, len(values)):
        out.append({
            "ts": equity[i]["ts"],
            "value": round(sharpe(values[i - window: i + 1]), 4),
        })
    return out


def equity_series(
    snapshots: list[dict], trades: list[dict], start_capital: float
) -> list[dict]:
    """Equity curve as a list of {ts, value} dicts, chronological.

    Primary source is ``portfolio_snapshots.total_value``. When no snapshots
    exist, the curve is reconstructed from cumulative realized trade PnL,
    prepended with the starting-capital point.
    """
    if snapshots:
        return [
            {"ts": s["ts"], "value": s["total_value"]}
            for s in snapshots
            if s.get("total_value") is not None
        ]
    if not trades:
        return []
    # Trades arrive newest-first from db.all_trades; reverse to chronological.
    chrono = list(reversed(trades))
    series = [{"ts": "start", "value": start_capital}]
    running = start_capital
    for t in chrono:
        running += t["pnl"] if t.get("pnl") is not None else 0.0
        series.append({"ts": t.get("cycle_id", ""), "value": running})
    return series
