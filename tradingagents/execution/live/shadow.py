"""Shadow replay: re-runs V2 sizing on the same input data.

Both live and shadow use tradingagents.strategies.v2_sizing as the single
source of truth, so signal_agreement should be 100%. Any divergence
indicates input mutation (e.g. stale cache, unit conversion, type coercion)
between the live decision path and this shadow.
"""
from __future__ import annotations

from dataclasses import dataclass

from tradingagents.execution.live.sizer import compute_size


@dataclass
class ShadowDecision:
    coin: str
    signal: int
    size: float


def compute_shadow_decision(**kwargs) -> ShadowDecision:
    res = compute_size(**kwargs)
    return ShadowDecision(coin=res.coin, signal=res.signal, size=res.final_size_notional)
