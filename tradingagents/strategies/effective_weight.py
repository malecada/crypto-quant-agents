"""Effective-weight formula — composes Layer 1 quant with Layer 2 LLM signal.

The single asset-agnostic policy that replaces a per-coin router. Every
coin runs through this formula; ``effective_weight ∈ [0, 1]`` collapses
to 0 when the LLM signal is uninformative or vetoed (recovers pure
quant) and approaches 1 only when regime, calibration, and history all
agree the LLM adds value.

Inputs and how they shape the weight:

* ``regime`` — regime-conditional weighting band per the literature
  (FINSABER, Springer 2026): bull ``(0.2, 0.3)``, sideways
  ``(0.6, 0.8)``, bear ``(0.4, 0.4)``. The mid-point of the band is
  the structural prior.

* ``llm_uncertainty`` — std-dev across N=5 Self-MoA samples. High
  disagreement → LLM is unreliable → dampen toward 0.

* ``rolling_llm_edge`` — rolling Sharpe of (quant × multiplier) minus
  pure-quant returns over the last ``rolling_edge_window_days``. None on
  cold start. Negative edge → coin's LLM signal has been hurting →
  dampen. Positive edge → amplify.

* ``unlock_flag`` — hard veto: if an insider unlock is imminent (T-30
  to T+14 per Tokenomist 2023), the LLM's directional view is overruled
  by the deterministic event signal. ``effective_weight = 0`` →
  pure-quant position.

This is a pure function with no I/O so it remains unit-testable in
isolation and the same calculation can be replayed offline for
ablations.
"""

from __future__ import annotations

import math
from typing import Optional

from tradingagents.strategies.contracts import RegimeLabel

DEFAULT_REGIME_WEIGHT: dict[RegimeLabel, tuple[float, float]] = {
    "bull": (0.2, 0.3),
    "sideways": (0.6, 0.8),
    "bear": (0.4, 0.4),
}


def _sigmoid(x: float, k: float = 1.0) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-k * x)
        return 1.0 / (1.0 + z)
    z = math.exp(k * x)
    return z / (1.0 + z)


def _uncertainty_dampener(uncertainty: float, k: float = 1.0) -> float:
    """Multiplicative scale ∈ (0, 1] decreasing in uncertainty.

    ``uncertainty=0`` → 1.0 (full weight). ``uncertainty=0.5`` (the std
    of a 5-sample set spread across [0, 1.5]) → ~0.6. ``k`` controls
    steepness.
    """
    return math.exp(-k * max(0.0, uncertainty))


def _edge_dampener(edge: Optional[float], k: float = 1.0) -> float:
    """Scale ∈ [0, 1] from rolling LLM edge.

    Cold start (``edge=None``) returns 1.0 — no historical evidence to
    discount the structural regime prior. Negative edge → fast collapse
    toward 0. Positive edge → asymptote to 1.
    """
    if edge is None:
        return 1.0
    return _sigmoid(float(edge), k)


def compute_effective_weight(
    regime: RegimeLabel,
    llm_uncertainty: float,
    rolling_llm_edge: Optional[float],
    unlock_flag: bool,
    regime_weighting: dict[RegimeLabel, tuple[float, float]] = DEFAULT_REGIME_WEIGHT,
    uncertainty_dampener_k: float = 1.0,
    edge_dampener_k: float = 1.0,
) -> float:
    """Asset-agnostic LLM influence weight ∈ [0, 1].

    BTC's emergent quant-dominant behavior shows up here when its
    rolling edge is negative or its uncertainty is high — not because
    a config knob says ``bitcoin: quant_only``.
    """
    if unlock_flag:
        return 0.0

    band = regime_weighting.get(regime, (0.5, 0.5))
    base = 0.5 * (band[0] + band[1])

    w = (
        base
        * _uncertainty_dampener(llm_uncertainty, uncertainty_dampener_k)
        * _edge_dampener(rolling_llm_edge, edge_dampener_k)
    )
    return float(min(1.0, max(0.0, w)))
