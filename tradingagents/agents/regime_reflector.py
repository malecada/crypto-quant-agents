"""Regime Reflector (FS-ReasoningAgent re-weighting, Tier A5).

Reads the Layer 1 regime label and emits a re-weighting note that
the Modulator agent injects into its prompt. The reflector itself is
deterministic — no LLM call — because the weighting policy is a stated
prior derived from the literature, not something we want a sampling
LLM to flip on us mid-run. The narrative *justification* uses the
chosen weights so the modulator's final narrative is auditable.

Bull → upweight Factual (FS-ReasoningAgent finding: factual reasoning
dominates trend-following regimes).
Sideways → upweight Subjective (narrative-driven phases).
Bear → upweight Factual (defensive on-chain / derivatives matter most).
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)

# (factual_weight, subjective_weight) per regime
_REGIME_WEIGHTS: dict[str, tuple[float, float]] = {
    "bull": (0.7, 0.3),
    "sideways": (0.3, 0.7),
    "bear": (0.7, 0.3),
}


def _build_note(regime: str, factual_w: float, subjective_w: float) -> str:
    return (
        f"Detected regime: {regime}. "
        f"Factual weight: {factual_w:.2f}, Subjective weight: {subjective_w:.2f}. "
        f"In {regime} regimes, "
        + (
            "factual / on-chain / derivatives signals dominate; "
            "narrative is mostly noise unless extreme."
            if factual_w >= subjective_w
            else "narrative drives short-term returns; "
            "factual signals matter mostly as risk gates."
        )
    )


def create_regime_reflector():
    """Deterministic reflector — does not need an LLM handle."""

    def reflector_node(state, name):
        qs = state.get("quant_signal") or {}
        regime = (
            qs.get("regime") if isinstance(qs, dict) else getattr(qs, "regime", None)
        ) or "sideways"
        factual_w, subjective_w = _REGIME_WEIGHTS.get(regime, (0.5, 0.5))
        note = _build_note(regime, factual_w, subjective_w)
        return {
            "regime_reflector_note": note,
            "factual_weight": factual_w,
            "subjective_weight": subjective_w,
            "sender": name,
        }

    return functools.partial(reflector_node, name="RegimeReflector")
