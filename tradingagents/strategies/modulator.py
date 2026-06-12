"""Layer 2 modulator — composes Layer 1 quant with LLM signal.

The graph-side LLM modulator agent (``tradingagents/agents/modulator.py``)
calls into ``apply_modulator`` once it has the LLM's raw multiplier and
uncertainty, so this module stays purely deterministic and unit-testable.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from tradingagents.strategies.contracts import (
    ModulatedPosition,
    ModulatorOutput,
    QuantSignal,
)
from tradingagents.strategies.effective_weight import compute_effective_weight
from tradingagents.strategies.rolling_edge import query_rolling_edge


def apply_modulator(
    quant_signal: QuantSignal,
    llm_output: ModulatorOutput,
    rolling_llm_edge: Optional[float] = None,
    config: Optional[dict] = None,
) -> ModulatedPosition:
    """Compose Layer 1 quant + Layer 2 LLM into a ``ModulatedPosition``.

    If ``rolling_llm_edge`` is ``None`` we look it up from the parquet
    store keyed on (coin, as_of_date). The unlock veto comes from
    ``quant_signal.deterministic_signals['unlock_flag']`` which Layer 1
    populates via ``deterministic_signals.compute_deterministic_pack``.
    """
    cfg = config or {}
    unlock_flag = bool(
        quant_signal.deterministic_signals.get("unlock_flag", False)
    )
    if rolling_llm_edge is None:
        rolling_llm_edge = query_rolling_edge(
            coin=quant_signal.coin,
            as_of_date=pd.to_datetime(quant_signal.as_of_date),
        )

    effective_weight = compute_effective_weight(
        regime=quant_signal.regime,
        llm_uncertainty=llm_output.uncertainty,
        rolling_llm_edge=rolling_llm_edge,
        unlock_flag=unlock_flag,
        regime_weighting={
            k: tuple(v) for k, v in
            cfg.get("regime_weighting", {}).items()
        } or None,
        uncertainty_dampener_k=float(cfg.get("uncertainty_dampener_k", 1.0)),
        edge_dampener_k=float(cfg.get("edge_dampener_k", 1.0)),
    ) if cfg.get("regime_weighting") else compute_effective_weight(
        regime=quant_signal.regime,
        llm_uncertainty=llm_output.uncertainty,
        rolling_llm_edge=rolling_llm_edge,
        unlock_flag=unlock_flag,
        uncertainty_dampener_k=float(cfg.get("uncertainty_dampener_k", 1.0)),
        edge_dampener_k=float(cfg.get("edge_dampener_k", 1.0)),
    )

    position = quant_signal.magnitude * (
        1.0 + effective_weight * (llm_output.multiplier - 1.0)
    )

    return ModulatedPosition(
        coin=quant_signal.coin,
        quant_direction=quant_signal.direction,
        quant_magnitude=quant_signal.magnitude,
        llm_multiplier=llm_output.multiplier,
        llm_confidence=llm_output.confidence,
        llm_uncertainty=llm_output.uncertainty,
        effective_weight=effective_weight,
        position=position,
        narrative=llm_output.narrative,
        regime=quant_signal.regime,
        unlock_flag=unlock_flag,
        rolling_llm_edge=rolling_llm_edge,
    )
