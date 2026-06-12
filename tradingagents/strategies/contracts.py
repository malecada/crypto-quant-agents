from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

RegimeLabel = Literal["bull", "sideways", "bear"]
DirectionLabel = Literal["long", "short", "flat"]


class QuantSignal(BaseModel):
    """Layer 1 output: deterministic quant signal for a single coin/date."""

    coin: str
    direction: DirectionLabel
    magnitude: float = Field(ge=-1.0, le=1.0)
    regime: RegimeLabel
    regime_confidence: float = Field(ge=0.0, le=1.0)
    hurst: float
    deterministic_signals: dict = Field(default_factory=dict)
    as_of_date: str


class ModulatorOutput(BaseModel):
    """Raw LLM modulator output before effective-weight composition."""

    multiplier: float = Field(ge=0.0, le=1.5)
    narrative: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0)


class ModulatedPosition(BaseModel):
    """Layer 2 output: composed position consumed by Layer 3 (PM/execution)."""

    coin: str
    quant_direction: DirectionLabel
    quant_magnitude: float
    llm_multiplier: float = Field(ge=0.0, le=1.5)
    llm_confidence: float = Field(ge=0.0, le=1.0)
    llm_uncertainty: float = Field(ge=0.0)
    effective_weight: float = Field(ge=0.0, le=1.0)
    position: float
    narrative: str
    regime: RegimeLabel
    unlock_flag: bool = False
    rolling_llm_edge: Optional[float] = None
