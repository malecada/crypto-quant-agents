"""Pydantic contracts for V3 quant pipeline.

These models are the only data types crossing module boundaries inside the V3
sub-package. Look-ahead-safety is enforced at construction (``as_of`` is required).
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

RegimeLabel = Literal["bull", "sideways", "bear"]


class RegimeState(BaseModel):
    """Output of the regime detector ensemble (HMM-3 + BOCPD + Hurst)."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    label: RegimeLabel
    confidence: float = Field(ge=0.0, le=1.0)
    hurst: float = Field(ge=0.0, le=1.0)
    changepoint_alert: bool
    posterior: dict[str, float]

    @field_validator("posterior")
    @classmethod
    def _posterior_keys(cls, v: dict[str, float]) -> dict[str, float]:
        if set(v.keys()) != {"bull", "sideways", "bear"}:
            raise ValueError("posterior must have keys {bull, sideways, bear}")
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"posterior must sum to 1.0, got {total}")
        return v


class FeatureBundle(BaseModel):
    """Per-bar feature vector composed of price, microstructure, derivatives blocks."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    coin: str
    as_of: pd.Timestamp
    price_features: dict[str, float]
    microstructure_features: dict[str, float]
    derivatives_features: dict[str, float]


class V3Signal(BaseModel):
    """Signal emitted by V3 quant pipeline; consumed by sizing layer."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    coin: str
    as_of: pd.Timestamp
    direction: Literal[-1, 0, 1]
    confidence: float = Field(ge=0.0, le=1.0)
    horizon: int = Field(gt=0)
    regime: RegimeState
