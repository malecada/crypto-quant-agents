"""V3 configuration — single source of truth for all knobs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

RegimeMode = Literal["trending", "mean_reverting", "uncertain"]


@dataclass(frozen=True)
class V3Config:
    target_annual_vol: float = 0.15
    max_leverage: float = 2.0
    horizons: tuple[int, ...] = (3, 7, 14, 21)
    cpcv_n_groups: int = 8
    cpcv_test_groups: int = 2
    embargo_bars: int = 14
    min_train_bars: int = 252

    hurst_lookback: int = 63
    hurst_trend_threshold: float = 0.55
    hurst_mr_threshold: float = 0.45

    cdap_dd_de_lever: float = 0.05
    cdap_dd_flat: float = 0.10
    cdap_min_regime_confidence: float = 0.6

    microstructure_parquet_dir: Path = field(
        default_factory=lambda: Path("data/microstructure")
    )
    derivatives_parquet_dir: Path = field(
        default_factory=lambda: Path("data/derivatives")
    )
    regime_pickle_dir: Path = field(
        default_factory=lambda: Path("data/checkpoints")
    )

    def horizon_weights(self, mode: RegimeMode) -> dict[int, float]:
        if set(self.horizons) != {3, 7, 14, 21}:
            raise ValueError(
                f"horizon_weights supports only default horizons (3,7,14,21); got {self.horizons}"
            )
        if mode == "trending":
            weights = {3: 0.10, 7: 0.20, 14: 0.35, 21: 0.35}
        elif mode == "mean_reverting":
            weights = {3: 0.35, 7: 0.35, 14: 0.20, 21: 0.10}
        else:
            weights = {3: 0.25, 7: 0.25, 14: 0.25, 21: 0.25}
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
