"""Isotonic calibration of LLM verbalized confidences (Tier B5).

Tian et al. ("Just Ask for Calibration", arXiv 2305.14975) shows
verbalized confidences are systematically over-confident on RLHF
models, and a single isotonic regression on a held-out period reduces
ECE by ~50%. We fit one calibrator per coin from the existing
``data/agent_signals_pit_*`` JSONs (raw verbalized confidence paired
with realised forward-return sign), pickle to
``data/checkpoints/isotonic_{coin}.pkl``, and apply at modulator time.

Pure-pandas + sklearn — no LLM calls — so ablations are deterministic.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Optional

import numpy as np
from sklearn.isotonic import IsotonicRegression

logger = logging.getLogger(__name__)


class IsotonicCalibrator:
    """Per-coin isotonic remap of raw confidence ∈ [0, 1] to calibrated."""

    def __init__(self):
        self._iso: Optional[IsotonicRegression] = None
        self.n_train: int = 0
        self.coin: Optional[str] = None

    def fit(
        self,
        raw_confidences: np.ndarray,
        realised_outcomes: np.ndarray,
        coin: str = "",
    ) -> "IsotonicCalibrator":
        """Fit isotonic regression on (raw_conf, outcome) pairs.

        ``raw_confidences`` ∈ [0, 1], ``realised_outcomes`` ∈ {0, 1}
        (1 = correct directional call, 0 = incorrect).
        """
        x = np.asarray(raw_confidences, dtype=float)
        y = np.asarray(realised_outcomes, dtype=float)
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 10:
            raise ValueError(f"need ≥10 samples to fit isotonic; got {len(x)}")
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(x, y)
        self._iso = iso
        self.n_train = int(len(x))
        self.coin = coin
        return self

    def transform(self, raw_confidence: float) -> float:
        if self._iso is None:
            return float(raw_confidence)
        return float(self._iso.predict([raw_confidence])[0])

    def to_pkl(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_pkl(cls, path: str) -> "IsotonicCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def load_or_identity(coin: str, root: str = "data/checkpoints") -> IsotonicCalibrator:
    """Load a fitted calibrator for ``coin`` or return an identity instance.

    Identity = ``transform`` returns its input unchanged. Used by the
    modulator at runtime so an unfit coin degrades gracefully to raw
    confidences instead of crashing.
    """
    path = os.path.join(root, f"isotonic_{coin}.pkl")
    if not os.path.exists(path):
        c = IsotonicCalibrator()
        c.coin = coin
        return c
    try:
        return IsotonicCalibrator.from_pkl(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"isotonic load failed for {coin}: {exc}; using identity")
        c = IsotonicCalibrator()
        c.coin = coin
        return c
