"""V3 multi-horizon ensemble: per-horizon EnsembleModel + isotonic calibration.

For each horizon h:
  1. Build h-step-ahead binary labels: sign(future_ret_h) > 0 → 1
  2. Split train/holdout 80/20 preserving time order (calibrator on last 20%)
  3. Fit EnsembleModel on first 80%
  4. Fit isotonic calibrator on holdout 20%
  5. Predict-time: feed raw model probs through calibrator
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from tradingagents.strategies.v3.models.calibration import calibrate_probabilities
from tradingagents.strategies.v3.models.ensemble import EnsembleModel

logger = logging.getLogger(__name__)


@dataclass
class _PerHorizonModel:
    horizon: int
    ensemble: EnsembleModel
    calibrator: Callable[[np.ndarray], np.ndarray] | None = None


class MultiHorizonEnsemble:
    """Trains and serves per-horizon ensemble + isotonic calibration."""

    def __init__(
        self,
        horizons: tuple[int, ...] = (3, 7, 14, 21),
        holdout_fraction: float = 0.20,
    ) -> None:
        self.horizons = tuple(horizons)
        self.holdout_fraction = float(holdout_fraction)
        self._models: dict[int, _PerHorizonModel] = {}

    def fit(
        self,
        features_df: pd.DataFrame,
        returns_series: pd.Series,
        members: tuple[str, ...] = ("lgb", "xgb", "catboost"),
        use_calibration: bool = True,
    ) -> "MultiHorizonEnsemble":
        """Fit per-horizon ensemble models.

        Args:
            features_df: Feature DataFrame aligned with returns_series index.
            returns_series: Daily returns series.
            members: Ensemble member names to train. Use ``("lgb",)`` for
                lgb-only (recommended after root-cause analysis in
                data/diagnostics/v3_root_cause.md).
            use_calibration: When True (default), fit an isotonic calibrator
                on the holdout 20% and use it at predict time. When False, raw
                model probabilities are used directly — confirmed to yield
                wider proba spread and better short-signal coverage.
                NOTE: the 80/20 split is retained even when use_calibration=False
                (we train on first 80% and leave holdout unused), so train set
                size is identical in both modes.
        """
        if not features_df.index.equals(returns_series.index):
            raise ValueError("features_df and returns_series must share an index")

        for h in self.horizons:
            # Build h-step-ahead label: sign of future cumulative log return
            future_ret = (
                np.log1p(returns_series).rolling(window=h).sum().shift(-h)
            )
            valid_mask = future_ret.notna()
            X_full = features_df.loc[valid_mask].values
            y_full = (future_ret.loc[valid_mask] > 0).astype(int).values
            n = len(X_full)
            if n < 30:
                logger.warning(
                    "Horizon %d has only %d samples after labeling; skipping", h, n
                )
                continue

            n_train = int(n * (1.0 - self.holdout_fraction))
            X_train, X_holdout = X_full[:n_train], X_full[n_train:]
            y_train, y_holdout = y_full[:n_train], y_full[n_train:]

            ensemble = EnsembleModel(horizon=h, members=members)
            ensemble.fit(X_train, y_train)

            calibrator: Callable | None = None
            if use_calibration and len(X_holdout) >= 10 and len(set(y_holdout)) == 2:
                try:
                    calibrator = calibrate_probabilities(
                        ensemble, X_holdout, y_holdout
                    )
                except Exception:
                    logger.exception(
                        "Calibration failed for horizon %d; using raw probs", h
                    )

            self._models[h] = _PerHorizonModel(
                horizon=h, ensemble=ensemble, calibrator=calibrator
            )

        if not self._models:
            raise RuntimeError("MultiHorizonEnsemble: no horizon trained")
        return self

    def predict_proba(self, features_df: pd.DataFrame) -> dict[int, np.ndarray]:
        if not self._models:
            raise RuntimeError("MultiHorizonEnsemble: must fit before predict")
        X = features_df.values
        out: dict[int, np.ndarray] = {}
        for h, ph in self._models.items():
            raw = ph.ensemble.predict_proba(X)[:, 1]
            if ph.calibrator is not None:
                out[h] = ph.calibrator(raw)
            else:
                out[h] = raw
        return out

    @property
    def fitted_horizons(self) -> tuple[int, ...]:
        return tuple(self._models.keys())


import pickle
from pathlib import Path


def train_multi_horizon(
    features_parquet: Path | str,
    returns_csv: Path | str,
    out_dir: Path | str,
    coin: str,
    horizons: tuple[int, ...] = (3, 7, 14, 21),
    members: tuple[str, ...] = ("lgb", "xgb", "catboost"),
    holdout_fraction: float = 0.20,
) -> Path:
    """Load features+returns, fit MultiHorizonEnsemble, pickle to disk.

    Returns the pickle path.
    """
    features_parquet = Path(features_parquet)
    returns_csv = Path(returns_csv)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_parquet(features_parquet)
    returns_df = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
    if isinstance(returns_df, pd.DataFrame):
        if returns_df.shape[1] == 1:
            returns_series = returns_df.iloc[:, 0]
        else:
            raise ValueError(
                f"returns_csv {returns_csv} must have exactly one data column"
            )
    else:
        returns_series = returns_df

    # Align indices
    common_idx = features.index.intersection(returns_series.index)
    features = features.loc[common_idx]
    returns_series = returns_series.loc[common_idx]

    mhe = MultiHorizonEnsemble(horizons=horizons, holdout_fraction=holdout_fraction)
    mhe.fit(features, returns_series, members=members)

    out_path = out_dir / f"v3_models_{coin}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(mhe, f)
    logger.info(
        "Wrote %s — horizons=%s members=%s",
        out_path,
        mhe.fitted_horizons,
        members,
    )
    return out_path


from tradingagents.strategies.v3.contracts import RegimeState  # noqa: E402
from tradingagents.strategies.v3.config import V3Config  # noqa: E402

_DEADBAND = 0.05


def _regime_mode(regime: RegimeState, config: V3Config) -> str:
    if regime.hurst > config.hurst_trend_threshold:
        return "trending"
    if regime.hurst < config.hurst_mr_threshold:
        return "mean_reverting"
    return "uncertain"


def consensus_signal(
    probas: dict[int, float],
    regime: RegimeState,
    config: V3Config,
    deadband: float = _DEADBAND,
) -> tuple[int, float]:
    """Combine per-horizon probabilities into a single (direction, confidence).

    direction is +1/-1 if the weighted probability is more than ``deadband``
    away from 0.5; 0 otherwise. confidence is ``2 * |p - 0.5|``.
    """
    mode = _regime_mode(regime, config)
    weights = config.horizon_weights(mode)

    # Restrict weights to horizons present in probas; renormalize.
    active_weights = {h: w for h, w in weights.items() if h in probas}
    total_w = sum(active_weights.values())
    if total_w <= 0:
        return 0, 0.0
    normed_weights = {h: w / total_w for h, w in active_weights.items()}

    weighted_p = sum(normed_weights[h] * probas[h] for h in normed_weights)
    diff = weighted_p - 0.5
    if abs(diff) <= deadband:
        return 0, 0.0
    direction = 1 if diff > 0 else -1
    confidence = float(min(1.0, max(0.0, 2.0 * abs(diff))))
    return direction, confidence


def select_features_per_horizon(
    model: EnsembleModel | object,
    X_holdout: np.ndarray,
    feature_names: list[str],
    drop_bottom_pct: float = 0.20,
) -> list[str]:
    """Drop bottom-quantile features by mean |SHAP value|.

    Uses shap.TreeExplainer on the LGB member of an EnsembleModel (TreeExplainer
    is fastest + works reliably on LGB). If shap is not installed, returns
    ``feature_names`` unchanged with a warning.

    ``drop_bottom_pct`` of features (rounded down) are dropped; remaining
    features are returned in their original order.
    """
    try:
        import shap
        if shap is None:  # explicit None check for monkeypatch-stubbed path
            raise ImportError("shap not available")
    except ImportError:
        logger.warning("shap not available — feature selection skipped")
        return list(feature_names)

    # Find the LGB member (TreeExplainer is fastest on LGB)
    lgb_model = None
    if isinstance(model, EnsembleModel):
        lgb_model = model._fitted_members.get("lgb") if hasattr(model, "_fitted_members") else None
    if lgb_model is None:
        logger.warning("No LGB member available for SHAP — feature selection skipped")
        return list(feature_names)

    try:
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(X_holdout)
        # For binary classifiers shap_values is a list [neg, pos] in older versions
        # or a 2-D array in newer versions. Normalize to 2-D positive-class.
        if isinstance(shap_values, list):
            sv = np.asarray(shap_values[1])
        else:
            sv = np.asarray(shap_values)
        if sv.ndim == 3:
            # (n_samples, n_features, n_classes) — pick positive class
            sv = sv[..., 1]
        importance = np.abs(sv).mean(axis=0)
    except Exception:
        logger.exception("SHAP computation failed — feature selection skipped")
        return list(feature_names)

    n_drop = int(len(feature_names) * drop_bottom_pct)
    if n_drop <= 0:
        return list(feature_names)

    # Indices of features to drop (lowest importance)
    drop_idx = set(np.argsort(importance)[:n_drop].tolist())
    return [name for i, name in enumerate(feature_names) if i not in drop_idx]
