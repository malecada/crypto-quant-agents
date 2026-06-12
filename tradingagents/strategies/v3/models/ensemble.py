"""V3 GBDT ensemble: LGB + (optional) XGB + (optional) CatBoost simple average.

LGB is mandatory; XGB and CatBoost are optional v3 extras. If either optional
member fails to import, the ensemble drops it and continues with the rest.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


def _make_lgb(**kwargs):
    """Factory for LGBMClassifier — extracted to allow monkeypatching in tests."""
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 5),
        num_leaves=kwargs.get("num_leaves", 31),
        learning_rate=kwargs.get("learning_rate", 0.05),
        verbose=-1,
        n_jobs=kwargs.get("n_jobs", 1),
        random_state=kwargs.get("random_state", 42),
    )


def _make_xgb(**kwargs):
    """Factory for XGBClassifier — extracted for monkeypatching + import isolation."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=kwargs.get("n_estimators", 200),
        max_depth=kwargs.get("max_depth", 5),
        learning_rate=kwargs.get("learning_rate", 0.05),
        n_jobs=kwargs.get("n_jobs", 1),
        random_state=kwargs.get("random_state", 42),
        verbosity=0,
        eval_metric="logloss",
    )


def _make_catboost(**kwargs):
    """Factory for CatBoostClassifier — extracted for monkeypatching + import isolation."""
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=kwargs.get("n_estimators", 200),
        depth=kwargs.get("max_depth", 5),
        learning_rate=kwargs.get("learning_rate", 0.05),
        thread_count=kwargs.get("n_jobs", 1),
        random_state=kwargs.get("random_state", 42),
        verbose=False,
    )


_FACTORY_MAP: dict[str, str] = {
    "lgb": "_make_lgb",
    "xgb": "_make_xgb",
    "catboost": "_make_catboost",
}


class EnsembleModel:
    """Simple-average ensemble over LGB + XGB + CatBoost.

    Members not in ``members`` are skipped. Optional members (xgb, catboost)
    silently drop on ImportError; lgb is mandatory and raises RuntimeError if
    its factory fails.
    """

    def __init__(
        self,
        horizon: int,
        members: tuple[str, ...] = ("lgb", "xgb", "catboost"),
        **member_kwargs,
    ) -> None:
        self.horizon = horizon
        self.requested_members = tuple(members)
        self.member_kwargs = dict(member_kwargs)
        self._fitted_members: dict[str, object] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> "EnsembleModel":
        import sys

        _this_module = sys.modules[__name__]
        for name in self.requested_members:
            factory_name = _FACTORY_MAP.get(name)
            if factory_name is None:
                logger.warning("Unknown ensemble member %s — skipping", name)
                continue
            # Look up the factory via module attribute so monkeypatching works in tests
            factory = getattr(_this_module, factory_name)
            try:
                model = factory(**self.member_kwargs)
            except ImportError as e:
                if name == "lgb":
                    raise RuntimeError(f"lightgbm is required: {e}") from e
                logger.warning("%s unavailable (%s) — dropped from ensemble", name, e)
                continue
            try:
                model.fit(X, y, sample_weight=weights)
            except TypeError:
                # Some classifiers don't accept sample_weight kwarg
                model.fit(X, y)
            self._fitted_members[name] = model
        if not self._fitted_members:
            raise RuntimeError("EnsembleModel: no members fit successfully")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted_members:
            raise RuntimeError("EnsembleModel: must fit before predict")
        probs = []
        for name, model in self._fitted_members.items():
            p = model.predict_proba(X)
            probs.append(p)
        return np.mean(probs, axis=0)

    @property
    def fitted_member_names(self) -> tuple[str, ...]:
        return tuple(self._fitted_members.keys())
