"""Tests for V3 GBDT ensemble (LGB + XGB + CatBoost simple average)."""

from __future__ import annotations

import numpy as np
import pytest


def _make_synthetic_classification(n: int = 200, n_features: int = 10, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    # Linear-separable y based on sum of first 3 features
    y = (X[:, :3].sum(axis=1) > 0).astype(int)
    return X, y


def test_ensemble_fit_predict_default_members():
    from tradingagents.strategies.v3.models.ensemble import EnsembleModel

    X, y = _make_synthetic_classification()
    model = EnsembleModel(horizon=7)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    # On a learnable problem, ensemble should beat 50% on training
    pred = (proba[:, 1] > 0.5).astype(int)
    assert (pred == y).mean() > 0.7


def test_ensemble_lgb_only():
    from tradingagents.strategies.v3.models.ensemble import EnsembleModel

    X, y = _make_synthetic_classification()
    model = EnsembleModel(horizon=14, members=("lgb",))
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (200, 2)


def test_ensemble_average_three_identical_stubs(monkeypatch):
    """If LGB, XGB, CatBoost all return identical probabilities, the average
    should equal that same vector."""
    from tradingagents.strategies.v3.models import ensemble as ens

    class _StubModel:
        def __init__(self, *args, **kwargs):
            self._fitted = False

        def fit(self, X, y, sample_weight=None):
            self._fitted = True
            return self

        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (X.shape[0], 1))

    monkeypatch.setattr(ens, "_make_lgb", lambda **kw: _StubModel())
    monkeypatch.setattr(ens, "_make_xgb", lambda **kw: _StubModel())
    monkeypatch.setattr(ens, "_make_catboost", lambda **kw: _StubModel())

    X, y = _make_synthetic_classification()
    model = ens.EnsembleModel(horizon=3)
    model.fit(X, y)
    proba = model.predict_proba(X[:5])
    np.testing.assert_allclose(proba, np.tile([0.3, 0.7], (5, 1)), atol=1e-9)


def test_ensemble_drops_missing_optional_member(monkeypatch, caplog):
    """If xgboost import fails, ensemble should drop xgb and continue with lgb+catboost."""
    import logging
    from tradingagents.strategies.v3.models import ensemble as ens

    def _broken_xgb(**kwargs):
        raise ImportError("xgboost not installed")

    monkeypatch.setattr(ens, "_make_xgb", _broken_xgb)

    X, y = _make_synthetic_classification()
    with caplog.at_level(logging.WARNING):
        model = ens.EnsembleModel(horizon=7)
        model.fit(X, y)
        proba = model.predict_proba(X[:5])
    assert "xgb" in caplog.text.lower() or "xgboost" in caplog.text.lower()
    assert proba.shape == (5, 2)


def test_ensemble_requires_lgb(monkeypatch):
    """If lgb import fails, ensemble should raise — LGB is mandatory."""
    from tradingagents.strategies.v3.models import ensemble as ens

    def _broken_lgb(**kwargs):
        raise ImportError("lightgbm not installed")

    monkeypatch.setattr(ens, "_make_lgb", _broken_lgb)

    X, y = _make_synthetic_classification()
    model = ens.EnsembleModel(horizon=7)
    with pytest.raises(RuntimeError, match="lightgbm"):
        model.fit(X, y)
