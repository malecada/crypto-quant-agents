"""Tests for V3 isotonic calibration."""

from __future__ import annotations

import numpy as np
import pytest


def test_calibrator_returns_callable():
    from tradingagents.strategies.v3.models.calibration import calibrate_probabilities

    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] > 0).astype(int)

    class _DummyModel:
        def predict_proba(self, X):
            # 0.6 if first feature > 0 else 0.4
            return np.column_stack([1.0 - (X[:, 0] > 0) * 0.2 - 0.4,
                                    (X[:, 0] > 0) * 0.2 + 0.4])

    model = _DummyModel()
    calib = calibrate_probabilities(model, X, y)
    assert callable(calib)
    raw = model.predict_proba(X)[:, 1]
    cal = calib(raw)
    assert cal.shape == raw.shape
    assert ((0.0 <= cal) & (cal <= 1.0)).all()


def test_calibration_reduces_brier_on_biased_classifier():
    from tradingagents.strategies.v3.models.calibration import calibrate_probabilities

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    # Skewed class balance
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)

    # Biased classifier — always returns 0.5 + 0.05*y_true (overconfident in 0)
    class _BiasedModel:
        def __init__(self, X, y):
            self._lookup = dict(zip(map(tuple, X), y))

        def predict_proba(self, X):
            preds = np.array([self._lookup[tuple(x)] for x in X])
            # Squash to [0.45, 0.55] — overconfident-near-50% biased pattern
            p = 0.45 + 0.10 * preds.astype(float)
            return np.column_stack([1.0 - p, p])

    # Fit calibrator on first 250, evaluate on last 250
    # Model is initialized with all X so the lookup covers both splits
    model = _BiasedModel(X, y)
    raw = model.predict_proba(X[:250])[:, 1]
    raw_test = model.predict_proba(X[250:])[:, 1]

    calib = calibrate_probabilities(model, X[:250], y[:250])
    cal_test = calib(raw_test)

    # Brier score on test (lower is better)
    brier_raw = np.mean((raw_test - y[250:]) ** 2)
    brier_cal = np.mean((cal_test - y[250:]) ** 2)
    assert brier_cal <= brier_raw + 1e-6, (
        f"Calibrated Brier {brier_cal} should be <= raw {brier_raw}"
    )


def test_calibration_monotonic():
    """Isotonic calibration must be non-decreasing."""
    from tradingagents.strategies.v3.models.calibration import calibrate_probabilities

    rng = np.random.default_rng(42)
    n = 300
    raw_probs = rng.uniform(0.0, 1.0, size=n)
    y = (raw_probs + rng.normal(0, 0.1, n) > 0.5).astype(int)
    X = raw_probs.reshape(-1, 1)

    class _IdentityModel:
        def predict_proba(self, X):
            p = X[:, 0]
            return np.column_stack([1.0 - p, p])

    model = _IdentityModel()
    calib = calibrate_probabilities(model, X, y)

    # Sort raw → calibrated should be non-decreasing
    test_raw = np.linspace(0.0, 1.0, 50)
    test_cal = calib(test_raw)
    diffs = np.diff(test_cal)
    assert (diffs >= -1e-9).all(), "Isotonic calibration must be monotonic"
