"""Shared fixtures for V3 tests.

Synthetic OHLCV + tiny model factory avoid network calls and large fits.
Look-ahead invariants are tested by injecting future-tagged rows and asserting
the builder rejects them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=400, freq="D", tz="UTC")
    rets = rng.normal(0.001, 0.02, size=400)
    rets[100:140] += 0.01
    rets[260:300] -= 0.012
    prices = 30000.0 * np.exp(np.cumsum(rets))
    high = prices * (1.0 + np.abs(rng.normal(0, 0.005, 400)))
    low = prices * (1.0 - np.abs(rng.normal(0, 0.005, 400)))
    volume = rng.lognormal(20.0, 0.4, 400)
    return pd.DataFrame(
        {
            "open": prices,
            "high": high,
            "low": low,
            "close": prices,
            "volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def tiny_lgb_factory():
    """Returns a callable that trains a tiny LGBMClassifier on (X, y).

    Used wherever a real model is needed without a heavy fit. 50 estimators,
    depth 3, no parallelism — completes in <1s on dev box.
    """
    from lightgbm import LGBMClassifier

    def _factory(X: np.ndarray, y: np.ndarray):
        model = LGBMClassifier(
            n_estimators=50,
            max_depth=3,
            num_leaves=8,
            learning_rate=0.1,
            verbose=-1,
            n_jobs=1,
        )
        model.fit(X, y)
        return model

    return _factory
