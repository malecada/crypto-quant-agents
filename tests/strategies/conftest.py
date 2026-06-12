"""Shared fixtures for strategies-level tests.

Re-exports the ``synthetic_ohlcv`` fixture from the v3 conftest so that
non-v3 test files (e.g. test_quant_signal_provider.py) can use it without
duplicating the fixture definition.
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
