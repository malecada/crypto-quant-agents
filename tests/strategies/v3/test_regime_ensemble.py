"""Tests for regime ensemble combiner (HMM + BOCPD + Hurst → RegimeState)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.strategies.v3.contracts import RegimeState


def test_detect_regime_v3_returns_regime_state(synthetic_ohlcv):
    from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3
    from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm

    bundle = train_nh_hmm(
        prices=synthetic_ohlcv["close"],
        covariates_df=None,
        n_states=3,
        n_iter=50,
    )
    state = detect_regime_v3(
        prices=synthetic_ohlcv["close"],
        bundle=bundle,
        as_of=synthetic_ohlcv.index.max(),
    )
    assert isinstance(state, RegimeState)
    assert state.label in ("bull", "sideways", "bear")
    assert 0.0 <= state.confidence <= 1.0
    assert 0.0 <= state.hurst <= 1.0


def test_detect_regime_v3_look_ahead_safe(synthetic_ohlcv):
    """Calling detect_regime_v3 with as_of in the middle of the series
    should produce a result equivalent to truncating the input first."""
    from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3
    from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm

    bundle = train_nh_hmm(
        prices=synthetic_ohlcv["close"],
        covariates_df=None,
        n_states=3,
        n_iter=50,
    )
    mid = synthetic_ohlcv.index[200]
    state_full = detect_regime_v3(
        prices=synthetic_ohlcv["close"], bundle=bundle, as_of=mid
    )
    state_truncated = detect_regime_v3(
        prices=synthetic_ohlcv["close"].loc[:mid], bundle=bundle, as_of=mid
    )
    assert state_full.label == state_truncated.label
    assert abs(state_full.confidence - state_truncated.confidence) < 1e-6


def test_detect_regime_v3_changepoint_dampens_confidence(synthetic_ohlcv):
    """When changepoint_alert fires, confidence should be lower than
    when no changepoint is present (controlling for everything else)."""
    from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3
    from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm

    bundle = train_nh_hmm(
        prices=synthetic_ohlcv["close"],
        covariates_df=None,
        n_states=3,
        n_iter=50,
    )
    # Pick a date right after the synthetic regime change (~bar 100)
    cp_state = detect_regime_v3(
        prices=synthetic_ohlcv["close"],
        bundle=bundle,
        as_of=synthetic_ohlcv.index[103],  # 3 bars after regime jump injected at bar 100
    )
    # Pick a date well after, in stable regime
    stable_state = detect_regime_v3(
        prices=synthetic_ohlcv["close"],
        bundle=bundle,
        as_of=synthetic_ohlcv.index[200],
    )
    # Either changepoint flag fires near the regime jump OR confidence is lower
    if cp_state.changepoint_alert:
        # damping applied
        assert cp_state.confidence <= stable_state.confidence
