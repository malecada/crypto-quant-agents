"""Schema tests for V3 pydantic contracts."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from tradingagents.strategies.v3.contracts import (
    FeatureBundle,
    RegimeState,
    V3Signal,
)


def test_regime_state_valid():
    rs = RegimeState(
        label="bull",
        confidence=0.85,
        hurst=0.62,
        changepoint_alert=False,
        posterior={"bull": 0.85, "sideways": 0.10, "bear": 0.05},
    )
    assert rs.label == "bull"
    assert rs.confidence == 0.85


def test_regime_state_invalid_label():
    with pytest.raises(ValidationError):
        RegimeState(
            label="moon",
            confidence=0.5,
            hurst=0.5,
            changepoint_alert=False,
            posterior={"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        )


def test_regime_state_confidence_bounds():
    with pytest.raises(ValidationError):
        RegimeState(
            label="bull",
            confidence=1.5,
            hurst=0.5,
            changepoint_alert=False,
            posterior={"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        )


def test_v3_signal_valid():
    rs = RegimeState(
        label="sideways",
        confidence=0.4,
        hurst=0.5,
        changepoint_alert=False,
        posterior={"bull": 0.4, "sideways": 0.3, "bear": 0.3},
    )
    sig = V3Signal(
        coin="bitcoin",
        as_of=pd.Timestamp("2026-04-15"),
        direction=1,
        confidence=0.7,
        horizon=14,
        regime=rs,
    )
    assert sig.direction == 1


def test_v3_signal_direction_bounds():
    rs = RegimeState(
        label="bull",
        confidence=0.5,
        hurst=0.5,
        changepoint_alert=False,
        posterior={"bull": 1.0, "sideways": 0.0, "bear": 0.0},
    )
    with pytest.raises(ValidationError):
        V3Signal(
            coin="bitcoin",
            as_of=pd.Timestamp("2026-04-15"),
            direction=2,
            confidence=0.5,
            horizon=7,
            regime=rs,
        )


def test_feature_bundle_round_trip():
    bundle = FeatureBundle(
        coin="bitcoin",
        as_of=pd.Timestamp("2026-04-15"),
        price_features={"sma_30": 64500.0, "ret_5d": 0.04},
        microstructure_features={"vpin_50": 0.42, "ofi_d_w": 0.10},
        derivatives_features={"funding_z_30": -0.5},
    )
    assert bundle.price_features["sma_30"] == 64500.0
    assert bundle.microstructure_features["vpin_50"] == 0.42


def test_regime_state_posterior_sum_validation():
    with pytest.raises(ValidationError):
        RegimeState(
            label="bull",
            confidence=0.5,
            hurst=0.5,
            changepoint_alert=False,
            posterior={"bull": 0.5, "sideways": 0.5, "bear": 0.5},
        )
