"""Tests for QuantSignalProvider abstraction (V2 + V3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_v2_provider_returns_quant_signal(monkeypatch):
    from tradingagents.strategies.contracts import QuantSignal
    from tradingagents.strategies.quant_signal_provider import V2QuantSignalProvider

    def _fake_get_quant_signal(coin, date, base_dir=None):
        return QuantSignal(
            coin=coin,
            direction="long",
            magnitude=0.5,
            regime="bull",
            regime_confidence=0.7,
            hurst=0.55,
            deterministic_signals={"lgb_h7": 0.01, "lgb_h14": 0.02},
            as_of_date=date,
        )

    monkeypatch.setattr(
        "tradingagents.strategies.quant_signal_provider._v2_get_quant_signal",
        _fake_get_quant_signal,
    )
    provider = V2QuantSignalProvider(base_dir="data/multi_2coins_v2")
    sig = provider.signal(coin="bitcoin", as_of=pd.Timestamp("2026-01-15", tz="UTC"))
    assert isinstance(sig, QuantSignal)
    assert sig.coin == "bitcoin"
    assert sig.direction == "long"
    assert sig.as_of_date == "2026-01-15"


def test_v3_provider_returns_quant_signal(synthetic_ohlcv):
    from tradingagents.strategies.contracts import QuantSignal
    from tradingagents.strategies.quant_signal_provider import V3QuantSignalProvider
    from tradingagents.strategies.v3.config import V3Config
    from tradingagents.strategies.v3.models.multi_horizon import MultiHorizonEnsemble
    from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm

    prices = synthetic_ohlcv["close"]
    returns = prices.pct_change().fillna(0.0)

    regime_bundle = train_nh_hmm(prices=prices, covariates_df=None, n_states=3, n_iter=50)

    features = pd.DataFrame(
        {
            "ret_1d": prices.pct_change().fillna(0.0),
            "ret_5d": prices.pct_change(5).fillna(0.0),
            "vol_5d": prices.pct_change().rolling(5).std().fillna(0.0),
            "vol_21d": prices.pct_change().rolling(21).std().fillna(0.0),
        },
        index=prices.index,
    )
    mhe = MultiHorizonEnsemble(horizons=(7,))
    mhe.fit(features, returns, members=("lgb",))

    provider = V3QuantSignalProvider(
        prices=prices,
        regime_bundle=regime_bundle,
        multi_horizon_bundle=mhe,
        microstructure_features=pd.DataFrame(index=prices.index),
        derivatives_features=pd.DataFrame(index=prices.index),
        config=V3Config(),
    )
    sig = provider.signal(coin="bitcoin", as_of=prices.index[200])
    assert isinstance(sig, QuantSignal)
    assert sig.coin == "bitcoin"
    assert sig.direction in ("long", "short", "flat")
    assert -1.0 <= sig.magnitude <= 1.0
    assert sig.regime in ("bull", "sideways", "bear")
    assert 0.0 <= sig.regime_confidence <= 1.0


def test_build_provider_v2(monkeypatch):
    from tradingagents.strategies.quant_signal_provider import (
        V2QuantSignalProvider,
        build_provider,
    )
    p = build_provider("v2", base_dir="data/multi_2coins_v2")
    assert isinstance(p, V2QuantSignalProvider)


def test_build_provider_unknown_version_raises():
    from tradingagents.strategies.quant_signal_provider import build_provider
    with pytest.raises(ValueError, match="quant version"):
        build_provider("v99")
