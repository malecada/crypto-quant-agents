"""Tests for the --quant-version flag plumbing."""

from __future__ import annotations

import pandas as pd
import pytest


def test_default_active_version_is_v2():
    from tradingagents.strategies import quant_signal_provider as qsp
    qsp.set_active_quant_version("v2")
    assert qsp.get_active_quant_version() == "v2"


def test_set_invalid_version_raises():
    from tradingagents.strategies import quant_signal_provider as qsp
    with pytest.raises(ValueError):
        qsp.set_active_quant_version("v99")


def test_get_active_quant_signal_v2_path(monkeypatch):
    """v2 path: routes through _v2_get_quant_signal with the right args."""
    from tradingagents.strategies.contracts import QuantSignal
    from tradingagents.strategies import quant_signal_provider as qsp

    captured = {}

    def _fake_v2(coin, date, base_dir=None):
        captured["args"] = (coin, date, base_dir)
        return QuantSignal(
            coin=coin, direction="long", magnitude=0.3,
            regime="bull", regime_confidence=0.6, hurst=0.55,
            deterministic_signals={}, as_of_date=date,
        )

    monkeypatch.setattr(qsp, "_v2_get_quant_signal", _fake_v2)
    qsp.set_active_quant_version("v2")
    sig = qsp.get_active_quant_signal("bitcoin", pd.Timestamp("2026-01-15", tz="UTC"))
    assert captured["args"][0] == "bitcoin"
    assert captured["args"][1] == "2026-01-15"
    assert sig.direction == "long"


def test_get_active_quant_signal_v3_requires_state(monkeypatch):
    from tradingagents.strategies import quant_signal_provider as qsp
    qsp.clear_v3_provider_state()
    qsp.set_active_quant_version("v3")
    with pytest.raises(RuntimeError, match="V3 state not set"):
        qsp.get_active_quant_signal("bitcoin", pd.Timestamp("2026-01-15", tz="UTC"))
    # Reset so other tests don't see v3
    qsp.set_active_quant_version("v2")


def test_get_active_quant_signal_v3_path_with_state(synthetic_ohlcv):
    from tradingagents.strategies import quant_signal_provider as qsp
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

    qsp.set_v3_provider_state(
        coin="bitcoin",
        prices=prices,
        regime_bundle=regime_bundle,
        multi_horizon_bundle=mhe,
        microstructure_features=pd.DataFrame(index=prices.index),
        derivatives_features=pd.DataFrame(index=prices.index),
        config=V3Config(),
    )
    qsp.set_active_quant_version("v3")
    sig = qsp.get_active_quant_signal("bitcoin", prices.index[200])
    assert sig.coin == "bitcoin"
    assert sig.direction in ("long", "short", "flat")
    # Reset
    qsp.clear_v3_provider_state()
    qsp.set_active_quant_version("v2")


def test_v3_state_keyed_by_coin(monkeypatch):
    """Per-coin state: dispatch picks the right state for each coin."""
    from tradingagents.strategies.contracts import QuantSignal
    from tradingagents.strategies import quant_signal_provider as qsp

    # Build two minimal mock V3QuantSignalProviders that return distinguishable signals.
    class _FakeProviderBitcoin:
        def signal(self, coin, as_of):
            return QuantSignal(
                coin=coin, direction="long", magnitude=0.8,
                regime="bull", regime_confidence=0.9, hurst=0.6,
                deterministic_signals={"v3_quant": True},
                as_of_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            )

    class _FakeProviderEthereum:
        def signal(self, coin, as_of):
            return QuantSignal(
                coin=coin, direction="short", magnitude=-0.5,
                regime="bear", regime_confidence=0.7, hurst=0.45,
                deterministic_signals={"v3_quant": True},
                as_of_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            )

    # Patch V3QuantSignalProvider constructor to return per-coin fakes.
    _state_to_provider = {}

    def _fake_v3_init(self, **kwargs):
        # Identify which coin by inspecting prices identity stored in state
        self._kwargs = kwargs

    def _fake_v3_signal(self, coin, as_of):
        return _state_to_provider[coin].signal(coin, as_of)

    monkeypatch.setattr(
        "tradingagents.strategies.quant_signal_provider.V3QuantSignalProvider.__init__",
        _fake_v3_init,
    )
    monkeypatch.setattr(
        "tradingagents.strategies.quant_signal_provider.V3QuantSignalProvider.signal",
        _fake_v3_signal,
    )

    _state_to_provider["bitcoin"] = _FakeProviderBitcoin()
    _state_to_provider["ethereum"] = _FakeProviderEthereum()

    # Register dummy state for two coins (content doesn't matter — mocked above)
    dummy_series = pd.Series([1.0], index=[pd.Timestamp("2026-01-01")])
    dummy_df = pd.DataFrame(index=[pd.Timestamp("2026-01-01")])

    qsp.clear_v3_provider_state()
    for c in ("bitcoin", "ethereum"):
        qsp.set_v3_provider_state(
            coin=c,
            prices=dummy_series,
            regime_bundle=None,
            multi_horizon_bundle=None,
            microstructure_features=dummy_df,
            derivatives_features=dummy_df,
            config=None,
        )

    qsp.set_active_quant_version("v3")
    ts = pd.Timestamp("2026-01-15")

    sig_btc = qsp.get_active_quant_signal("bitcoin", ts)
    sig_eth = qsp.get_active_quant_signal("ethereum", ts)

    assert sig_btc.direction == "long", "bitcoin should dispatch to long signal"
    assert sig_eth.direction == "short", "ethereum should dispatch to short signal"
    assert sig_btc.coin == "bitcoin"
    assert sig_eth.coin == "ethereum"

    # Partial clear: remove only bitcoin; ethereum state must remain
    qsp.clear_v3_provider_state(coin="bitcoin")
    with pytest.raises(RuntimeError, match="V3 state not set"):
        qsp.get_active_quant_signal("bitcoin", ts)
    sig_eth2 = qsp.get_active_quant_signal("ethereum", ts)
    assert sig_eth2.direction == "short"

    # Full clear
    qsp.clear_v3_provider_state()
    with pytest.raises(RuntimeError, match="V3 state not set"):
        qsp.get_active_quant_signal("ethereum", ts)

    qsp.set_active_quant_version("v2")
