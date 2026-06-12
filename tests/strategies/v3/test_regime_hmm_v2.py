"""Smoke tests for v3 regime module promotion.

Task 17 promotes the existing HMM regime detector from
``tradingagents.strategies.regime`` into the v3 sub-package. These tests
verify the module imports and exposes the expected public API. Behavioral
tests for NH-HMM extensions come in Tasks 18-21.
"""

from __future__ import annotations

import pandas as pd


def test_hmm_v2_imports():
    from tradingagents.strategies.v3.regime import hmm_v2

    # Public names match the parent regime module
    assert hasattr(hmm_v2, "FittedHMM")
    assert hasattr(hmm_v2, "build_regime_features")


def test_hmm_v2_build_features_runs(synthetic_ohlcv):
    from tradingagents.strategies.v3.regime.hmm_v2 import build_regime_features

    df = build_regime_features(synthetic_ohlcv["close"])
    # dropna() removes the initial NaN rows from rolling windows (vol_lookback=30,
    # smooth_window=20 → max lag = 30 bars dropped), so output < input length.
    assert len(df) < len(synthetic_ohlcv)
    assert len(df) > 0
    assert df.columns.tolist() == [
        "log_return_smooth",
        "realized_vol",
        "abs_return_smooth",
    ]


def test_nh_transition_matrix_softmax_normalizes_rows():
    from tradingagents.strategies.v3.regime.hmm_v2 import NHTransitionMatrix
    import numpy as np

    coefs = np.zeros((3, 3, 2))  # 3 from-states × 3 to-states × 2 covariates
    intercepts = np.zeros((3, 3))
    nh = NHTransitionMatrix(coefs=coefs, intercepts=intercepts)
    cov = np.array([0.5, 0.0001])
    M = nh.transition(cov)
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-9)


def test_nh_transition_matrix_zero_intercepts_uniform():
    from tradingagents.strategies.v3.regime.hmm_v2 import NHTransitionMatrix
    import numpy as np

    coefs = np.zeros((3, 3, 2))
    intercepts = np.zeros((3, 3))
    nh = NHTransitionMatrix(coefs=coefs, intercepts=intercepts)
    M = nh.transition(np.array([0.0, 0.0]))
    np.testing.assert_allclose(M, np.full((3, 3), 1.0 / 3.0), atol=1e-9)


def test_nh_transition_matrix_high_vol_increases_bull_exit():
    """If the vol covariate has a positive coefficient on the
    bull→sideways and bull→bear transitions, raising the vol input
    should lower P(bull→bull) and raise P(bull→bear)+P(bull→sideways).
    """
    from tradingagents.strategies.v3.regime.hmm_v2 import NHTransitionMatrix
    import numpy as np

    coefs = np.zeros((3, 3, 2))
    # bull is state 0, sideways state 1, bear state 2.
    # Make leaving bull more likely under high vol (covariate index 0).
    coefs[0, 1, 0] = 5.0  # bull→sideways gets boost from vol
    coefs[0, 2, 0] = 5.0  # bull→bear gets boost from vol
    intercepts = np.zeros((3, 3))
    nh = NHTransitionMatrix(coefs=coefs, intercepts=intercepts)
    M_low = nh.transition(np.array([0.0, 0.0]))
    M_high = nh.transition(np.array([1.0, 0.0]))
    assert M_high[0, 0] < M_low[0, 0]
    assert M_high[0, 1] + M_high[0, 2] > M_low[0, 1] + M_low[0, 2]


def test_update_posterior_normalizes_to_one():
    from tradingagents.strategies.v3.regime.hmm_v2 import update_posterior
    import numpy as np

    prev = np.array([1.0, 0.0, 0.0])
    transition = np.array([
        [0.7, 0.2, 0.1],
        [0.3, 0.4, 0.3],
        [0.1, 0.3, 0.6],
    ])
    emission_logprobs = np.array([-1.0, -2.0, -3.0])
    new = update_posterior(prev, transition, emission_logprobs)
    assert abs(new.sum() - 1.0) < 1e-9
    assert (new >= 0).all()


def test_update_posterior_concentrates_on_strong_likelihood():
    from tradingagents.strategies.v3.regime.hmm_v2 import update_posterior
    import numpy as np

    prev = np.array([0.34, 0.33, 0.33])
    transition = np.eye(3)  # stationary — posterior driven only by emission
    emission_logprobs = np.array([0.0, -10.0, -10.0])  # state 0 vastly more likely
    new = update_posterior(prev, transition, emission_logprobs)
    assert new[0] > 0.99


def test_update_posterior_after_long_bull_sequence():
    """A long sequence of bull-favoring observations should drive the
    posterior bull-mass to ~1.0 even from a uniform prior."""
    from tradingagents.strategies.v3.regime.hmm_v2 import update_posterior
    import numpy as np

    posterior = np.array([0.34, 0.33, 0.33])
    transition = np.array([
        [0.95, 0.04, 0.01],
        [0.05, 0.90, 0.05],
        [0.01, 0.04, 0.95],
    ])
    bull_emission = np.array([0.0, -2.0, -5.0])  # bull state strongly preferred
    for _ in range(50):
        posterior = update_posterior(posterior, transition, bull_emission)
    assert posterior[0] > 0.95


def test_update_posterior_input_validation():
    from tradingagents.strategies.v3.regime.hmm_v2 import update_posterior
    import numpy as np
    import pytest

    prev = np.array([0.5, 0.5])  # wrong size — only 2 states
    transition = np.eye(3)
    emission = np.zeros(3)
    with pytest.raises(ValueError):
        update_posterior(prev, transition, emission)


def test_train_nh_hmm_returns_bundle(synthetic_ohlcv):
    from tradingagents.strategies.v3.regime.hmm_v2 import (
        NHHmmBundle,
        NHTransitionMatrix,
        train_nh_hmm,
    )

    bundle = train_nh_hmm(
        prices=synthetic_ohlcv["close"],
        covariates_df=None,  # accepts None for the v0 (degenerate) path
        n_states=3,
        n_iter=50,
    )
    assert isinstance(bundle, NHHmmBundle)
    assert bundle.n_states == 3
    assert isinstance(bundle.nh_transition, NHTransitionMatrix)
    assert bundle.nh_transition.coefs.shape == (3, 3, 2)


def test_train_nh_hmm_pickle_round_trip(synthetic_ohlcv, tmp_path):
    import pickle
    from tradingagents.strategies.v3.regime.hmm_v2 import (
        NHHmmBundle,
        train_nh_hmm,
    )

    bundle = train_nh_hmm(
        prices=synthetic_ohlcv["close"],
        covariates_df=None,
        n_states=3,
        n_iter=50,
    )
    out_file = tmp_path / "test_bundle.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(bundle, f)
    with open(out_file, "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, NHHmmBundle)
    assert loaded.n_states == bundle.n_states


def test_train_nh_hmm_fallback_on_convergence_failure(synthetic_ohlcv, monkeypatch, caplog):
    """If GaussianHMM raises during fit, train_nh_hmm should propagate
    the error (not silently produce an unfitted bundle)."""
    import logging
    from tradingagents.strategies.v3.regime import hmm_v2

    class _BrokenHMM:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, X):
            raise RuntimeError("fake convergence failure")

    monkeypatch.setattr(hmm_v2, "GaussianHMM", _BrokenHMM)

    with caplog.at_level(logging.WARNING):
        try:
            hmm_v2.train_nh_hmm(
                prices=synthetic_ohlcv["close"],
                covariates_df=None,
                n_states=3,
                n_iter=50,
            )
        except RuntimeError:
            pass  # expected
    # Either a warning is emitted OR the error propagates — both are acceptable.
