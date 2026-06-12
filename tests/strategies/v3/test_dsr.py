from __future__ import annotations

import numpy as np

from tradingagents.strategies.v3.backtest.dsr import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)


def test_expected_max_sharpe_monotonic_in_n_trials():
    e1 = expected_max_sharpe(n_trials=1, var_sr=1.0)
    e10 = expected_max_sharpe(n_trials=10, var_sr=1.0)
    e100 = expected_max_sharpe(n_trials=100, var_sr=1.0)
    assert e1 < e10 < e100


def test_dsr_zero_when_observed_equals_expected_max():
    sr_obs = 1.5
    sr_exp = 1.5
    se_sr = 0.5
    dsr = deflated_sharpe_ratio(
        sr_observed=sr_obs,
        sr_expected_under_null=sr_exp,
        se_sr=se_sr,
    )
    assert abs(dsr - 0.5) < 1e-6  # Φ((0)/se) = 0.5


def test_dsr_high_when_observed_much_higher_than_expected():
    dsr = deflated_sharpe_ratio(
        sr_observed=3.5,
        sr_expected_under_null=1.0,
        se_sr=0.3,
    )
    assert dsr > 0.99


def test_dsr_low_when_observed_below_expected():
    dsr = deflated_sharpe_ratio(
        sr_observed=0.5,
        sr_expected_under_null=1.5,
        se_sr=0.3,
    )
    assert dsr < 0.01


def test_dsr_with_cpcv_pipeline(synthetic_ohlcv):
    """Simulate evaluating a noise strategy on synthetic data via CPCV; DSR should be near 0.5."""
    from tradingagents.strategies.v3.backtest.cpcv import cpcv_splits
    from tradingagents.strategies.v3.backtest.dsr import (
        variance_of_sr,
    )

    rng = np.random.default_rng(0)
    rets = rng.normal(0.0, 0.02, size=len(synthetic_ohlcv))

    sharpes = []
    for split in cpcv_splits(n_samples=len(rets), n_groups=8, test_groups=2, embargo=14):
        test_rets = rets[split.test_idx]
        sharpes.append(np.mean(test_rets) / (np.std(test_rets) + 1e-9))

    sr_obs = float(np.mean(sharpes))
    var_sr = variance_of_sr(np.array(sharpes))
    sr_exp = expected_max_sharpe(n_trials=12, var_sr=var_sr)
    dsr = deflated_sharpe_ratio(sr_obs, sr_exp, np.sqrt(max(var_sr, 1e-9)))
    # noise strategy → DSR should be < 0.95
    assert dsr < 0.95
