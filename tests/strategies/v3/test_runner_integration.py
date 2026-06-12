"""End-to-end integration test for the V3 runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_run_v3_backtest_end_to_end(synthetic_ohlcv):
    """Wire all V3 components on synthetic data, confirm BacktestResult shape."""
    from tradingagents.strategies.v3.backtest.runner_v3 import run_v3_backtest
    from tradingagents.strategies.v3.config import V3Config
    from tradingagents.strategies.v3.models.multi_horizon import MultiHorizonEnsemble
    from tradingagents.strategies.v3.regime.hmm_v2 import train_nh_hmm

    prices = synthetic_ohlcv["close"]
    returns = prices.pct_change().fillna(0.0)

    # Train regime bundle (deterministic, fast)
    regime_bundle = train_nh_hmm(
        prices=prices, covariates_df=None, n_states=3, n_iter=50
    )

    # Build features for multi-horizon training
    features = pd.DataFrame(
        {
            "ret_1d": prices.pct_change().fillna(0.0),
            "ret_5d": prices.pct_change(5).fillna(0.0),
            "vol_5d": prices.pct_change().rolling(5).std().fillna(0.0),
            "vol_21d": prices.pct_change().rolling(21).std().fillna(0.0),
        },
        index=prices.index,
    )

    # Train multi-horizon ensemble — lgb only for fast test
    mhe = MultiHorizonEnsemble(horizons=(7,))
    mhe.fit(features, returns, members=("lgb",))

    # Empty microstructure / derivatives features (skip — runner should handle)
    micro = pd.DataFrame(index=prices.index)
    deriv = pd.DataFrame(index=prices.index)

    cfg = V3Config()

    start = prices.index[100]
    end = prices.index[300]

    result = run_v3_backtest(
        coin="bitcoin",
        prices=prices,
        returns=returns,
        microstructure_features=micro,
        derivatives_features=deriv,
        regime_bundle=regime_bundle,
        multi_horizon_bundle=mhe,
        config=cfg,
        start=start,
        end=end,
        ticker="BTC",
    )

    # Validate result
    from tradingagents.backtesting.engine import BacktestResult
    assert isinstance(result, BacktestResult)
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics
    assert isinstance(result.metrics["sharpe_ratio"], float)
