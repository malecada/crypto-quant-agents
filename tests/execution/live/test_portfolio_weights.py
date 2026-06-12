"""C1 fix — per-coin portfolio weights for live sizing.

The live cycle must scale each coin's vol-targeted size fraction by its
renormalized portfolio weight before converting to a quantity, otherwise an
N-coin book runs ~N x the gross exposure of the validated backtest
(baseline_v5_mix.portfolio_return weights the per-coin sleeves to sum to 1).
"""
import math


def test_four_core_coins_renormalize_to_equal_weight():
    from tradingagents.execution.live.config import compute_portfolio_weights

    w = compute_portfolio_weights(
        ["bitcoin", "ethereum", "binancecoin", "solana"]
    )
    assert set(w) == {"bitcoin", "ethereum", "binancecoin", "solana"}
    for coin in w:
        assert math.isclose(w[coin], 0.25, rel_tol=1e-9)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)


def test_eight_coin_core_satellite_split():
    from tradingagents.execution.live.config import compute_portfolio_weights

    universe = [
        "bitcoin", "ethereum", "binancecoin", "solana",
        "ripple", "dogecoin", "cardano", "tron",
    ]
    w = compute_portfolio_weights(universe)
    for core in ("bitcoin", "ethereum", "binancecoin", "solana"):
        assert math.isclose(w[core], 0.15, rel_tol=1e-9)
    for sat in ("ripple", "dogecoin", "cardano", "tron"):
        assert math.isclose(w[sat], 0.10, rel_tol=1e-9)
    assert math.isclose(sum(w.values()), 1.0, rel_tol=1e-9)


def test_subset_renormalizes_to_sum_one():
    from tradingagents.execution.live.config import compute_portfolio_weights

    w = compute_portfolio_weights(["bitcoin", "ethereum"])
    assert math.isclose(w["bitcoin"], 0.5, rel_tol=1e-9)
    assert math.isclose(w["ethereum"], 0.5, rel_tol=1e-9)


def test_live_weights_match_backtest_source_of_truth():
    """Drift guard: live weights must equal baseline_v5_mix.PORTFOLIO_WEIGHTS,
    the published source that produced SR +3.18."""
    from tradingagents.execution.live.config import _V5_PORTFOLIO_WEIGHTS
    from scripts.baseline_v5_mix import PORTFOLIO_WEIGHTS

    assert _V5_PORTFOLIO_WEIGHTS == PORTFOLIO_WEIGHTS
