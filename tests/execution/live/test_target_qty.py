"""C1 fix — converting a per-coin size fraction to a signed target quantity.

The bug: runner computed `qty = final_size_notional * portfolio_before / ref_price`,
multiplying each coin's full-equity size fraction by the WHOLE portfolio value.
With N coins that is ~N x the gross exposure the validated backtest runs, because
the backtest weights the per-coin sleeves to sum to 1. The fix folds the coin's
renormalized portfolio weight into the conversion.
"""
import math


def test_qty_scales_by_portfolio_weight():
    from tradingagents.execution.live.sizer import target_position_qty

    # fraction 0.3 of equity, $10k book, 25% weight, $100 price
    qty = target_position_qty(
        size_fraction=0.3, portfolio_value=10_000.0,
        weight=0.25, ref_price=100.0,
    )
    # 0.3 * 10000 * 0.25 / 100 = 7.5  (the un-weighted bug would give 30.0)
    assert math.isclose(qty, 7.5, rel_tol=1e-9)


def test_qty_preserves_short_sign():
    from tradingagents.execution.live.sizer import target_position_qty

    qty = target_position_qty(
        size_fraction=-0.3, portfolio_value=10_000.0,
        weight=0.25, ref_price=100.0,
    )
    assert qty < 0
    assert math.isclose(qty, -7.5, rel_tol=1e-9)


def test_qty_zero_on_nonpositive_price():
    from tradingagents.execution.live.sizer import target_position_qty

    assert target_position_qty(
        size_fraction=0.3, portfolio_value=10_000.0, weight=0.25, ref_price=0.0,
    ) == 0.0


def test_equal_fraction_book_gross_equals_single_sleeve():
    """The anti-over-leverage property: when every coin is sized at the same
    fraction f, the total gross notional of the weighted book equals f x equity
    (one sleeve), NOT N x f x equity. This is exactly what the missing weight
    used to break."""
    from tradingagents.execution.live.config import compute_portfolio_weights
    from tradingagents.execution.live.sizer import target_position_qty

    universe = ["bitcoin", "ethereum", "binancecoin", "solana"]
    weights = compute_portfolio_weights(universe)
    portfolio = 10_000.0
    f = 0.3
    prices = {"bitcoin": 60_000.0, "ethereum": 3_000.0,
              "binancecoin": 600.0, "solana": 150.0}

    gross = 0.0
    for coin in universe:
        qty = target_position_qty(
            size_fraction=f, portfolio_value=portfolio,
            weight=weights[coin], ref_price=prices[coin],
        )
        gross += abs(qty) * prices[coin]

    # gross == f * portfolio (one sleeve), not 4 * f * portfolio
    assert math.isclose(gross, f * portfolio, rel_tol=1e-9)
