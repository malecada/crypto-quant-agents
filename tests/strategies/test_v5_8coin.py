# tests/strategies/test_v5_8coin.py
"""Tests for the V5 MIX 8-coin expansion: symbol map, cost tiers, weighting."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_tron_symbol_resolves():
    from tradingagents.dataflows.coingecko_binance import _KNOWN_SYMBOLS
    assert _KNOWN_SYMBOLS["tron"] == "TRXUSDT"


# --- Task 8: two-tier cost function ----------------------------------------

def test_costs_for_coin_core_unchanged():
    from scripts.baseline_v5_mix import COSTS, costs_for_coin
    c = costs_for_coin("bitcoin")
    assert c == COSTS  # core coins get the legacy cost dict verbatim


def test_costs_for_coin_satellite_haircut():
    from scripts.baseline_v5_mix import COSTS, costs_for_coin
    c = costs_for_coin("ripple")  # default haircut = 1.5
    assert c["slippage"] == pytest.approx(COSTS["slippage"] * 1.5)
    assert c["price_impact"] == pytest.approx(COSTS["price_impact"] * 1.5)
    assert c["fee_rate"] == COSTS["fee_rate"]  # non-haircut keys unchanged


def test_costs_for_coin_satellite_haircut_param():
    from scripts.baseline_v5_mix import COSTS, costs_for_coin
    c = costs_for_coin("dogecoin", sat_haircut=2.0)
    assert c["slippage"] == pytest.approx(COSTS["slippage"] * 2.0)
    assert c["price_impact"] == pytest.approx(COSTS["price_impact"] * 2.0)


# --- Task 9: core/satellite portfolio weighting ----------------------------

def test_portfolio_weights_sum_to_one():
    from scripts.baseline_v5_mix import PORTFOLIO_WEIGHTS
    assert sum(PORTFOLIO_WEIGHTS.values()) == pytest.approx(1.0)
    assert PORTFOLIO_WEIGHTS["bitcoin"] > PORTFOLIO_WEIGHTS["ripple"]


def test_portfolio_return_weighted():
    from scripts.baseline_v5_mix import portfolio_return
    idx = pd.date_range("2022-01-01", periods=3)
    df = pd.DataFrame({"bitcoin": [0.10, 0.0, 0.0],
                       "ripple": [0.0, 0.20, 0.0]}, index=idx)
    weights = {"bitcoin": 0.15, "ripple": 0.10}
    # subset renormalizes: 0.15/0.25=0.6 BTC, 0.10/0.25=0.4 XRP
    out = portfolio_return(df, weights)
    assert out.iloc[0] == pytest.approx(0.10 * 0.6)
    assert out.iloc[1] == pytest.approx(0.20 * 0.4)
    assert out.iloc[2] == pytest.approx(0.0)


def test_portfolio_return_4coin_equals_mean():
    """Regression guard: 4 equal-weight core coins reproduce df.mean()."""
    from scripts.baseline_v5_mix import portfolio_return, PORTFOLIO_WEIGHTS
    idx = pd.date_range("2022-01-01", periods=4)
    df = pd.DataFrame({c: [0.01, 0.02, 0.03, 0.04]
                       for c in ["bitcoin", "ethereum", "binancecoin", "solana"]},
                      index=idx)
    out = portfolio_return(df, PORTFOLIO_WEIGHTS)
    pd.testing.assert_series_equal(out, df.mean(axis=1), check_names=False)
