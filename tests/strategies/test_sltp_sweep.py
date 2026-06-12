# tests/strategies/test_sltp_sweep.py
"""Tests for take-profit extension to run_coin_backtest + sweep harness."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_strategy_v2 import run_coin_backtest  # noqa: E402

# Regression anchor for V5 MIX baseline (4.5-yr window 2021-11-07 → 2026-04-15).
# §20 thesis figure was 3.25; current canonical run is 3.178 (verified 2026-05-19,
# confirmed bit-identical after Task 3 refactor). Update after any deliberate
# data/model change; tolerance ±0.05 is intentionally wide to absorb minor drift.
_V5_ANCHOR_SR = 3.18
_V5_ANCHOR_TOL = 0.05


def test_take_profit_triggers_and_diverges_from_no_tp_path():
    """TP=5% fires on monotonically rising long → equity diverges from TP=off path.

    One-bar-flatten semantics (mirror of SL): when TP fires at bar i, the bar's
    return is already credited, target_pos is set to 0, and the position is
    re-entered at bar i+1 (entry_equity resets).  With a small round-trip fee,
    each forced exit+re-entry costs something, so eq_tp[-1] < eq_no_tp[-1].
    """
    n = 15
    dates = np.arange(n)
    prices = 100.0 * (1.01 ** np.arange(n))
    positions = np.ones(n)

    costs_with_fee = dict(
        fee_rate=0.001, slippage=0.0, spread=0.0,
        price_impact=0.0, funding_rate=0.0,
        max_portfolio_dd=1.0,  # disabled
    )
    common = dict(
        dates=dates, prices=prices, positions=positions,
        initial_capital=10_000.0, stop_loss=1.0,
        **costs_with_fee,
    )
    eq_no_tp, _ = run_coin_backtest(take_profit=0.0, **common)
    eq_tp, _ = run_coin_backtest(take_profit=0.05, **common)
    eq_no_tp = np.asarray(eq_no_tp)
    eq_tp = np.asarray(eq_tp)

    assert not np.allclose(eq_no_tp, eq_tp), \
        "TP=0.05 produced identical equity to TP=off — TP never fired or costs are zero"
    assert eq_tp[-1] < eq_no_tp[-1], (
        f"TP final {eq_tp[-1]:.2f} should be < no-TP {eq_no_tp[-1]:.2f} "
        "(each TP-triggered round-trip costs fees)"
    )


def test_take_profit_zero_is_bit_identical_to_no_tp_kwarg():
    """take_profit=0 must produce IDENTICAL equity to omitting the kwarg."""
    rng = np.random.default_rng(42)
    n = 200
    dates = np.arange(n)
    # synthetic price walk
    rets = rng.normal(0.0005, 0.02, size=n)
    prices = 100.0 * np.cumprod(1 + rets)
    # positions: mostly +1, occasional flat, occasional -1
    positions = rng.choice([-1.0, 0.0, 1.0], size=n, p=[0.3, 0.2, 0.5])

    common = dict(
        dates=dates, prices=prices, positions=positions,
        initial_capital=10_000.0, stop_loss=0.03,
        fee_rate=0.0004, slippage=0.0005, spread=0.0001,
        price_impact=0.00005, funding_rate=0.0001 / 8,
        max_portfolio_dd=0.15,
    )

    eq_no_kwarg, m_no = run_coin_backtest(**common)
    eq_tp_zero, m_tp = run_coin_backtest(take_profit=0.0, **common)

    np.testing.assert_array_equal(
        np.asarray(eq_no_kwarg), np.asarray(eq_tp_zero),
        err_msg="take_profit=0.0 changed equity vs no-kwarg path"
    )
    assert m_no == m_tp, "metrics dict diverged when take_profit=0.0"


def test_stop_loss_still_fires_when_take_profit_enabled():
    """With both SL and TP set, a falling position still exits via SL.

    The engine uses one-bar-flatten-then-re-entry semantics: when SL fires at
    bar i the full bar P&L is already credited, then prev_pos is set to 0.  If
    positions[i+1] is still non-zero the position is re-entered at bar i+1,
    paying an extra round-trip fee.  On a monotonically-falling price this
    means the SL-on path ends LOWER than the SL-off path (extra re-entry fees).
    The test therefore checks:
      (a) SL fires → equity paths diverge (not identical)
      (b) adding TP on top of SL does NOT suppress SL → SL+TP path == SL-only path
    """
    dates = np.arange(10)
    # Price falls 5%/bar from 100. Long position throughout.
    prices = 100.0 * (0.95 ** np.arange(10))
    positions = np.ones(10)

    costs = dict(
        fee_rate=0.001, slippage=0.0, spread=0.0,
        price_impact=0.0, funding_rate=0.0,
        max_portfolio_dd=1.0,  # disabled
    )

    # Path A: SL disabled, TP active
    eq_no_sl, _ = run_coin_backtest(
        dates=dates, prices=prices, positions=positions,
        initial_capital=10_000.0,
        stop_loss=1.0,         # SL effectively disabled
        take_profit=0.05,
        **costs,
    )
    # Path B: SL active, TP also active
    eq_sl_and_tp, _ = run_coin_backtest(
        dates=dates, prices=prices, positions=positions,
        initial_capital=10_000.0,
        stop_loss=0.03,
        take_profit=0.05,
        **costs,
    )
    # Path C: SL active, TP disabled (baseline for SL-in-isolation)
    eq_sl_only, _ = run_coin_backtest(
        dates=dates, prices=prices, positions=positions,
        initial_capital=10_000.0,
        stop_loss=0.03,
        take_profit=0.0,
        **costs,
    )
    eq_no_sl = np.asarray(eq_no_sl)
    eq_sl_and_tp = np.asarray(eq_sl_and_tp)
    eq_sl_only = np.asarray(eq_sl_only)

    # (a) SL fires: paths must diverge from the SL-off baseline.
    assert not np.allclose(eq_sl_and_tp, eq_no_sl), (
        "SL+TP path is identical to SL-disabled path — SL never fired"
    )
    # (b) TP does not suppress SL: SL+TP path must equal SL-only path on an
    #     adverse (falling) price where TP can never trigger.
    #     Guards against a future bug where the TP branch clobbers target_pos
    #     even when trade_up < take_profit (missing >0 guard, sign flip, etc).
    np.testing.assert_array_equal(
        eq_sl_and_tp, eq_sl_only,
        err_msg="SL+TP path diverged from SL-only path: TP incorrectly suppressed or altered SL"
    )


@pytest.mark.slow
def test_v5_baseline_reproduces_published_sharpe():
    """V5 MIX baseline cell (SL=0.03, EE=0.015, TP=off) must reproduce the
    canonical 4.5-yr WF Sharpe. Anchor + tolerance + rationale documented at
    _V5_ANCHOR_SR. Slow (~30s wall). Invoke: pytest -m slow.
    """
    from scripts.baseline_v5_mix import (  # noqa: E402
        COSTS, DEFAULT_ROUTING, run_coin,
    )

    start, end = "2021-11-07", "2026-04-15"
    coin_rets = {}
    for coin, pdir in DEFAULT_ROUTING.items():
        coin_rets[coin] = run_coin(
            coin, PROJECT_ROOT / pdir, start, end,
            kelly_fraction=0.5,
            early_exit_loss=0.015,
            costs_override=dict(COSTS),  # canonical V5 cost config; COSTS sets take_profit=0.0
        )

    df = pd.DataFrame(coin_rets).dropna()
    port = df.mean(axis=1)
    ann = np.sqrt(252)
    sr = float(port.mean() / port.std() * ann)

    assert abs(sr - _V5_ANCHOR_SR) < _V5_ANCHOR_TOL, (
        f"V5 baseline reproduction drifted: got SR={sr:.3f}, "
        f"expected {_V5_ANCHOR_SR} ± {_V5_ANCHOR_TOL}"
    )
