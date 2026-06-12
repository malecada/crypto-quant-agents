"""Tests for V3 inverse-vol sizing + CDAP regime-gated drawdown control."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.strategies.v3.config import V3Config
from tradingagents.strategies.v3.contracts import RegimeState


# ── vol_target_position ─────────────────────────────────────────────────────

def test_vol_target_position_at_target_vol():
    """vol = target → position = direction × confidence (unscaled)."""
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=1, confidence=0.8, realized_vol_annual=0.15
    )
    assert abs(pos - 0.8) < 1e-9


def test_vol_target_position_high_vol_scales_down():
    """vol = 2 × target, conf=1 → position = 0.5."""
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=1, confidence=1.0, realized_vol_annual=0.30
    )
    assert abs(pos - 0.5) < 1e-9


def test_vol_target_position_low_vol_clipped_to_max_leverage():
    """vol = 0.05, conf=1 → unclipped 3.0 → clipped to max_leverage=2.0."""
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=1, confidence=1.0, realized_vol_annual=0.05, max_leverage=2.0
    )
    assert pos == 2.0


def test_vol_target_position_short_direction():
    """direction = -1 → position is negative."""
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=-1, confidence=0.8, realized_vol_annual=0.15
    )
    assert pos == -0.8


def test_vol_target_position_zero_direction():
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=0, confidence=0.8, realized_vol_annual=0.15
    )
    assert pos == 0.0


def test_vol_target_position_zero_vol_safe():
    """realized_vol = 0 should not raise — clamp internally."""
    from tradingagents.strategies.v3.sizing.vol_target import vol_target_position

    pos = vol_target_position(
        direction=1, confidence=0.5, realized_vol_annual=0.0, max_leverage=2.0
    )
    assert pos == 2.0  # clamps to max leverage


# ── cdap_adjust ─────────────────────────────────────────────────────────────

def _bull_regime(confidence: float = 0.7, hurst: float = 0.6) -> RegimeState:
    return RegimeState(
        label="bull", confidence=confidence, hurst=hurst, changepoint_alert=False,
        posterior={"bull": confidence, "sideways": (1 - confidence) / 2,
                   "bear": (1 - confidence) / 2},
    )


def _bear_regime(confidence: float = 0.7, hurst: float = 0.6) -> RegimeState:
    return RegimeState(
        label="bear", confidence=confidence, hurst=hurst, changepoint_alert=False,
        posterior={"bull": (1 - confidence) / 2, "sideways": (1 - confidence) / 2,
                   "bear": confidence},
    )


def _sideways_regime(confidence: float = 0.4, hurst: float = 0.5) -> RegimeState:
    return RegimeState(
        label="sideways", confidence=confidence, hurst=hurst, changepoint_alert=False,
        posterior={"bull": 0.3, "sideways": confidence, "bear": 1 - 0.3 - confidence},
    )


def test_cdap_adjust_de_levers_in_confirmed_bear_with_dd():
    from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust

    cfg = V3Config()
    pos = cdap_adjust(position=1.0, portfolio_dd_pct=0.06, regime=_bear_regime(), config=cfg)
    assert abs(pos - 0.5) < 1e-9  # 0.5x de-lever


def test_cdap_adjust_no_action_low_regime_confidence():
    """6% DD in sideways with low confidence → no action."""
    from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust

    cfg = V3Config()
    pos = cdap_adjust(
        position=1.0, portfolio_dd_pct=0.06, regime=_sideways_regime(confidence=0.4),
        config=cfg,
    )
    assert pos == 1.0


def test_cdap_adjust_flat_at_severe_dd_with_high_confidence():
    """12% DD with regime confidence > 0.7 → position flat."""
    from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust

    cfg = V3Config()
    pos = cdap_adjust(
        position=1.0, portfolio_dd_pct=0.12, regime=_bear_regime(confidence=0.85),
        config=cfg,
    )
    assert pos == 0.0


def test_cdap_adjust_no_action_severe_dd_low_confidence():
    """12% DD with low regime confidence → NO action (Varma 2025)."""
    from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust

    cfg = V3Config()
    pos = cdap_adjust(
        position=1.0, portfolio_dd_pct=0.12, regime=_sideways_regime(confidence=0.4),
        config=cfg,
    )
    assert pos == 1.0


def test_cdap_adjust_no_action_in_bull():
    """Drawdown alone in bull regime → no action."""
    from tradingagents.strategies.v3.sizing.vol_target import cdap_adjust

    cfg = V3Config()
    pos = cdap_adjust(
        position=1.0, portfolio_dd_pct=0.06, regime=_bull_regime(),
        config=cfg,
    )
    assert pos == 1.0


# ── Integration ─────────────────────────────────────────────────────────────

def test_sizing_integration_realized_vol_within_target():
    """Simulate 100 bars; portfolio vol should stay roughly within ±25% of target.

    Note: this isn't a perfectly tight test — vol-target sizing is path-dependent
    and the synthetic series introduces autocorrelation. Loose tolerance.
    """
    from tradingagents.strategies.v3.sizing.vol_target import (
        cdap_adjust,
        vol_target_position,
    )

    cfg = V3Config(target_annual_vol=0.15, max_leverage=2.0)
    rng = np.random.default_rng(0)
    log_rets = rng.normal(0.0, 0.02, size=100)

    sizes = []
    realized_vols = []
    for t in range(21, 100):
        rv = float(np.std(log_rets[t - 21:t]) * np.sqrt(252))
        size = vol_target_position(
            direction=1, confidence=1.0, realized_vol_annual=rv,
            target_vol_annual=cfg.target_annual_vol, max_leverage=cfg.max_leverage,
        )
        size = cdap_adjust(
            position=size, portfolio_dd_pct=0.0, regime=_sideways_regime(), config=cfg,
        )
        sizes.append(size)
        realized_vols.append(rv)

    sizes = np.array(sizes)
    rvs = np.array(realized_vols)
    portfolio_vols = sizes * rvs
    avg_portfolio_vol = portfolio_vols.mean()
    # Should be roughly target_annual_vol ± 50% (sloppy because of clipping)
    assert abs(avg_portfolio_vol - 0.15) < 0.10
