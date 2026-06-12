"""V3 inverse-vol-target position sizing + CDAP regime-gated drawdown control.

Per spec §4.5:
- ``vol_target_position`` sizes positions inversely to realized volatility,
  scaled by directional signal × confidence, clipped to ±max_leverage.
- ``cdap_adjust`` (Varma 2025 CDAP framework) only de-levers / goes flat
  when drawdown coincides with a confirmed regime shift, not on arbitrary
  percentage thresholds.
"""

from __future__ import annotations

from tradingagents.strategies.v3.config import V3Config
from tradingagents.strategies.v3.contracts import RegimeState


_VOL_FLOOR = 1e-9  # avoid division-by-zero on degenerate inputs


def vol_target_position(
    direction: int,
    confidence: float,
    realized_vol_annual: float,
    target_vol_annual: float = 0.15,
    max_leverage: float = 2.0,
) -> float:
    """Compute leverage-clipped position from direction, confidence, realized vol.

    position = clip(direction × confidence × (target_vol / max(realized_vol, ε)),
                    -max_leverage, +max_leverage)
    """
    if direction not in (-1, 0, 1):
        raise ValueError(f"direction must be one of {{-1, 0, 1}}; got {direction}")
    if direction == 0:
        return 0.0
    confidence = max(0.0, min(1.0, float(confidence)))
    rv = max(float(realized_vol_annual), _VOL_FLOOR)
    base = direction * confidence * (target_vol_annual / rv)
    return float(max(-max_leverage, min(max_leverage, base)))


def cdap_adjust(
    position: float,
    portfolio_dd_pct: float,
    regime: RegimeState,
    config: V3Config,
) -> float:
    """Conditional Drawdown Action Protocol (Varma 2025).

    Logic (per spec §4.5):
      - dd > 5% AND regime in {bear} AND regime.confidence > 0.6 → 0.5x
      - dd > 10% AND regime.confidence > 0.7 → flat (0)
      - else: no action

    Note: drawdown alone (without regime confirmation) does NOT trigger
    de-leveraging — this is the central CDAP insight.
    """
    if portfolio_dd_pct > config.cdap_dd_flat and regime.confidence > 0.7:
        return 0.0
    if (
        portfolio_dd_pct > config.cdap_dd_de_lever
        and regime.label == "bear"
        and regime.confidence > config.cdap_min_regime_confidence
    ):
        return position * 0.5
    return position
