"""Single-step stateful min-hold for the live cycle.

Transcribes the hold layer of the validated backtest
(:func:`tradingagents.strategies.v2_sizing.build_positions_with_hold`) to ONE
bar, given the prior persisted state. Returns the PRE-trend signed base
position; the runner applies the current bar's SMA multiplier separately so a
held position tracks the daily trend multiplier exactly as ``apply_trend_filter``
does over the whole series in the backtest.

Live previously re-sized every cycle (stateless), which churns positions the
backtest would hold for >= ``min_hold`` bars — BT11 credits ~90% of V5's alpha
to this V2 sizing + hold discipline, so the deployed strategy was not the
validated one. This module closes that parity gap (P1).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HoldState:
    current_dir: int
    bars_held: int
    entry_price: float
    entry_base: float  # pre-trend signed size frozen at entry/flip


def step_hold_state(state: HoldState, *, sig: int, vol_ok: bool,
                    fresh_base: float, price: float,
                    min_hold: int, early_exit_loss: float) -> tuple[HoldState, float]:
    """Advance one bar of ``build_positions_with_hold``.

    Args:
        state: prior persisted state (``current_dir=0`` when flat).
        sig: this bar's consensus signal (+1 / -1 / 0).
        vol_ok: this bar's vol-regime gate.
        fresh_base: this bar's PRE-trend sized position
            (``apply_leverage(vol_targeted_size(...))``) — consumed only on a
            real entry or flip.
        price: this bar's reference (asof) close — for the early-exit PnL.
        min_hold: minimum bars to hold a winner before a flip is allowed.
        early_exit_loss: loss fraction past which a loser may exit early
            (after 3 bars) when the signal has also flipped/gone flat.

    Returns:
        (new_state, base_target) — ``base_target`` is the signed pre-trend
        position to hold this bar (frozen at the entry sleeve during a hold).

    The control flow mirrors ``v2_sizing.build_positions_with_hold`` lines
    159-198 for a single index ``i``; keep them in lockstep (a golden test
    asserts byte-for-byte parity).
    """
    current_dir = state.current_dir
    bars_held = state.bars_held
    entry_price = state.entry_price
    current_pos = state.entry_base  # frozen pre-trend base while holding

    if current_dir != 0:
        bars_held += 1

    # Early exit for losers (v2_sizing lines 166-173).
    if current_dir != 0 and bars_held >= 3 and bars_held < min_hold:
        if entry_price > 0 and price > 0:
            pnl = current_dir * (price - entry_price) / entry_price
            signal_changed = (sig != current_dir)
            if pnl < -early_exit_loss and signal_changed:
                current_pos = 0.0
                current_dir = 0
                bars_held = 0

    # Entry from flat (v2_sizing lines 176-183).
    if current_dir == 0 and sig != 0 and vol_ok:
        current_pos = fresh_base
        current_dir = sig
        bars_held = 0
        entry_price = price

    # Flip: only after hold expired AND signal reversed (v2_sizing lines 186-194).
    elif (current_dir != 0 and sig != 0 and sig != current_dir
          and bars_held >= min_hold and vol_ok):
        current_pos = fresh_base
        current_dir = sig
        bars_held = 0
        entry_price = price

    new_state = HoldState(current_dir=current_dir, bars_held=bars_held,
                          entry_price=entry_price, entry_base=current_pos)
    return new_state, current_pos
