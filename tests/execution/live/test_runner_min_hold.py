"""P1: the runner's min-hold wiring contract.

Pins the freeze-base + reapply-trend behavior the runner relies on (held
positions keep the entry sleeve frozen and re-scale by the current bar's SMA
multiplier), and that entry/flip bars reproduce the old stateless size.
"""
from tradingagents.execution.live.hold_sizer import HoldState, step_hold_state


def test_entry_then_hold_freezes_base_and_reapplies_trend():
    # Bar 1: enter long, pre-trend base 0.40. On entry held_fraction =
    # base_target * sma_mult; with sma_mult 1.5 -> 0.60 (== old final_size_notional).
    st = HoldState(0, 0, 0.0, 0.0)
    st, base = step_hold_state(st, sig=1, vol_ok=True, fresh_base=0.40,
                               price=100.0, min_hold=7, early_exit_loss=0.015)
    assert base == 0.40
    assert abs(base * 1.5 - 0.60) < 1e-12
    assert st.current_dir == 1 and st.bars_held == 0

    # Bar 2: still long, within hold. A fresh re-size of 0.10 is IGNORED; base
    # stays frozen at 0.40. Today's sma_mult 0.5 -> held_fraction 0.20.
    st, base = step_hold_state(st, sig=1, vol_ok=True, fresh_base=0.10,
                               price=101.0, min_hold=7, early_exit_loss=0.015)
    assert base == 0.40
    assert st.bars_held == 1
    assert abs(base * 0.5 - 0.20) < 1e-12


def test_no_signal_bar_maintains_position_and_increments_bars():
    # Holding long; this bar has no signal (fresh_base 0). Position maintained,
    # bars_held increments (so the hold clock keeps running toward min_hold).
    st = HoldState(current_dir=1, bars_held=2, entry_price=100.0, entry_base=0.40)
    st, base = step_hold_state(st, sig=0, vol_ok=False, fresh_base=0.0,
                               price=102.0, min_hold=7, early_exit_loss=0.015)
    assert base == 0.40
    assert st.current_dir == 1 and st.bars_held == 3


def test_flip_blocked_until_min_hold():
    # Long, opposite signal arrives at bars_held<min_hold (winner, no early
    # exit) -> NO flip, base frozen.
    st = HoldState(current_dir=1, bars_held=2, entry_price=100.0, entry_base=0.40)
    st, base = step_hold_state(st, sig=-1, vol_ok=True, fresh_base=-0.35,
                               price=105.0, min_hold=7, early_exit_loss=0.015)
    assert st.current_dir == 1 and base == 0.40  # held, not flipped
    # Advance to min_hold then flip.
    for _ in range(5):
        st, base = step_hold_state(st, sig=1, vol_ok=True, fresh_base=0.40,
                                   price=106.0, min_hold=7, early_exit_loss=0.015)
    st, base = step_hold_state(st, sig=-1, vol_ok=True, fresh_base=-0.35,
                               price=106.0, min_hold=7, early_exit_loss=0.015)
    assert st.current_dir == -1 and base == -0.35  # flip now allowed
