"""Golden parity: step_hold_state must reproduce build_positions_with_hold.

The live single-step hold function, iterated bar-by-bar over the same inputs,
must produce the identical PRE-trend position series the backtest's
build_positions_with_hold produces in one vectorized pass.
"""
import numpy as np

from tradingagents.strategies.v2_sizing import (
    apply_leverage,
    build_positions_with_hold,
    vol_targeted_size,
)
from tradingagents.execution.live.hold_sizer import HoldState, step_hold_state


def _fresh_base(sig, conf, rv, target_vol, kelly, max_lev):
    return apply_leverage(
        vol_targeted_size(int(sig), conf, rv, target_vol, kelly), conf, max_lev
    )


def _run_live(signals, vol_ok, confidence, realized_vol, prices,
              *, target_vol, kelly, max_lev, min_hold, eel):
    state = HoldState(current_dir=0, bars_held=0, entry_price=0.0, entry_base=0.0)
    out = np.zeros(len(signals))
    for i in range(len(signals)):
        fresh = _fresh_base(signals[i], confidence[i], realized_vol[i],
                            target_vol, kelly, max_lev)
        state, base_target = step_hold_state(
            state, sig=int(signals[i]), vol_ok=bool(vol_ok[i]),
            fresh_base=fresh, price=float(prices[i]),
            min_hold=min_hold, early_exit_loss=eel,
        )
        out[i] = base_target
    return out


def test_step_matches_build_positions_with_hold_random():
    tv, kf, ml, mh, eel = 0.10, 0.25, 3.0, 7, 0.015
    for seed in range(8):
        rng = np.random.default_rng(seed)
        n = 150
        signals = rng.choice([-1, 0, 1], size=n)
        vol_ok = rng.random(n) > 0.2
        confidence = rng.random(n) * 0.5 + 0.5
        realized_vol = rng.random(n) * 0.04 + 0.01
        prices = 100 * np.cumprod(1 + rng.normal(0, 0.02, size=n))

        ref = build_positions_with_hold(
            signals, vol_ok, confidence, realized_vol, prices,
            target_vol=tv, kelly_fraction=kf, max_leverage=ml,
            min_hold=mh, early_exit_loss=eel,
        )
        live = _run_live(signals, vol_ok, confidence, realized_vol, prices,
                         target_vol=tv, kelly=kf, max_lev=ml, min_hold=mh, eel=eel)
        assert np.allclose(live, ref, atol=1e-12), f"mismatch seed={seed}"


def test_early_exit_then_same_bar_reentry():
    """A loser that breaches early_exit_loss with a flipped signal exits AND
    re-enters the new direction on the same bar — must match the backtest."""
    tv, kf, ml, mh, eel = 0.10, 0.25, 3.0, 7, 0.05
    # 6 bars: enter long bar0, drift down, bar4 flips to -1 with a big loss.
    signals = np.array([1, 1, 1, 1, -1, -1])
    vol_ok = np.array([True] * 6)
    confidence = np.array([0.8] * 6)
    realized_vol = np.array([0.02] * 6)
    prices = np.array([100.0, 99.0, 98.0, 97.0, 90.0, 89.0])
    ref = build_positions_with_hold(
        signals, vol_ok, confidence, realized_vol, prices,
        target_vol=tv, kelly_fraction=kf, max_leverage=ml,
        min_hold=mh, early_exit_loss=eel,
    )
    live = _run_live(signals, vol_ok, confidence, realized_vol, prices,
                     target_vol=tv, kelly=kf, max_lev=ml, min_hold=mh, eel=eel)
    assert np.allclose(live, ref, atol=1e-12)
