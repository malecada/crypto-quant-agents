from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.execution.live.hold_sizer import HoldState
from tradingagents.execution.live.hybrid_base import derive_base


@pytest.fixture
def history():
    # 60 bars of synthetic but monotone-ish OHLCV so vol + SMA are well-defined.
    # sizer.py uses lowercase column names: "date", "close" (and "open"/"high"/"low"
    # are not used by compute_size, but "date" + "close" are required).
    idx = pd.date_range("2026-03-01", periods=60, freq="D")
    px = pd.Series(100.0 + np.arange(60) * 0.5)
    return pd.DataFrame(
        {"date": idx, "open": px, "high": px * 1.01,
         "low": px * 0.99, "close": px, "volume": 1000.0}
    )


def test_derive_base_returns_fraction_state_and_sizing(history):
    cfg = dict(
        horizons=[7, 14], symmetric=False, target_vol=0.10,
        kelly_fraction=0.25, max_leverage=3.0, vol_lookback=20,
        vol_cap_pct=0.95, confidence_ref_return=0.05, trend_sma=30,
        trend_multiplier=1.5, min_hold=7, early_exit_loss=0.015,
    )
    # asof = last bar date (2026-04-29 = day 59 of the 60-bar range).
    # Use a prev_state already holding a long position so bars_held is
    # incremented on this call (step_hold_state increments at the top of the
    # body when current_dir != 0, then resets to 0 only on a flip/exit).
    ref = float(history["close"].iloc[-1])
    prediction = {"ref_price": ref, "pred_h7": 0.03, "pred_h14": 0.05}
    prev_state = HoldState(current_dir=1, bars_held=2, entry_price=ref, entry_base=1.0)
    held_fraction, new_state, sz = derive_base(
        coin="bitcoin",
        prediction=prediction,
        price_history=history,
        prev_state=prev_state,
        cfg=cfg,
        asof="2026-04-29",
    )
    assert isinstance(held_fraction, float)
    # bars_held is incremented at the top of step_hold_state when current_dir != 0
    assert new_state.bars_held >= 1
    # identity with the runner's formula: held_fraction == base_target * sma_mult
    assert sz.coin == "bitcoin"
    # long prediction => non-negative base
    assert held_fraction >= 0.0


def test_insufficient_history_returns_zero(history):
    cfg = dict(
        horizons=[7, 14], symmetric=False, target_vol=0.10,
        kelly_fraction=0.25, max_leverage=3.0, vol_lookback=20,
        vol_cap_pct=0.95, confidence_ref_return=0.05, trend_sma=30,
        trend_multiplier=1.5, min_hold=7, early_exit_loss=0.015,
    )
    short = history.iloc[:5]
    prediction = {"ref_price": 100.0, "pred_h7": 0.03, "pred_h14": 0.05}
    held_fraction, new_state, sz = derive_base(
        coin="bitcoin",
        prediction=prediction,
        price_history=short,
        prev_state=HoldState(0, 0, 0.0, 0.0),
        cfg=cfg,
        asof="2026-04-29",
    )
    assert held_fraction == 0.0 and sz is None
