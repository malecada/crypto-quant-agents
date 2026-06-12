"""Golden-value tests pinning v2_sizing functions to current backtest behavior."""
import math

import numpy as np
import pytest


def test_vol_targeted_size_basic():
    from tradingagents.strategies.v2_sizing import vol_targeted_size

    result = vol_targeted_size(
        signal=1, confidence=0.5, realized_vol=0.40,
        target_vol=0.10, kelly_fraction=0.5,
    )
    assert math.isclose(result, 0.0625, rel_tol=1e-9)


def test_vol_targeted_size_zero_signal_returns_zero():
    from tradingagents.strategies.v2_sizing import vol_targeted_size
    assert vol_targeted_size(0, 0.8, 0.40, 0.10, 0.5) == 0.0


def test_vol_targeted_size_nan_vol_returns_zero():
    from tradingagents.strategies.v2_sizing import vol_targeted_size
    assert vol_targeted_size(1, 0.5, float("nan"), 0.10, 0.5) == 0.0


def test_vol_targeted_size_zero_vol_returns_zero():
    from tradingagents.strategies.v2_sizing import vol_targeted_size
    assert vol_targeted_size(1, 0.5, 0.0, 0.10, 0.5) == 0.0


def test_vol_targeted_size_short_signal():
    from tradingagents.strategies.v2_sizing import vol_targeted_size
    result = vol_targeted_size(-1, 1.0, 0.20, 0.10, 0.5)
    assert math.isclose(result, -0.25, rel_tol=1e-9)


def test_apply_leverage_zero_base_returns_zero():
    from tradingagents.strategies.v2_sizing import apply_leverage
    assert apply_leverage(0.0, 0.8, 3.0) == 0.0


def test_apply_leverage_at_max_confidence_hits_3x_factor():
    from tradingagents.strategies.v2_sizing import apply_leverage
    result = apply_leverage(0.5, 1.0, 3.0)
    assert math.isclose(result, 1.5, rel_tol=1e-9)


def test_apply_leverage_capped_at_max_leverage():
    from tradingagents.strategies.v2_sizing import apply_leverage
    assert apply_leverage(2.0, 1.0, 3.0) == 3.0


def test_apply_leverage_short_capped_negative():
    from tradingagents.strategies.v2_sizing import apply_leverage
    assert apply_leverage(-2.0, 1.0, 3.0) == -3.0


def test_compute_realized_vol_uses_252():
    from tradingagents.strategies.v2_sizing import compute_realized_vol
    np.random.seed(42)
    prices = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, 50)))
    vol = compute_realized_vol(prices, lookback=20)
    assert np.all(np.isnan(vol[:20]))
    assert not np.isnan(vol[20])
    assert 0.1 < vol[-1] < 0.25


def test_vol_regime_mask_short_history_passes():
    from tradingagents.strategies.v2_sizing import vol_regime_mask
    vol = np.array([np.nan] * 5 + [0.2, 0.3, 0.25])
    mask = vol_regime_mask(vol, percentile_cap=0.95)
    assert mask.tolist() == [False] * 5 + [True, True, True]


def test_vol_regime_mask_caps_high_vol():
    from tradingagents.strategies.v2_sizing import vol_regime_mask
    history = list(np.linspace(0.1, 0.3, 25))
    vol = np.array(history + [1.5])
    mask = vol_regime_mask(vol, percentile_cap=0.95)
    assert bool(mask[-1]) is False


def test_term_structure_signals_symmetric_full_agreement():
    from tradingagents.strategies.v2_sizing import generate_term_structure_signals
    import pandas as pd

    df = pd.DataFrame({
        "ref_price": [100, 100, 100],
        "pred_h7":   [105, 95, 102],
        "pred_h14":  [110, 90, 99],
    })
    signals, conf = generate_term_structure_signals(
        df, horizons=[7, 14], confidence_ref=0.02, asymmetric=False,
    )
    assert signals.tolist() == [1.0, -1.0, 0.0]
    assert conf[0] == 1.0
    assert conf[1] == 1.0
    assert conf[2] == 0.0
