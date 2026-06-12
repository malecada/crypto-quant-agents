import math
import numpy as np
import pandas as pd


def _fake_history(coin, days=60):
    dates = pd.date_range("2026-03-01", periods=days, freq="D")
    rng = np.random.default_rng(42)
    prices = 60000 * np.exp(np.cumsum(rng.normal(0, 0.02, days)))
    return pd.DataFrame({"date": dates, "close": prices, "coin": [coin] * days})


def test_sizer_long_signal_produces_positive_size():
    from tradingagents.execution.live.sizer import compute_size

    pred = {"ref_price": 60000.0, "pred_h7": 63000.0, "pred_h14": 66000.0}
    history = _fake_history("BTC")
    result = compute_size(
        coin="BTC", prediction=pred, price_history=history,
        horizons=[7, 14], symmetric=True,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        vol_lookback=20, vol_cap_pct=0.95, confidence_ref=0.02,
        trend_sma=30, trend_multiplier=1.5,
    )
    assert result.signal == 1
    assert result.final_size_notional > 0
    assert result.confidence > 0


def test_sizer_disagreeing_horizons_returns_zero_in_symmetric():
    from tradingagents.execution.live.sizer import compute_size

    pred = {"ref_price": 60000.0, "pred_h7": 63000.0, "pred_h14": 57000.0}
    history = _fake_history("BTC")
    result = compute_size(
        coin="BTC", prediction=pred, price_history=history,
        horizons=[7, 14], symmetric=True,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        vol_lookback=20, vol_cap_pct=0.95, confidence_ref=0.02,
        trend_sma=30, trend_multiplier=1.5,
    )
    assert result.signal == 0
    assert result.final_size_notional == 0


def test_sizer_caps_at_max_leverage():
    from tradingagents.execution.live.sizer import compute_size

    pred = {"ref_price": 60000.0, "pred_h7": 90000.0, "pred_h14": 95000.0}
    history = _fake_history("BTC")
    result = compute_size(
        coin="BTC", prediction=pred, price_history=history,
        horizons=[7, 14], symmetric=True,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        vol_lookback=20, vol_cap_pct=0.95, confidence_ref=0.02,
        trend_sma=30, trend_multiplier=1.5,
    )
    assert abs(result.final_size_notional) <= 3.0
