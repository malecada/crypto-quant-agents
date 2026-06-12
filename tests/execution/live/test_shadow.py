import numpy as np
import pandas as pd


def test_shadow_decision_for_same_inputs_matches_sizer():
    from tradingagents.execution.live.shadow import compute_shadow_decision
    from tradingagents.execution.live.sizer import compute_size

    pred = {"ref_price": 60000.0, "pred_h7": 63000.0, "pred_h14": 66000.0}
    rng = np.random.default_rng(42)
    history = pd.DataFrame({
        "date": pd.date_range("2026-03-01", periods=60, freq="D"),
        "close": 60000 * np.exp(np.cumsum(rng.normal(0, 0.02, 60))),
    })
    kwargs = dict(
        coin="BTC", prediction=pred, price_history=history,
        horizons=[7, 14], symmetric=True,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        vol_lookback=20, vol_cap_pct=0.95, confidence_ref=0.02,
        trend_sma=30, trend_multiplier=1.5,
    )
    sizer_result = compute_size(**kwargs)
    shadow_result = compute_shadow_decision(**kwargs)
    assert shadow_result.signal == sizer_result.signal
    assert shadow_result.size == sizer_result.final_size_notional
