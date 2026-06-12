"""Tests for fit_pooled_full + predict_pooled — the live inference path."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_pooled():
    """Two coins, 200 rows each, target = noisy linear function of features."""
    rng = np.random.default_rng(0)
    rows = []
    for coin in ("BTC", "ETH"):
        for _ in range(200):
            f1 = rng.normal()
            f2 = rng.normal()
            target = 100 + 5 * f1 - 3 * f2 + rng.normal(0, 0.5)
            rows.append({"coin_id": coin, "feature_a": f1, "feature_b": f2,
                          "ref_price": 100.0, "prices_h7": target})
    return pd.DataFrame(rows)


def test_fit_pooled_full_returns_bundle(synthetic_pooled):
    from tradingagents.models.lgb_model import fit_pooled_full

    bundle = fit_pooled_full(synthetic_pooled, horizon=7)
    assert "booster" in bundle
    assert bundle["target_col"] == "prices_h7"
    assert bundle["horizon"] == 7
    assert bundle["n_train_rows"] == 400
    assert "feature_a" in bundle["feature_names"]
    assert "feature_b" in bundle["feature_names"]
    # ref_price and target should not appear as features
    assert "prices_h7" not in bundle["feature_names"]


def test_predict_pooled_returns_float(synthetic_pooled):
    from tradingagents.models.lgb_model import fit_pooled_full, predict_pooled

    bundle = fit_pooled_full(synthetic_pooled, horizon=7)
    feature_row = pd.DataFrame([{"feature_a": 1.0, "feature_b": -1.0,
                                  "ref_price": 100.0, "coin_id": "BTC"}])
    pred = predict_pooled(bundle, feature_row)
    assert isinstance(pred, float)
    # Predicted target ≈ 100 + 5*1 - 3*(-1) = 108; with noise it should be in a sane range
    assert 90 < pred < 130


def test_fit_pooled_full_drops_nan_targets():
    from tradingagents.models.lgb_model import fit_pooled_full

    df = pd.DataFrame({
        "coin_id": ["BTC"] * 50 + ["ETH"] * 50,
        "feature_a": np.random.default_rng(1).normal(0, 1, 100),
        "ref_price": [100.0] * 100,
        "prices_h7": [np.nan] * 10 + list(np.random.default_rng(2).uniform(50, 150, 90)),
    })
    bundle = fit_pooled_full(df, horizon=7)
    assert bundle["n_train_rows"] == 90  # 100 - 10 NaN
