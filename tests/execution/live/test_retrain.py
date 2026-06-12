from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


def _fake_transformed_df():
    """Pretend output of build_pooled_dataset + _transform_pooled — has prices_h7/h14."""
    n = 300
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "coin_id": (["BTC"] * 100) + (["ETH"] * 100) + (["BNB"] * 100),
        "ref_price": list(rng.uniform(50, 70000, n)),
        "feature_a": rng.normal(0, 1, n),
        "feature_b": rng.normal(0, 1, n),
        "prices_h7": list(rng.uniform(50, 70000, n)),
        "prices_h14": list(rng.uniform(50, 70000, n)),
    })


def test_retrain_writes_per_horizon_bundles(tmp_path):
    from tradingagents.execution.live import retrain

    fake_df = _fake_transformed_df()

    with patch.object(retrain, "build_pooled_dataset", return_value=fake_df), \
         patch.object(retrain, "_transform_pooled", return_value=fake_df), \
         patch.object(retrain, "fit_pooled_full") as mock_fit:
        # Return distinct bundles per horizon so we can verify both were called
        mock_fit.side_effect = lambda df, horizon, **kw: {
            "booster": object(),
            "feature_names": ["feature_a", "feature_b"],
            "horizon": horizon,
            "target_col": f"prices_h{horizon}",
            "n_train_rows": 300,
        }
        # V5 API: single-coin routing dict — equivalent coverage to the
        # legacy `coins=["BTC", "ETH", "BNB"]` call (one pool, one route).
        artifact = retrain.run_retrain(
            routing={"BTC": {"feature_set": "193f",
                              "pool": ["BTC", "ETH", "BNB"]}},
            horizons=[7, 14],
            asof="2026-05-11",
            checkpoint_dir=tmp_path,
        )
    assert artifact.path.exists()
    assert artifact.n_train_rows == 300
    assert len(artifact.sha) == 64
    # Both horizons fit (one route × 2 horizons)
    assert mock_fit.call_count == 2

    # Loaded checkpoint is a composite: {route_id: {horizon: bundle}}.
    import joblib
    loaded = joblib.load(artifact.path)
    assert set(loaded.keys()) == {"BTC_193f"}
    assert set(loaded["BTC_193f"].keys()) == {7, 14}
    assert loaded["BTC_193f"][7]["target_col"] == "prices_h7"
    assert loaded["BTC_193f"][14]["target_col"] == "prices_h14"


def test_retrain_fallback_atomic(monkeypatch, tmp_path):
    """If retrain raises, fallback returns the most recent prior composite."""
    from tradingagents.execution.live import retrain

    routing = {
        "bitcoin": {"feature_set": "78f", "pool": ["bitcoin", "ethereum"]},
    }

    # Seed a prior composite on disk
    prior_path = tmp_path / "lgb_v5_mix_20260513.pkl"
    import joblib
    joblib.dump({"bitcoin_78f": {7: {"x": 1}, 14: {"x": 2}}}, prior_path)

    def fake_retrain(**kw):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(retrain, "run_retrain", fake_retrain)

    artifact = retrain.run_retrain_with_fallback(
        routing=routing, horizons=[7, 14], asof="20260514",
        checkpoint_dir=tmp_path, retrain_id="cycle-test",
    )

    assert artifact.path == prior_path  # fell back


def test_retrain_atomic_no_half_file(monkeypatch, tmp_path):
    """If joblib.dump raises after start, no .pkl is left behind."""
    from tradingagents.execution.live import retrain
    import pandas as pd

    routing = {
        "bitcoin": {"feature_set": "78f", "pool": ["bitcoin", "ethereum"]},
    }

    monkeypatch.setattr(retrain, "build_pooled_dataset",
                         lambda **kw: pd.DataFrame({"prices": [1.0]},
                                                    index=pd.to_datetime(["2026-01-01"])))
    monkeypatch.setattr(retrain, "_transform_pooled",
                         lambda df, h: df.assign(prices_h7=1.0, prices_h14=1.0, coin_id="bitcoin"))
    monkeypatch.setattr(retrain, "fit_pooled_full",
                         lambda df, horizon: {"horizon": horizon, "feature_names": ["prices"],
                                                "booster": None, "scaler": None,
                                                "coin_to_int": {"bitcoin": 0},
                                                "n_train_rows": 1,
                                                "target_col": f"prices_h{horizon}"})

    def boom(*a, **k): raise RuntimeError("disk full")
    monkeypatch.setattr("joblib.dump", boom)

    with pytest.raises(RuntimeError):
        retrain.run_retrain(routing=routing, horizons=[7, 14], asof="20260514",
                              checkpoint_dir=tmp_path, retrain_id="x")
    # No lgb_v5_mix_*.pkl left behind (only the tmp may exist transiently)
    leftover = list(tmp_path.glob("lgb_v5_mix_*.pkl"))
    assert leftover == [], f"unexpected files: {leftover}"


def test_run_retrain_composite_four_routes(monkeypatch, tmp_path):
    """run_retrain produces composite bundle with 4 routes."""
    import pandas as pd
    from tradingagents.execution.live import retrain

    routing = {
        "bitcoin":     {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
        "ethereum":    {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
        "binancecoin": {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "binancecoin"]},
        "solana":      {"feature_set": "193f", "pool": ["bitcoin", "ethereum", "solana"]},
    }

    calls = []

    def fake_build_pooled_dataset(coin_universe, lookback_days, horizons, trade_date,
                                    add_technical, add_cross_asset, add_onchain, add_onchain_pit):
        calls.append({"pool": tuple(coin_universe), "pit": add_onchain_pit})
        return pd.DataFrame({"prices": [1.0]}, index=pd.to_datetime(["2026-01-01"]))

    def fake_transform_pooled(df, horizons):
        df = df.copy()
        for h in horizons:
            df[f"prices_h{h}"] = 1.0
        df["coin_id"] = "bitcoin"
        return df

    def fake_fit_pooled_full(df, horizon):
        return {"horizon": horizon, "feature_names": ["prices"],
                "booster": None, "scaler": None, "coin_to_int": {"bitcoin": 0},
                "n_train_rows": 1, "target_col": f"prices_h{horizon}"}

    monkeypatch.setattr(retrain, "build_pooled_dataset", fake_build_pooled_dataset)
    monkeypatch.setattr(retrain, "_transform_pooled", fake_transform_pooled)
    monkeypatch.setattr(retrain, "fit_pooled_full", fake_fit_pooled_full)

    artifact = retrain.run_retrain(
        routing=routing, horizons=[7, 14], asof="20260514",
        checkpoint_dir=tmp_path, retrain_id="cycle-test",
    )

    # Verify composite layout
    import joblib
    composite = joblib.load(artifact.path)
    assert set(composite.keys()) == {"bitcoin_78f", "ethereum_193f",
                                       "binancecoin_78f", "solana_193f"}
    assert set(composite["bitcoin_78f"].keys()) == {7, 14}

    # Verify 4 pools fetched with correct add_onchain_pit flags
    by_pool_pit = {(c["pool"], c["pit"]) for c in calls}
    assert (("bitcoin", "ethereum"), False) in by_pool_pit
    assert (("bitcoin", "ethereum"), True) in by_pool_pit
    assert (("bitcoin", "ethereum", "binancecoin"), False) in by_pool_pit
    assert (("bitcoin", "ethereum", "solana"), True) in by_pool_pit

    # Atomic: file exists, naming matches lgb_v5_mix_{asof}.pkl
    assert artifact.path.name == "lgb_v5_mix_20260514.pkl"
    assert artifact.routes == sorted(composite.keys())
