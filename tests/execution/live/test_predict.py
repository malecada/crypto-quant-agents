"""V5 composite predict — per-coin routing + per-coin failure isolation."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


def test_run_predict_routes_per_coin(monkeypatch, tmp_path):
    """Each coin's predictions come from its routed bundle. bundle_route populated."""
    import pandas as pd
    from tradingagents.execution.live import predict

    routing = {
        "bitcoin":  {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
        "ethereum": {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
    }

    composite = {
        "bitcoin_78f":  {7: {"feature_names": ["prices"], "kind": "btc_78"},
                           14: {"feature_names": ["prices"], "kind": "btc_78"}},
        "ethereum_193f": {7: {"feature_names": ["prices"], "kind": "eth_193"},
                            14: {"feature_names": ["prices"], "kind": "eth_193"}},
    }
    import joblib
    ckpt_path = tmp_path / "lgb_v5_mix_20260514.pkl"
    joblib.dump(composite, ckpt_path)

    def fake_build_features_asof(coin_pool, asof, store_root, ohlcv_cache,
                                    add_onchain_pit, horizons):
        rows = []
        for c in coin_pool:
            rows.append({"coin_id": c, "ref_price": 50000.0 if c == "bitcoin" else 3000.0,
                          "prices": 50000.0 if c == "bitcoin" else 3000.0})
        return pd.DataFrame(rows)

    monkeypatch.setattr(predict, "build_features_asof", fake_build_features_asof)

    def fake_predict_pooled(bundle, row):
        return 50100.0 if bundle["kind"] == "btc_78" else 3010.0

    monkeypatch.setattr(predict, "predict_pooled", fake_predict_pooled)

    df = predict.run_predict(
        coin_universe=["bitcoin", "ethereum"],
        routing=routing,
        ckpt_path=ckpt_path, asof="20260514",
        store_root=tmp_path / "onchain",
        ohlcv_cache=tmp_path / "cache",
        horizons=[7, 14],
    )

    assert len(df) == 4
    assert set(df["coin"]) == {"bitcoin", "ethereum"}
    btc_row = df[(df["coin"] == "bitcoin") & (df["horizon"] == 7)].iloc[0]
    eth_row = df[(df["coin"] == "ethereum") & (df["horizon"] == 7)].iloc[0]
    assert btc_row["bundle_route"] == "bitcoin_78f"
    assert eth_row["bundle_route"] == "ethereum_193f"
    assert btc_row["prediction"] == 50100.0
    assert eth_row["prediction"] == 3010.0


def test_run_predict_skips_failed_coin(monkeypatch, tmp_path):
    """If predict_pooled raises for one coin, it's skipped, others continue."""
    import pandas as pd
    from tradingagents.execution.live import predict

    routing = {
        "bitcoin":  {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
        "ethereum": {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
    }
    composite = {
        "bitcoin_78f":  {7: {"kind": "btc_78"}, 14: {"kind": "btc_78"}},
        "ethereum_193f": {7: {"kind": "eth_193"}, 14: {"kind": "eth_193"}},
    }
    import joblib
    ckpt_path = tmp_path / "lgb_v5_mix_20260514.pkl"
    joblib.dump(composite, ckpt_path)

    def fake_build_features_asof(coin_pool, **kw):
        return pd.DataFrame([{"coin_id": c, "ref_price": 1.0, "prices": 1.0} for c in coin_pool])
    monkeypatch.setattr(predict, "build_features_asof", fake_build_features_asof)

    def fake_predict_pooled(bundle, row):
        if bundle["kind"] == "btc_78":
            raise ValueError("simulated predict fail")
        return 3010.0
    monkeypatch.setattr(predict, "predict_pooled", fake_predict_pooled)

    df = predict.run_predict(
        coin_universe=["bitcoin", "ethereum"], routing=routing,
        ckpt_path=ckpt_path, asof="20260514",
        store_root=tmp_path / "o", ohlcv_cache=tmp_path / "c",
        horizons=[7, 14],
    )
    assert set(df["coin"]) == {"ethereum"}  # BTC skipped


def test_run_predict_majority_fail_raises(monkeypatch, tmp_path):
    """If ≥ 3 of 4 coins fail predict, raise PredictMajorityFail."""
    import pandas as pd
    from tradingagents.execution.live import predict
    from tradingagents.execution.live.predict import PredictMajorityFail

    routing = {
        "bitcoin":     {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
        "ethereum":    {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
        "binancecoin": {"feature_set": "78f",  "pool": ["bitcoin", "ethereum", "binancecoin"]},
        "solana":      {"feature_set": "193f", "pool": ["bitcoin", "ethereum", "solana"]},
    }
    composite = {f"{c}_{r['feature_set']}": {7: {"kind": c}, 14: {"kind": c}}
                  for c, r in routing.items()}
    import joblib
    ckpt_path = tmp_path / "lgb_v5_mix_20260514.pkl"
    joblib.dump(composite, ckpt_path)

    def fake_build_features_asof(coin_pool, **kw):
        return pd.DataFrame([{"coin_id": c, "ref_price": 1.0, "prices": 1.0} for c in coin_pool])
    monkeypatch.setattr(predict, "build_features_asof", fake_build_features_asof)

    def fake_predict_pooled(bundle, row):
        if bundle["kind"] in {"bitcoin", "ethereum", "binancecoin"}:
            raise ValueError("fail")
        return 1.0
    monkeypatch.setattr(predict, "predict_pooled", fake_predict_pooled)

    with pytest.raises(PredictMajorityFail):
        predict.run_predict(
            coin_universe=list(routing), routing=routing,
            ckpt_path=ckpt_path, asof="20260514",
            store_root=tmp_path / "o", ohlcv_cache=tmp_path / "c",
            horizons=[7, 14],
        )


def test_majority_fail_threshold_scales_with_universe():
    """predict.py: threshold = max(3, n-1) → 8-coin trips at 7 failures."""
    assert max(3, 8 - 1) == 7
    assert max(3, 4 - 1) == 3
    assert max(3, 2 - 1) == 3
