"""Predict must use the same add_onchain_pit flag the retrain used for that route."""
from __future__ import annotations

import pandas as pd
import joblib
import pytest


def test_predict_passes_correct_add_onchain_pit_per_route(monkeypatch, tmp_path):
    from tradingagents.execution.live import predict

    routing = {
        "bitcoin":  {"feature_set": "78f",  "pool": ["bitcoin", "ethereum"]},
        "ethereum": {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]},
    }
    composite = {
        "bitcoin_78f":  {7: {"kind": "btc_78"}},
        "ethereum_193f": {7: {"kind": "eth_193"}},
    }
    ckpt_path = tmp_path / "lgb_v5_mix_20260514.pkl"
    joblib.dump(composite, ckpt_path)

    captured: list[bool] = []

    def fake_build_features_asof(coin_pool, asof, store_root, ohlcv_cache,
                                    add_onchain_pit, horizons):
        captured.append(add_onchain_pit)
        return pd.DataFrame([{"coin_id": c, "ref_price": 1.0, "prices": 1.0}
                                for c in coin_pool])

    monkeypatch.setattr(predict, "build_features_asof", fake_build_features_asof)
    monkeypatch.setattr(predict, "predict_pooled", lambda b, r: 1.0)

    predict.run_predict(
        coin_universe=["bitcoin", "ethereum"], routing=routing,
        ckpt_path=ckpt_path, asof="20260514",
        store_root=tmp_path / "o", ohlcv_cache=tmp_path / "c",
        horizons=[7],
    )

    # 2 coins → 2 build_features_asof calls
    assert captured == [False, True]
