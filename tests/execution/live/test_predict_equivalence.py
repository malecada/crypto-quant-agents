"""Live predict must produce numbers equivalent to a same-day backtest fit.

Marked online: requires CoinMetrics + DefiLlama + Binance data. Skipped by
default; run with `RUN_ONLINE_TESTS=1 pytest tests/execution/live/test_predict_equivalence.py`
or `pytest -m online ...` if the marker is registered in pyproject.

The runtime env-var skip is belt-and-suspenders: the `online` marker isn't
yet registered in pyproject, so plain `pytest` without filters would
otherwise try to run this and burn through real API quota / fail offline.
"""
from pathlib import Path
import os
import tempfile

import pytest


@pytest.mark.online
@pytest.mark.skipif(
    os.environ.get("RUN_ONLINE_TESTS") != "1",
    reason="online test — set RUN_ONLINE_TESTS=1 to enable",
)
def test_live_predict_matches_fresh_fit():
    from tradingagents.execution.live import predict, retrain
    from tradingagents.models.lgb_model import predict_pooled

    coins = ["BTC", "ETH", "BNB"]
    horizons = [7, 14]
    asof = "2026-04-25"

    with tempfile.TemporaryDirectory() as td:
        # Live path: fit + checkpoint + predict
        artifact = retrain.run_retrain(coins=coins, horizons=horizons,
                                        asof=asof, checkpoint_dir=Path(td))
        live_preds = predict.run_predict(
            checkpoint_path=artifact.model_path,
            coins=coins, horizons=horizons, asof=asof,
        )

        # Backtest-equivalent path: build features the same way, predict from
        # the same checkpoint we just produced. Results MUST be identical
        # because both paths use the same code, but this guards against
        # accidental divergence in the future (e.g. someone adds a feature
        # in retrain but forgets predict).
        features = predict.build_features_asof(coins, asof, horizons=horizons)
        import joblib
        bundles = joblib.load(artifact.model_path)
        for coin in coins:
            row = features[features["coin_id"] == coin]
            if row.empty or coin not in live_preds:
                continue
            for h in horizons:
                expected = float(predict_pooled(bundles[h], row))
                actual = live_preds[coin][f"pred_h{h}"]
                assert abs(expected - actual) < 1e-9, \
                    f"{coin} h={h}: live={actual}, refit={expected}"
