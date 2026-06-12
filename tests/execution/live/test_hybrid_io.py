# tests/execution/live/test_hybrid_io.py
from tradingagents.execution.live.journal import Journal
from tradingagents.execution.live.hybrid_io import read_cycle_predictions

def test_read_cycle_predictions_roundtrip(tmp_path):
    db = str(tmp_path / "trade_journal.db")
    j = Journal(db)
    j.log_cycle_start("2026-06-11", git_sha="abc")
    import pandas as pd
    preds_df = pd.DataFrame([
        {"coin": "bitcoin", "horizon": 7,  "prediction": 0.012, "ref_price": 65000.0, "bundle_route": "bitcoin_78f"},
        {"coin": "bitcoin", "horizon": 14, "prediction": 0.020, "ref_price": 65000.0, "bundle_route": "bitcoin_78f"},
        {"coin": "ethereum","horizon": 7,  "prediction": -0.005,"ref_price": 3200.0,  "bundle_route": "ethereum_193f"},
        {"coin": "ethereum","horizon": 14, "prediction": -0.001,"ref_price": 3200.0,  "bundle_route": "ethereum_193f"},
    ])
    j.record_predictions(cycle_id="2026-06-11", preds_df=preds_df)
    j.close()

    rows, preds = read_cycle_predictions(db, "2026-06-11")
    # rows: list of {coin, horizon, prediction, ref_price} (for staging)
    assert len(rows) == 4
    # preds: per-coin dict for sizer.compute_size
    assert preds["bitcoin"]["ref_price"] == 65000.0
    assert preds["bitcoin"]["pred_h7"] == 0.012
    assert preds["bitcoin"]["pred_h14"] == 0.020
    assert preds["ethereum"]["pred_h14"] == -0.001

def test_read_missing_cycle_returns_empty(tmp_path):
    db = str(tmp_path / "trade_journal.db")
    Journal(db).close()
    rows, preds = read_cycle_predictions(db, "2026-06-11")
    assert rows == [] and preds == {}
