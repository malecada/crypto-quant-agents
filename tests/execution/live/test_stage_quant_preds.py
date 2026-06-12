# tests/execution/live/test_stage_quant_preds.py
import pandas as pd
from tradingagents.execution.live.hybrid_compose import stage_quant_preds

def test_stage_writes_h7_h14_with_required_columns(tmp_path):
    # cycle preds rows: (coin, horizon, prediction, ref_price)
    rows = [
        {"coin": "bitcoin", "horizon": 7,  "prediction": 0.012, "ref_price": 65000.0},
        {"coin": "bitcoin", "horizon": 14, "prediction": 0.020, "ref_price": 65000.0},
        {"coin": "ethereum","horizon": 7,  "prediction": -0.005,"ref_price": 3200.0},
        {"coin": "ethereum","horizon": 14, "prediction": -0.001,"ref_price": 3200.0},
    ]
    out = stage_quant_preds(rows, date="2026-06-11", out_dir=tmp_path)
    h7 = pd.read_csv(out / "preds_lgb_h7.csv")
    h14 = pd.read_csv(out / "preds_lgb_h14.csv")
    assert set(["date", "coin_id", "ref_price", "prediction"]).issubset(h7.columns)
    assert set(["date", "coin_id", "ref_price", "prediction"]).issubset(h14.columns)
    btc7 = h7[h7["coin_id"] == "bitcoin"].iloc[0]
    assert btc7["prediction"] == 0.012 and btc7["ref_price"] == 65000.0
    assert h14[h14["coin_id"] == "ethereum"].iloc[0]["prediction"] == -0.001

def test_stage_empty_rows_returns_dir_without_crash(tmp_path):
    # read_cycle_predictions can legitimately return rows=[] (no preds this cycle)
    out = stage_quant_preds([], date="2026-06-11", out_dir=tmp_path / "empty")
    assert out.exists()
