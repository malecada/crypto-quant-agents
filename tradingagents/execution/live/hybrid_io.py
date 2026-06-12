# tradingagents/execution/live/hybrid_io.py
"""Read-only access to the quant journal for the hybrid cycle.

The Journal class has no reader for the predictions table, so we open our own
sqlite3 connection (WAL is enabled by the writer) and SELECT directly — the
same pattern the runner uses for its frequency-guard read (runner.py:554-561).
"""
from __future__ import annotations
import sqlite3


def read_cycle_predictions(
    quant_db_path: str, cycle_id: str
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Return (rows, preds) for a cycle.

    rows  = list[{coin, horizon, prediction, ref_price}]  (for stage_quant_preds)
    preds = {coin: {"ref_price": float, "pred_h7": float, "pred_h14": float}}
            (for sizer.compute_size). Only coins with both horizons present
            appear in preds.
    """
    conn = sqlite3.connect(quant_db_path)
    try:
        cur = conn.execute(
            "SELECT coin, horizon, pred_value, ref_price "
            "FROM predictions WHERE cycle_id = ? ORDER BY coin, horizon",
            (cycle_id,),
        )
        raw = cur.fetchall()
    finally:
        conn.close()

    rows: list[dict] = []
    preds: dict[str, dict] = {}
    for coin, horizon, pred_value, ref_price in raw:
        rows.append({"coin": coin, "horizon": int(horizon),
                     "prediction": float(pred_value), "ref_price": float(ref_price)})
        d = preds.setdefault(coin, {"ref_price": float(ref_price)})
        d[f"pred_h{int(horizon)}"] = float(pred_value)
    # drop coins missing either horizon
    preds = {c: d for c, d in preds.items() if "pred_h7" in d and "pred_h14" in d}
    return rows, preds
