"""Fit per-coin isotonic confidence calibrators (Tier B5).

Reads existing ``data/agent_signals_pit_p4/*.csv`` (one CSV per coin
with columns date, signal, confidence, trader_text), pairs each row's
verbalized confidence with the realised t+7 forward-return sign vs
the LLM's directional call, and fits an ``IsotonicCalibrator``
pickled to ``data/checkpoints/isotonic_{coin}.pkl``.

Forward returns are pulled from ``coingecko_binance._load_crypto_ohlcv``
so the script remains self-contained. Raw confidence in the CSVs is
either an integer 0-100, a 3-digit string "075", or a label
(LOW/MEDIUM/HIGH); we normalise to [0, 1].
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.calibration import IsotonicCalibrator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


_LABEL_TO_CONF = {"HIGH": 0.8, "MEDIUM": 0.5, "LOW": 0.2, "UNKNOWN": 0.5}
_SIGNAL_TO_DIR = {"BUY": 1, "OVERWEIGHT": 1, "HOLD": 0, "UNDERWEIGHT": -1, "SELL": -1}


def _parse_confidence(raw) -> float:
    if isinstance(raw, (int, float)) and not pd.isna(raw):
        v = float(raw)
        if v > 1.0:
            v /= 100.0
        return float(np.clip(v, 0.0, 1.0))
    s = str(raw).strip().upper()
    if s in _LABEL_TO_CONF:
        return _LABEL_TO_CONF[s]
    try:
        v = float(s) / 100.0
        return float(np.clip(v, 0.0, 1.0))
    except ValueError:
        return 0.5


def fit_for_coin(coin: str, csv_path: str, out_dir: str, horizon_days: int = 7) -> str:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise RuntimeError(f"empty signals CSV: {csv_path}")
    df["date"] = pd.to_datetime(df["date"])
    df["conf_raw"] = df["confidence"].apply(_parse_confidence)
    df["dir"] = df["signal"].astype(str).str.upper().map(_SIGNAL_TO_DIR).fillna(0)

    # Pull OHLCV through latest date in df
    last_date = df["date"].max().strftime("%Y-%m-%d")
    ohlcv = _load_crypto_ohlcv(coin, last_date)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
    ohlcv = ohlcv.sort_values("Date").reset_index(drop=True)
    price = ohlcv.set_index("Date")["Close"]

    # Forward return sign at t+horizon
    forward = price.shift(-horizon_days) / price - 1.0
    df = df.merge(
        forward.rename("forward_ret").reset_index().rename(columns={"Date": "date"}),
        on="date",
        how="left",
    )
    df = df.dropna(subset=["forward_ret"])

    # Outcome: 1 if direction call agrees with forward return sign
    df["outcome"] = ((df["dir"] * np.sign(df["forward_ret"])) > 0).astype(int)
    # HOLD calls (dir=0) treated as 1 if |forward_ret| < 1% (correct hold)
    is_hold = df["dir"] == 0
    df.loc[is_hold, "outcome"] = (df.loc[is_hold, "forward_ret"].abs() < 0.01).astype(int)

    if len(df) < 10:
        raise RuntimeError(f"too few labelled rows for {coin}: {len(df)}")

    cal = IsotonicCalibrator().fit(
        df["conf_raw"].values, df["outcome"].values, coin=coin
    )
    out_path = os.path.join(out_dir, f"isotonic_{coin}.pkl")
    cal.to_pkl(out_path)
    logger.info(
        f"{coin}: fit on {cal.n_train} rows, "
        f"raw_mean={df['conf_raw'].mean():.2f}, "
        f"hit_rate={df['outcome'].mean():.2f}, saved {out_path}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--signals-dir",
        default="data/agent_signals_pit_p4",
        help="Directory of <coin>_<start>_<end>.csv files",
    )
    p.add_argument("--out-dir", default="data/checkpoints")
    p.add_argument("--horizon-days", type=int, default=7)
    args = p.parse_args()

    if not os.path.isdir(args.signals_dir):
        raise SystemExit(f"signals dir not found: {args.signals_dir}")

    csvs = [f for f in os.listdir(args.signals_dir) if f.endswith(".csv")]
    if not csvs:
        raise SystemExit("no CSV files in signals dir")

    for fname in csvs:
        coin = fname.split("_")[0]
        try:
            fit_for_coin(coin, os.path.join(args.signals_dir, fname), args.out_dir, args.horizon_days)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"failed for {coin}: {exc}")


if __name__ == "__main__":
    main()
