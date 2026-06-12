"""Per-coin rolling LLM edge tracker.

The ``effective_weight`` formula uses ``rolling_llm_edge[coin]`` to
discount LLM influence on coins where the modulator has historically
hurt the quant baseline. Edge is the rolling Sharpe of
``(quant × multiplier)`` returns minus ``quant_only`` returns over the
last ``window_days``.

The trade journal at ``data/trade_journal.db`` records executed
positions with both the quant magnitude and the LLM multiplier, plus
the realised return on close. This module reads from there and writes
a per-(coin, as_of_date) Sharpe to ``data/rolling_edge.parquet`` so
the modulator can do an O(1) lookup.

PIT discipline: only trades whose realised exit is *strictly before*
``as_of_date`` are counted — open positions don't leak forward
information.

Cold-start: any coin with fewer than ``min_trades`` closed trades in
the window returns ``None`` from ``query_rolling_edge``. The
``effective_weight`` formula treats ``None`` as "no historical evidence"
and uses the structural regime prior unchanged.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/trade_journal.db")
DEFAULT_PARQUET = Path("data/rolling_edge.parquet")


def _load_closed_trades(db_path: Path = DEFAULT_DB) -> pd.DataFrame:
    """Load closed trades from the trade journal.

    The journal schema is the one in ``execution/logger.py``: trades
    table with ``coin, entry_ts, exit_ts, entry_price, exit_price,
    quant_magnitude, llm_multiplier, side``. Schema may evolve — return
    an empty frame on any error so cold start is graceful.
    """
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE exit_ts IS NOT NULL",
                conn,
            )
        if df.empty:
            return df
        for col in ("entry_ts", "exit_ts"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
        return df
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"trade journal read failed: {exc}")
        return pd.DataFrame()


def update_rolling_edge(
    as_of_date: pd.Timestamp,
    window_days: int = 30,
    min_trades: int = 10,
    db_path: Path = DEFAULT_DB,
    out_path: Path = DEFAULT_PARQUET,
) -> int:
    """Recompute rolling edge per coin and append to the parquet.

    Returns the number of (coin, as_of_date) rows written.
    """
    as_of = pd.to_datetime(as_of_date, utc=True)
    window_start = as_of - pd.Timedelta(days=window_days)
    trades = _load_closed_trades(db_path)
    if trades.empty:
        logger.info("no closed trades — rolling edge not updated")
        return 0

    trades = trades[trades["exit_ts"] < as_of]
    trades = trades[trades["exit_ts"] >= window_start]
    if trades.empty:
        return 0

    # Realised log return per trade. Side: ±1 (long/short).
    side = trades.get("side", pd.Series(["long"] * len(trades))).map(
        {"long": 1, "short": -1, "flat": 0}
    ).fillna(0)
    pct = (trades["exit_price"] / trades["entry_price"]) - 1.0
    realised_return = side * pct

    # Hybrid (quant × multiplier) vs pure-quant: position size scales
    # the same realised pct, but the *applied* multiplier is what we
    # paid for — we attribute (multiplier - 1) of the return to LLM.
    mult = trades.get("llm_multiplier", pd.Series([1.0] * len(trades)))
    quant_mag = trades.get("quant_magnitude", pd.Series([0.0] * len(trades)))
    hybrid_excess = (mult - 1.0) * realised_return * quant_mag

    out_rows = []
    for coin, sub in trades.assign(
        ret=realised_return,
        excess=hybrid_excess,
    ).groupby("coin"):
        if len(sub) < min_trades:
            continue
        excess = sub["excess"].values
        if np.std(excess) == 0:
            edge = 0.0
        else:
            edge = float(
                np.mean(excess) / np.std(excess) * np.sqrt(252)
            )
        out_rows.append({
            "coin": coin,
            "as_of_date": as_of,
            "edge": edge,
            "n_trades": int(len(sub)),
        })

    if not out_rows:
        return 0

    new_df = pd.DataFrame(out_rows)
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["coin", "as_of_date"], keep="last"
        )
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined = new_df

    combined.to_parquet(out_path, index=False)
    return len(out_rows)


def query_rolling_edge(
    coin: str,
    as_of_date,
    out_path: Path = DEFAULT_PARQUET,
) -> Optional[float]:
    """Return the most recent rolling edge for ``coin`` at or before ``as_of``.

    None on cold start (no parquet, no row, or fewer than min_trades).
    """
    if not out_path.exists():
        return None
    try:
        df = pd.read_parquet(out_path)
        as_of = pd.to_datetime(as_of_date, utc=True)
        df = df[(df["coin"] == coin.lower()) & (df["as_of_date"] <= as_of)]
        if df.empty:
            return None
        return float(df.sort_values("as_of_date").iloc[-1]["edge"])
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"rolling_edge query failed for {coin}: {exc}")
        return None
