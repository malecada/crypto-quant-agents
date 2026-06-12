"""FinCon CVRF — weekly Conceptual Verbal Reflection (Tier B1).

Singhi (arXiv 2510.08068) reports +31% return on BTC from a weekly
Reflect agent that distills the past 7 days of decisions, returns, and
post-mortems into 3-5 sentence "investment beliefs" persisted to a
beliefs store. Modulator / factual / subjective agents read the latest
belief on each new decision so the LLM stack has an explicit episodic
memory of what worked vs what didn't.

Cadence: weekly cron via ``scripts/weekly_reflect.py``. NOT bolted onto
the per-trade ``Reflector`` in graph/reflection.py — that serves a
different purpose (post-trade situation memory for BM25 retrieval).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("data/trade_journal.db")


def _load_week_trades(
    coin: str,
    week_start: pd.Timestamp,
    week_end: pd.Timestamp,
    db_path: Path = DEFAULT_DB,
) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    try:
        with sqlite3.connect(str(db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trades WHERE coin = ? "
                "AND entry_ts >= ? AND entry_ts < ?",
                conn,
                params=[coin.lower(), str(week_start), str(week_end)],
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


def _trades_summary(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "No trades executed in the past 7 days."
    n = len(trades)
    closed = trades.dropna(subset=["exit_ts"]) if "exit_ts" in trades.columns else trades
    if closed.empty:
        return f"{n} trades opened, none closed."
    if "entry_price" in closed.columns and "exit_price" in closed.columns:
        side = closed.get("side", pd.Series(["long"] * len(closed))).map(
            {"long": 1, "short": -1, "flat": 0}
        ).fillna(0)
        ret = side * (closed["exit_price"] / closed["entry_price"] - 1.0)
        n_win = int((ret > 0).sum())
        avg = float(ret.mean()) * 100
        worst = float(ret.min()) * 100
        best = float(ret.max()) * 100
        return (
            f"{n} trades, {len(closed)} closed. Win rate: {n_win}/{len(closed)}. "
            f"Avg return: {avg:+.2f}%. Best: {best:+.2f}%, worst: {worst:+.2f}%."
        )
    return f"{n} trades, {len(closed)} closed. Numeric details unavailable."


class CVRFReflector:
    """Weekly belief distiller. Pure helper; not a LangGraph node."""

    def __init__(self, llm):
        self.llm = llm

    def reflect_week(
        self,
        coin: str,
        week_end: pd.Timestamp,
        db_path: Path = DEFAULT_DB,
    ) -> Optional[dict]:
        """Return ``{week_end, coin, belief_text, supporting_trades_json}``."""
        week_end = pd.to_datetime(week_end, utc=True)
        week_start = week_end - pd.Timedelta(days=7)
        trades = _load_week_trades(coin, week_start, week_end, db_path)
        summary = _trades_summary(trades)

        sys = (
            "You distill the past 7 days of crypto trading activity into "
            "3-5 sentence 'investment beliefs' that the next week's "
            "decisions will be conditioned on. Be concrete: cite signal "
            "types (LGB consensus, regime, on-chain flow, sentiment) that "
            "either helped or hurt. No hedging language. End with a single "
            "actionable rule for next week."
        )
        user = (
            f"Coin: {coin}\n"
            f"Week ending: {week_end.date()}\n\n"
            f"Trade summary:\n{summary}\n\n"
            "Write the 3-5 sentence belief now."
        )

        try:
            result = self.llm.invoke([
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ])
            belief = result.content if hasattr(result, "content") else str(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"CVRF reflect failed for {coin}: {exc}")
            return None

        from tradingagents.dataflows.beliefs_store import encode_supporting
        return {
            "week_end": week_end,
            "coin": coin.lower(),
            "belief_text": belief.strip()[:2000],
            "supporting_trades_json": encode_supporting(
                trades.to_dict(orient="records") if not trades.empty else []
            ),
        }
