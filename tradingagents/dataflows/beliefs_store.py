"""Append-only belief store for FinCon CVRF reflection (Tier B1).

Each weekly reflection emits a 3-5 sentence "investment belief" per
coin — distilled from the past 7 days of trade decisions, returns, and
post-mortems. Beliefs are injected into the next-decision context for
modulator / factual / subjective agents so the LLM stack carries an
explicit episodic memory of what worked vs what didn't.

Layout: ``data/beliefs/weekly.parquet`` (single file, idempotent on
``(week_end, coin)``).

Schema:
  week_end                datetime64[ns, UTC]
  coin                    string
  belief_text             string
  supporting_trades_json  string   JSON-serialised list of trade ids / pnl
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PATH = Path("data/beliefs/weekly.parquet")

SCHEMA_COLS = [
    "week_end",
    "coin",
    "belief_text",
    "supporting_trades_json",
]


def upsert_beliefs(rows: list[dict], path: Path = DEFAULT_PATH) -> int:
    """Idempotent append on (week_end, coin)."""
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    missing = set(SCHEMA_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"upsert missing columns: {sorted(missing)}")
    df = df[SCHEMA_COLS].copy()
    df["week_end"] = pd.to_datetime(df["week_end"], utc=True)

    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        combined = df
    combined = combined.drop_duplicates(subset=["week_end", "coin"], keep="last")
    combined.to_parquet(path, index=False)
    return len(df)


def latest_belief(
    coin: str,
    as_of: Optional[datetime] = None,
    path: Path = DEFAULT_PATH,
) -> Optional[str]:
    """Return the most recent belief for ``coin`` at or before ``as_of``."""
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df = df[df["coin"] == coin.lower()]
        if as_of is not None:
            df = df[df["week_end"] <= pd.to_datetime(as_of, utc=True)]
        if df.empty:
            return None
        return str(df.sort_values("week_end").iloc[-1]["belief_text"])
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"latest_belief failed for {coin}: {exc}")
        return None


def encode_supporting(trades: list[dict]) -> str:
    return json.dumps(trades, default=str)
