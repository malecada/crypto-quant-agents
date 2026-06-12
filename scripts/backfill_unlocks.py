"""Backfill the token-unlock store from Tokenomist (or fallback CSV).

Tokenomist's free public API at ``api.tokenomist.ai`` is rate-limited
and may require auth tokens depending on plan tier. We attempt the
public endpoint, fall back to a manual CSV at ``data/unlocks/manual.csv``
if present, and otherwise warn + write zero rows. The plan's risk flag
calls for graceful degradation here.

Manual CSV schema (drop-in for offline ingestion):
  coin,unlock_date,amount_tokens,pct_circulating_supply,recipient_category
  binancecoin,2025-08-15T00:00:00Z,8000000,0.05,team
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.dataflows.unlocks import upsert_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TOKENOMIST_BASE = "https://api.tokenomist.ai/v1/unlocks"
MANUAL_CSV = "data/unlocks/manual.csv"


def _fetch_tokenomist(coin: str, through: str, timeout: float = 10.0) -> pd.DataFrame:
    """Call Tokenomist public API. Return empty DataFrame on any error."""
    try:
        r = requests.get(
            TOKENOMIST_BASE,
            params={"symbol": coin.upper(), "through": through},
            timeout=timeout,
        )
        if r.status_code != 200:
            logger.warning(
                f"tokenomist {coin}: HTTP {r.status_code} {r.text[:200]}"
            )
            return pd.DataFrame()
        data = r.json()
        rows = data.get("unlocks", []) if isinstance(data, dict) else data
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Normalise column names — Tokenomist's exact schema isn't pinned in
        # public docs; use defensive lookups.
        df = df.rename(columns={
            "date": "unlock_date",
            "amount": "amount_tokens",
            "pct_supply": "pct_circulating_supply",
            "category": "recipient_category",
        })
        if "unlock_date" not in df.columns:
            return pd.DataFrame()
        df["coin"] = coin.lower()
        df["source"] = "tokenomist"
        df["as_of_ts"] = datetime.now(tz=timezone.utc)
        for col in ("amount_tokens", "pct_circulating_supply"):
            if col not in df.columns:
                df[col] = 0.0
        if "recipient_category" not in df.columns:
            df["recipient_category"] = "unknown"
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"tokenomist {coin}: {exc}")
        return pd.DataFrame()


def _fetch_manual(coins: list[str]) -> pd.DataFrame:
    """Load the manual override CSV if present. Else empty."""
    if not os.path.exists(MANUAL_CSV):
        return pd.DataFrame()
    df = pd.read_csv(MANUAL_CSV)
    if df.empty:
        return df
    df = df[df["coin"].isin([c.lower() for c in coins])].copy()
    df["unlock_date"] = pd.to_datetime(df["unlock_date"], utc=True)
    if "as_of_ts" in df.columns:
        df["as_of_ts"] = pd.to_datetime(df["as_of_ts"], utc=True)
    else:
        df["as_of_ts"] = datetime.now(tz=timezone.utc)
    df["source"] = "manual"
    df["pct_circulating_supply"] = df.get("pct_circulating_supply", 0.0)
    if "recipient_category" not in df.columns:
        df["recipient_category"] = "unknown"
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--through", required=True, help="YYYY-MM-DD upper bound")
    args = p.parse_args()

    frames = []
    for coin in args.coins:
        df = _fetch_tokenomist(coin, args.through)
        if not df.empty:
            logger.info(f"tokenomist {coin}: {len(df)} rows")
            frames.append(df)

    manual = _fetch_manual(args.coins)
    if not manual.empty:
        logger.info(f"manual CSV: {len(manual)} rows")
        frames.append(manual)

    if not frames:
        logger.warning(
            "no unlock rows ingested (Tokenomist unreachable + no manual CSV); "
            "downstream code degrades to unlock_flag=False"
        )
        return

    combined = pd.concat(frames, ignore_index=True)
    n = upsert_rows(combined)
    logger.info(f"unlocks store: wrote {n} rows total")


if __name__ == "__main__":
    main()
