"""Extend DefiLlama free data into bitemporal on-chain store.

Adds:
  - Per-stablecoin daily mcap (USDT, USDC, DAI, USDe) → (coin="global", metric="stable_<sym>_mcap")
  - Chain TVL for Arbitrum, Solana, Polygon, Base → (coin="global", metric="tvl_<chain>")
  - Total DEX 7d-rolling volume → (coin="global", metric="dex_vol_total_7d")

Free, no auth. Daily granularity. Upserted into the existing PIT store with
as_of_ts = event_ts + 1 day to mirror DefiLlama's typical publication lag.

Usage:
    python scripts/fetch_defillama_extensions.py --since 2020-01-01
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.onchain_store import upsert_rows  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# DefiLlama stablecoin IDs (verified via /stablecoins endpoint 2026-05).
STABLES = {
    "usdt": 1,
    "usdc": 2,
    "dai": 5,
    "usde": 146,
}

# DefiLlama chain slugs.
CHAINS = ["Arbitrum", "Solana", "Polygon", "Base", "op-mainnet"]

PUB_LAG = timedelta(days=1)


def _get_json(url: str, label: str) -> object:
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            logger.warning("%s attempt %d failed: %s", label, attempt + 1, e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{label}: failed after retries")


def fetch_stable_mcap(sym: str, stable_id: int, since: pd.Timestamp) -> pd.DataFrame:
    url = f"https://stablecoins.llama.fi/stablecoincharts/all?stablecoin={stable_id}"
    data = _get_json(url, f"stable {sym}")
    rows = []
    for item in data:
        ts = pd.Timestamp(int(item["date"]), unit="s", tz="UTC")
        if ts < since:
            continue
        mcap = item.get("totalCirculatingUSD", {}).get("peggedUSD")
        if mcap is None or mcap == 0:
            continue
        rows.append({"event_ts": ts, "value": float(mcap)})
    return pd.DataFrame(rows)


def fetch_chain_tvl(chain: str, since: pd.Timestamp) -> pd.DataFrame:
    url = f"https://api.llama.fi/v2/historicalChainTvl/{chain}"
    data = _get_json(url, f"chain {chain}")
    rows = []
    for item in data:
        ts = pd.Timestamp(int(item["date"]), unit="s", tz="UTC")
        if ts < since:
            continue
        tvl = item.get("tvl")
        if tvl is None or tvl == 0:
            continue
        rows.append({"event_ts": ts, "value": float(tvl)})
    return pd.DataFrame(rows)


def fetch_dex_vol_total(since: pd.Timestamp) -> pd.DataFrame:
    url = "https://api.llama.fi/overview/dexs?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true"
    data = _get_json(url, "dex total")
    chart = data.get("totalDataChart", [])
    rows = []
    for item in chart:
        ts = pd.Timestamp(int(item[0]), unit="s", tz="UTC")
        if ts < since:
            continue
        vol = item[1]
        if vol is None:
            continue
        rows.append({"event_ts": ts, "value": float(vol)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.set_index("event_ts").sort_index()
    df["value"] = df["value"].rolling(7, min_periods=3).mean()
    df = df.dropna().reset_index()
    return df


def _wrap_rows(df: pd.DataFrame, coin: str, metric: str, source: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["event_ts"] = pd.to_datetime(out["event_ts"], utc=True)
    out["as_of_ts"] = out["event_ts"] + PUB_LAG
    out["coin"] = coin
    out["metric"] = metric
    out["source"] = source
    out["status"] = "final"
    return out[["event_ts", "as_of_ts", "coin", "metric", "value", "source", "status"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="2020-01-01")
    p.add_argument("--root", default="data/onchain")
    args = p.parse_args()

    since = pd.Timestamp(args.since, tz="UTC")
    all_frames: list[pd.DataFrame] = []

    for sym, sid in STABLES.items():
        logger.info("Fetching stable mcap: %s (id=%d)", sym, sid)
        raw = fetch_stable_mcap(sym, sid, since)
        wrapped = _wrap_rows(raw, "global", f"stable_{sym}_mcap", "defillama")
        if not wrapped.empty:
            logger.info("  %s: %d rows %s → %s",
                        sym, len(wrapped),
                        wrapped["event_ts"].min().date(), wrapped["event_ts"].max().date())
            all_frames.append(wrapped)
        time.sleep(0.4)

    for chain in CHAINS:
        logger.info("Fetching chain TVL: %s", chain)
        try:
            raw = fetch_chain_tvl(chain, since)
        except RuntimeError as exc:
            logger.warning("  %s: skipping (%s)", chain, exc)
            continue
        wrapped = _wrap_rows(raw, "global", f"tvl_{chain.lower()}", "defillama")
        if not wrapped.empty:
            logger.info("  %s: %d rows %s → %s",
                        chain, len(wrapped),
                        wrapped["event_ts"].min().date(), wrapped["event_ts"].max().date())
            all_frames.append(wrapped)
        time.sleep(0.4)

    logger.info("Fetching DEX total volume (7d rolling)")
    raw = fetch_dex_vol_total(since)
    wrapped = _wrap_rows(raw, "global", "dex_vol_total_7d", "defillama")
    if not wrapped.empty:
        logger.info("  dex_vol_total_7d: %d rows %s → %s",
                    len(wrapped),
                    wrapped["event_ts"].min().date(), wrapped["event_ts"].max().date())
        all_frames.append(wrapped)

    if not all_frames:
        logger.warning("No data fetched")
        return

    combined = pd.concat(all_frames, ignore_index=True)
    logger.info("Total rows: %d across %d metrics", len(combined), combined["metric"].nunique())
    written = upsert_rows(combined, root=Path(args.root))
    logger.info("Upserted; total rows in touched shards now: %d", written)


if __name__ == "__main__":
    main()
