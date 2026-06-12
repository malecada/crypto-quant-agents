#!/usr/bin/env python
"""Backfill on-chain metrics into the bitemporal onchain_store.

Sources (each emits rows in the canonical long-format schema):
  - coinmetrics : BTC + ETH network metrics (MVRV, flows, activity, ...)
  - defillama   : TVL, DEX volume, stablecoin market cap
  - beaconchain : ETH validator queue + ETH.STORE APR (ETH only)

Usage:
    python scripts/backfill_onchain.py \
        --start 2025-01-01 --end 2026-04-15 \
        --coins btc eth bnb \
        --sources coinmetrics defillama beaconchain
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows import coinmetrics, onchain_store  # noqa: E402

log = logging.getLogger("backfill_onchain")


CM_METRICS_BTC = [
    "AdrActCnt", "TxCnt", "HashRate", "CapMVRVCur", "CapMrktCurUSD",
    "FeeTotNtv", "FlowInExUSD", "FlowOutExUSD", "IssTotUSD", "SplyCur",
    "PriceUSD",
]
CM_METRICS_ETH = [
    "AdrActCnt", "TxCnt", "CapMVRVCur", "CapMrktCurUSD", "FeeTotNtv",
    "FlowInExUSD", "FlowOutExUSD", "IssTotUSD", "SplyCur", "PriceUSD",
]
CM_ASSET_MAP = {"btc": "btc", "eth": "eth", "bitcoin": "btc", "ethereum": "eth"}

# DefiLlama chain slugs per coin. BNB = BSC; BTC has no native DeFi so we map
# to an empty list (DefiLlama wrapped-BTC TVL lives on Ethereum).
DEFILLAMA_CHAIN_BY_COIN = {
    "btc": [],
    "eth": ["Ethereum"],
    "bnb": ["BSC"],
}

DEFILLAMA_BASE = "https://api.llama.fi"
DEFILLAMA_STABLES = "https://stablecoins.llama.fi"
BEACONCHAIN_BASE = "https://beaconcha.in/api/v1"

STABLE_LAG = timedelta(days=1)
FLASH_LAG = timedelta(days=7)


def parse_args():
    p = argparse.ArgumentParser(
        description="Backfill PIT on-chain metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD (UTC)")
    p.add_argument("--end", required=True, help="YYYY-MM-DD (UTC, exclusive)")
    p.add_argument("--coins", nargs="+", default=["btc", "eth", "bnb"])
    p.add_argument("--sources", nargs="+",
                   default=["coinmetrics", "defillama", "beaconchain"])
    p.add_argument("--out-dir", default="data/onchain")
    return p.parse_args()


def _mk_row(
    *, event_ts: datetime, value: float, coin: str, metric: str,
    source: str, status: str = "final",
    lag: timedelta = STABLE_LAG,
) -> dict:
    return {
        "event_ts": event_ts,
        "as_of_ts": event_ts + lag,
        "coin": coin.lower(),
        "metric": metric,
        "value": float(value),
        "source": source,
        "status": status,
    }


def backfill_coinmetrics(
    coins: list[str], start: datetime, end: datetime,
) -> pd.DataFrame:
    frames = []
    for coin in coins:
        asset = CM_ASSET_MAP.get(coin.lower())
        if asset is None:
            log.warning("CM: %s not supported — skipping", coin)
            continue
        metrics = CM_METRICS_BTC if asset == "btc" else CM_METRICS_ETH
        log.info("CM: fetching %s metrics for %s from %s to %s",
                 len(metrics), asset, start.date(), end.date())
        df = coinmetrics.fetch_asset_metrics_df(asset, metrics, start, end)
        if df.empty:
            log.warning("CM: no rows returned for %s", asset)
            continue
        # Normalize coin label back to canonical form used by store.
        df["coin"] = asset
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _llama_get(path: str, label: str) -> dict | list:
    backoff = 1.5
    for attempt in range(5):
        try:
            r = requests.get(path, timeout=30)
            if r.status_code == 429:
                log.warning("%s 429 — sleep %.1fs", label, backoff)
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            log.warning("%s error: %s (retry %d)", label, e, attempt + 1)
            time.sleep(backoff)
    raise RuntimeError(f"{label} failed after retries")


def backfill_defillama(
    coins: list[str], start: datetime, end: datetime,
) -> pd.DataFrame:
    rows: list[dict] = []
    # 1. Per-chain TVL for coins tied to a DeFi chain.
    for coin in coins:
        chains = DEFILLAMA_CHAIN_BY_COIN.get(coin.lower(), [])
        for chain in chains:
            url = f"{DEFILLAMA_BASE}/v2/historicalChainTvl/{chain}"
            log.info("DefiLlama: TVL %s → coin %s", chain, coin)
            payload = _llama_get(url, f"TVL {chain}")
            if not isinstance(payload, list):
                continue
            for item in payload:
                ts = datetime.fromtimestamp(item["date"], tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
                if not (start <= ts < end):
                    continue
                rows.append(_mk_row(
                    event_ts=ts, value=item["tvl"], coin=coin,
                    metric=f"tvl_{chain.lower()}", source="defillama",
                ))
            time.sleep(0.3)

    # 2. Total stablecoin market cap history (shared across coins — write as
    #    coin="global" so queries per coin can still union it in).
    log.info("DefiLlama: stablecoin history")
    stable_payload = _llama_get(
        f"{DEFILLAMA_STABLES}/stablecoincharts/all", "stablecoin history")
    if isinstance(stable_payload, list):
        for item in stable_payload:
            try:
                ts = datetime.fromtimestamp(int(item["date"]), tz=timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0)
            except (KeyError, ValueError, TypeError):
                continue
            if not (start <= ts < end):
                continue
            peg = item.get("totalCirculatingUSD")
            if isinstance(peg, dict):
                peg = peg.get("peggedUSD")
            if peg is None:
                continue
            rows.append(_mk_row(
                event_ts=ts, value=float(peg), coin="global",
                metric="stablecoin_mcap_total", source="defillama",
            ))
    return pd.DataFrame.from_records(rows, columns=onchain_store.SCHEMA_COLS) if rows else pd.DataFrame()


def backfill_beaconchain(
    coins: list[str], start: datetime, end: datetime,
) -> pd.DataFrame:
    # beaconcha.in dropped its free tier in 2026 — all endpoints now return
    # 401 without API key. Kept as a stub so the --sources arg is stable.
    # ETH staking coverage comes from CoinMetrics IssTotUSD post-Merge.
    log.warning(
        "beaconcha.in requires an API key as of 2026. Skipping. "
        "Set BEACONCHAIN_API_KEY and re-enable to use."
    )
    return pd.DataFrame()


SOURCE_DISPATCH = {
    "coinmetrics": backfill_coinmetrics,
    "defillama": backfill_defillama,
    "beaconchain": backfill_beaconchain,
}


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    out_root = Path(args.out_dir)

    total = 0
    for source in args.sources:
        fn = SOURCE_DISPATCH.get(source)
        if fn is None:
            log.error("Unknown source: %s", source)
            continue
        log.info("── source: %s ──", source)
        df = fn(args.coins, start, end)
        if df.empty:
            log.info("%s returned no rows", source)
            continue
        written = onchain_store.upsert_rows(df, root=out_root)
        log.info("%s: wrote %d rows", source, written)
        total += len(df)

    log.info("Backfill complete: %d rows ingested across sources", total)


if __name__ == "__main__":
    main()
