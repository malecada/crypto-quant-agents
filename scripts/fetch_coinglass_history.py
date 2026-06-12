"""Pull full historical derivatives data from Coinglass v4 (Hobbyist tier).

Endpoints (all confirmed accessible on Hobbyist tier with full 5+yr history):
  - open-interest/aggregated-history          : cross-exchange OI OHLC (USD)
  - liquidation/aggregated-history            : long+short liq USD across 3 exchanges
  - global-long-short-account-ratio/history   : retail account-weighted L/S ratio
  - top-long-short-position-ratio/history     : top-trader position-weighted (smart-money)
  - top-long-short-account-ratio/history      : top-trader account-weighted
  - taker-buy-sell-volume/history             : aggressive buy vs sell ratio
  - funding-rate/oi-weight-history            : cross-exchange OI-weighted funding

Reads ``COINGLASS_API_KEY`` from env (load via python-dotenv from worktree .env).
Writes raw to ``data/derivatives_raw/{SYMBOL}_{slug}.parquet`` and aggregates
all into ``data/derivatives/{coin}.parquet`` for V3 runner consumption.

Usage:
    python scripts/fetch_coinglass_history.py
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

API_BASE = "https://open-api-v4.coinglass.com"
HOBBYIST_RATE_DELAY = 2.5  # 30/min hard cap → ~24/min safe

# Coin → (symbol-base, USDT-pair-symbol)
COIN_TO_SYMS = {
    "bitcoin":     ("BTC", "BTCUSDT"),
    "ethereum":    ("ETH", "ETHUSDT"),
    "binancecoin": ("BNB", "BNBUSDT"),
    "solana":      ("SOL", "SOLUSDT"),
}

# Cross-exchange list used for liquidation aggregation.
LIQ_EXCHANGES = "Binance,OKX,Bybit,Bitget,Bitmex,Bitfinex,dYdX,Kraken,CoinEx,HTX"

ENDPOINTS = {
    "oi_agg":          "/api/futures/open-interest/aggregated-history",
    "liq_agg":         "/api/futures/liquidation/aggregated-history",
    "ls_global":       "/api/futures/global-long-short-account-ratio/history",
    "ls_top_position": "/api/futures/top-long-short-position-ratio/history",
    "ls_top_account":  "/api/futures/top-long-short-account-ratio/history",
    "taker_vol":       "/api/futures/taker-buy-sell-volume/history",
    "funding_w":       "/api/futures/funding-rate/oi-weight-history",
}


def _request(path: str, params: dict, key: str) -> list[dict]:
    for attempt in range(4):
        r = requests.get(API_BASE + path, params=params, headers={"CG-API-KEY": key}, timeout=20)
        if r.status_code == 429:
            wait = 5 * (attempt + 1)
            logger.warning("Coinglass 429 — backoff %ds", wait)
            time.sleep(wait)
            continue
        r.raise_for_status()
        payload = r.json()
        if payload.get("code") != "0":
            raise RuntimeError(f"Coinglass error: {payload}")
        return payload.get("data", [])
    raise RuntimeError(f"Coinglass: failed after retries on {path}")


def fetch_oi_agg(symbol: str, key: str) -> pd.DataFrame:
    data = _request(ENDPOINTS["oi_agg"],
                    {"symbol": symbol, "interval": "1d", "exchanges": "Binance", "limit": 4500}, key)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    out = pd.DataFrame({
        "oi_open": df["open"].astype(float).values,
        "oi_high": df["high"].astype(float).values,
        "oi_low":  df["low"].astype(float).values,
        "oi_close": df["close"].astype(float).values,
    }, index=df["ts"])
    return out.sort_index()


def fetch_liq_agg(symbol: str, key: str) -> pd.DataFrame:
    data = _request(ENDPOINTS["liq_agg"],
                    {"symbol": symbol, "interval": "1d", "exchange_list": LIQ_EXCHANGES, "limit": 4500}, key)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    out = pd.DataFrame({
        "liq_long_usd":  df["aggregated_long_liquidation_usd"].astype(float).values,
        "liq_short_usd": df["aggregated_short_liquidation_usd"].astype(float).values,
    }, index=df["ts"])
    out["liq_total_usd"] = out["liq_long_usd"] + out["liq_short_usd"]
    out["liq_asym_24h"] = (out["liq_long_usd"] - out["liq_short_usd"]) / out["liq_total_usd"].replace(0.0, float("nan"))
    return out.sort_index()


def fetch_ls_ratio(slug: str, pair: str, key: str) -> pd.DataFrame:
    data = _request(ENDPOINTS[slug],
                    {"exchange": "Binance", "symbol": pair, "interval": "1d", "limit": 4500}, key)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    cols = [c for c in df.columns if c not in ("time", "ts")]
    out = df[cols].astype(float)
    out.index = df["ts"].values
    out.index.name = "ts"
    return out.add_prefix(f"{slug}_").sort_index()


def fetch_taker_vol(pair: str, key: str) -> pd.DataFrame:
    data = _request(ENDPOINTS["taker_vol"],
                    {"exchange": "Binance", "symbol": pair, "interval": "1d", "limit": 4500}, key)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    buy = df["taker_buy_volume_usd"].astype(float).values
    sell = df["taker_sell_volume_usd"].astype(float).values
    total = buy + sell
    asym = (buy - sell) / pd.Series(total).replace(0.0, float("nan"))
    out = pd.DataFrame({
        "taker_buy_vol_usd": buy,
        "taker_sell_vol_usd": sell,
        "taker_buy_sell_ratio": (buy / pd.Series(sell).replace(0.0, float("nan"))).values,
        "taker_asym": asym.values,
    }, index=df["ts"])
    return out.sort_index()


def fetch_funding_weighted(symbol: str, key: str) -> pd.DataFrame:
    data = _request(ENDPOINTS["funding_w"],
                    {"symbol": symbol, "interval": "1d", "limit": 4500}, key)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df["ts"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    out = pd.DataFrame({
        "funding_oiw_close": df["close"].astype(float).values,
    }, index=df["ts"])
    return out.sort_index()


def main() -> None:
    key = os.environ.get("COINGLASS_API_KEY")
    if not key:
        raise SystemExit("COINGLASS_API_KEY not set — add to .env")

    _data_root = Path(os.environ.get("TRADINGAGENTS_DATA_ROOT", "data"))
    cache_dir = _data_root / "derivatives_raw"; cache_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = _data_root / "derivatives"; daily_dir.mkdir(parents=True, exist_ok=True)
    bak_suffix = pd.Timestamp.utcnow().strftime(".bak.cg.%Y%m%d")

    for coin, (sym_base, pair) in COIN_TO_SYMS.items():
        logger.info("=== %s (%s / %s) ===", coin, sym_base, pair)

        frames: dict[str, pd.DataFrame] = {}
        # OI
        frames["oi"] = fetch_oi_agg(sym_base, key); time.sleep(HOBBYIST_RATE_DELAY)
        # Liquidations
        frames["liq"] = fetch_liq_agg(sym_base, key); time.sleep(HOBBYIST_RATE_DELAY)
        # L/S ratios (3 endpoints)
        for slug in ("ls_global", "ls_top_position", "ls_top_account"):
            frames[slug] = fetch_ls_ratio(slug, pair, key); time.sleep(HOBBYIST_RATE_DELAY)
        # Taker volume
        frames["taker"] = fetch_taker_vol(pair, key); time.sleep(HOBBYIST_RATE_DELAY)
        # Funding OI-weighted
        frames["funding_w"] = fetch_funding_weighted(sym_base, key); time.sleep(HOBBYIST_RATE_DELAY)

        for name, df in frames.items():
            if df.empty:
                logger.warning("  %s: empty", name)
                continue
            # Normalize index to tz-aware UTC for consistent joining
            if df.index.tz is None:
                df.index = pd.to_datetime(df.index).tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")
            frames[name] = df
            cache_file = cache_dir / f"{pair}_cg_{name}.parquet"
            df.to_parquet(cache_file)
            logger.info("  cached %s: %s (%d rows, %s → %s)",
                        name, cache_file.name, len(df),
                        df.index.min().date(), df.index.max().date())

        # Outer-join everything into the per-coin daily aggregate
        non_empty = [df for df in frames.values() if not df.empty]
        if not non_empty:
            logger.warning("  %s: nothing to merge", coin)
            continue
        merged_cg = pd.concat(non_empty, axis=1).sort_index()
        merged_cg.index = merged_cg.index.tz_convert("UTC") if merged_cg.index.tz is not None else merged_cg.index

        daily_file = daily_dir / f"{coin}.parquet"
        if daily_file.exists():
            existing = pd.read_parquet(daily_file)
            shutil.copy2(daily_file, daily_file.with_suffix(daily_file.suffix + bak_suffix))
        else:
            existing = pd.DataFrame()

        if not existing.empty and existing.index.tz is None:
            existing.index = pd.to_datetime(existing.index).tz_localize("UTC")
        if existing.empty:
            out = merged_cg
        else:
            # drop any pre-existing cg_* columns to avoid stale double-merge
            existing = existing.loc[:, ~existing.columns.str.startswith(("oi_", "liq_", "ls_", "taker_", "funding_oiw"))]
            out = existing.join(merged_cg, how="outer").sort_index()
        out.to_parquet(daily_file)
        logger.info("  → %s now %d rows × %d cols (%s)",
                    daily_file.name, len(out), len(out.columns), list(out.columns))


if __name__ == "__main__":
    main()
