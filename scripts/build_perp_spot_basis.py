"""Compute daily perp-spot basis from Binance Futures + Spot klines (public).

Fetches daily klines for BTCUSDT / ETHUSDT (perp and spot), aligns by date,
computes annualized basis = (perp_close - spot_close) / spot_close * 365,
and writes to ``data/derivatives_raw/{symbol}_basis.parquet`` plus appends
``perp_price`` + ``spot_price`` + ``basis_annual`` columns to the existing
``data/derivatives/{coin}.parquet`` daily aggregate.

Usage:
    python scripts/build_perp_spot_basis.py --symbols BTCUSDT ETHUSDT --start 2021-11-01
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.strategies.v3.features._http import with_backoff  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_PERP_URL = "https://fapi.binance.com/fapi/v1/klines"
_SPOT_URL = "https://api.binance.com/api/v3/klines"


def _fetch_kline_page(url: str, symbol: str, start_ms: int, end_ms: int, limit: int = 1500) -> list[list]:
    resp = requests.get(
        url,
        params={"symbol": symbol, "interval": "1d", "startTime": start_ms, "endTime": end_ms, "limit": limit},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_klines(url: str, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    while cursor_ms < end_ms:
        page = with_backoff(lambda: _fetch_kline_page(url, symbol, cursor_ms, end_ms))
        if not page:
            break
        rows.extend(page)
        last_open = page[-1][0]
        if last_open <= cursor_ms:
            break
        cursor_ms = last_open + 86_400_000
        time.sleep(0.1)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(
        {
            "open": [float(r[1]) for r in rows],
            "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows],
            "close": [float(r[4]) for r in rows],
            "volume": [float(r[5]) for r in rows],
        },
        index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
    )
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="first")].sort_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"])
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--cache-dir", default="data/derivatives_raw")
    p.add_argument("--daily-dir", default="data/derivatives")
    args = p.parse_args()

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp.utcnow().tz_convert("UTC").normalize() + pd.Timedelta(days=1)
    cache_dir = Path(args.cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    daily_dir = Path(args.daily_dir); daily_dir.mkdir(parents=True, exist_ok=True)
    symbol_to_coin = {"BTCUSDT": "bitcoin", "ETHUSDT": "ethereum", "BNBUSDT": "binancecoin", "SOLUSDT": "solana"}
    bak_suffix = pd.Timestamp.utcnow().strftime(".bak.%Y%m%d")

    for symbol in args.symbols:
        logger.info("%s: fetching perp klines", symbol)
        perp = fetch_klines(_PERP_URL, symbol, start, end)
        logger.info("%s: fetching spot klines", symbol)
        spot = fetch_klines(_SPOT_URL, symbol, start, end)
        if perp.empty or spot.empty:
            logger.warning("%s: empty klines (perp=%d, spot=%d) — skipping", symbol, len(perp), len(spot))
            continue

        perp_close = perp["close"].rename("perp_price")
        spot_close = spot["close"].rename("spot_price")
        basis = pd.concat([perp_close, spot_close], axis=1).dropna()
        basis["basis_annual"] = (basis["perp_price"] - basis["spot_price"]) / basis["spot_price"] * 365.0
        basis_file = cache_dir / f"{symbol}_basis.parquet"
        basis.to_parquet(basis_file)
        logger.info("%s: wrote %s (%d rows, basis mean=%+.4f stdev=%.4f)",
                    symbol, basis_file.name, len(basis),
                    float(basis["basis_annual"].mean()), float(basis["basis_annual"].std()))

        coin = symbol_to_coin.get(symbol)
        if coin is None:
            continue
        daily_file = daily_dir / f"{coin}.parquet"
        if daily_file.exists():
            existing = pd.read_parquet(daily_file)
            shutil.copy2(daily_file, daily_file.with_suffix(daily_file.suffix + bak_suffix))
        else:
            existing = pd.DataFrame()

        basis_for_merge = basis[["perp_price", "spot_price", "basis_annual"]].copy()
        if existing.empty:
            merged = basis_for_merge
        else:
            merged = existing.join(basis_for_merge, how="outer")
        merged = merged.sort_index()
        merged.to_parquet(daily_file)
        logger.info("%s: %s now %d rows, %d cols (%s)",
                    symbol, daily_file.name, len(merged), len(merged.columns), list(merged.columns))


if __name__ == "__main__":
    main()
