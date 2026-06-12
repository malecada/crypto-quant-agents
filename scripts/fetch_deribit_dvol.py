"""Fetch Deribit DVOL (implied volatility index) historical for BTC + ETH.

Deribit public endpoint /public/get_volatility_index_data returns OHLC of
DVOL at requested resolution. Free, no auth. Returns daily values from
~2021-06-09 (BTC) / ~2022-03-25 (ETH).

Writes to data/options/{currency}_dvol.parquet with columns:
  dvol_open, dvol_high, dvol_low, dvol_close

Usage:
    python scripts/fetch_deribit_dvol.py --currencies BTC ETH --start 2021-06-01
"""

from __future__ import annotations

import argparse
import logging
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

_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def _fetch_chunk(currency: str, start_ms: int, end_ms: int) -> list[list]:
    resp = requests.get(
        _URL,
        params={"currency": currency, "start_timestamp": start_ms, "end_timestamp": end_ms, "resolution": "86400"},
        timeout=15,
    )
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("result", {}).get("data", [])


def fetch_dvol(currency: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[list] = []
    chunk_size_ms = 500 * 86_400_000  # 500 days per request (Deribit cap)
    while cursor_ms < end_ms:
        chunk_end = min(cursor_ms + chunk_size_ms, end_ms)
        page = with_backoff(lambda: _fetch_chunk(currency, cursor_ms, chunk_end))
        if not page:
            break
        rows.extend(page)
        cursor_ms = chunk_end + 1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["dvol_open", "dvol_high", "dvol_low", "dvol_close"])
    df = pd.DataFrame(
        {
            "dvol_open": [float(r[1]) for r in rows],
            "dvol_high": [float(r[2]) for r in rows],
            "dvol_low": [float(r[3]) for r in rows],
            "dvol_close": [float(r[4]) for r in rows],
        },
        index=pd.to_datetime([r[0] for r in rows], unit="ms", utc=True),
    )
    df.index.name = "ts"
    return df[~df.index.duplicated(keep="first")].sort_index()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--currencies", nargs="+", default=["BTC", "ETH"])
    p.add_argument("--start", default="2021-06-01")
    p.add_argument("--out-dir", default="data/options")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp.utcnow().tz_convert("UTC").normalize() + pd.Timedelta(days=1)

    for currency in args.currencies:
        logger.info("%s: fetching DVOL %s → %s", currency, start.date(), end.date())
        df = fetch_dvol(currency, start, end)
        if df.empty:
            logger.warning("%s: empty", currency)
            continue
        out_file = out_dir / f"{currency.lower()}_dvol.parquet"
        df.to_parquet(out_file)
        logger.info(
            "%s: wrote %s (%d rows %s → %s; close mean=%.2f std=%.2f)",
            currency, out_file.name, len(df),
            df.index.min().date(), df.index.max().date(),
            float(df["dvol_close"].mean()), float(df["dvol_close"].std()),
        )


if __name__ == "__main__":
    main()
