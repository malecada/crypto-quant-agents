"""Pull Binance aggTrades for a date range, compute daily microstructure features,
write to ``data/microstructure/{coin}.parquet``.

Usage:
    # REST API (slow, paginates per-1000-trades):
    python scripts/build_microstructure_features.py \\
        --coins bitcoin ethereum \\
        --start 2024-05-01 --end 2026-04-15

    # Binance Vision archive (fast, pre-built daily CSVs):
    python scripts/build_microstructure_features.py \\
        --coins bitcoin ethereum \\
        --start 2025-12-01 --end 2026-04-15 \\
        --use-vision \\
        --no-raw-cache \\
        --out-dir data/microstructure_vpin
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.strategies.v3.features.microstructure import (  # noqa: E402
    build_daily_microstructure_features,
    compute_vpin,
    compute_vpin_fast,
    fetch_aggtrades,
    fetch_aggtrades_vision,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_COIN_TO_SYMBOL = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
}


def _aggregate_one_day(day_df: pd.DataFrame) -> dict:
    """Compute per-day VPIN + OFI stats from tick-level trades.

    Uses the vectorised ``compute_vpin_fast`` for performance on large datasets
    (BTC has ~2M trades/day; vectorised is ~1000x faster than the iterator).
    """
    ibm = day_df["is_buyer_maker"].values.astype(bool)
    qty = day_df["qty"].values.astype(float)
    sell_vol = float(qty[ibm].sum())
    buy_vol = float(qty[~ibm].sum())
    total = sell_vol + buy_vol
    ofi = (buy_vol - sell_vol) / total if total > 0 else 0.0
    aggressor = buy_vol / total if total > 0 else 0.0
    vpin = compute_vpin_fast(day_df, n_buckets=50)
    return {
        "vpin_50": vpin,
        "ofi_d": ofi,
        "aggressor_ratio": aggressor,
        "_buy_vol": buy_vol,
        "_sell_vol": sell_vol,
    }


def _finalize_features(
    rows: list[dict],
    dates: list,
    z_window: int = 30,
    weekly_window: int = 7,
) -> pd.DataFrame:
    """Turn per-day stats into full feature DataFrame with Z-scores and weekly OFI."""
    df = pd.DataFrame(rows, index=pd.Index(dates, name="date")).sort_index()
    df["vpin_50_z"] = (
        (df["vpin_50"] - df["vpin_50"].rolling(z_window).mean())
        / df["vpin_50"].rolling(z_window).std()
    )
    weekly_buy = df["_buy_vol"].rolling(weekly_window).sum()
    weekly_sell = df["_sell_vol"].rolling(weekly_window).sum()
    df["ofi_d_w"] = (weekly_buy - weekly_sell) / (
        weekly_buy + weekly_sell
    ).replace(0.0, np.nan)
    df = df.drop(columns=["_buy_vol", "_sell_vol"])
    return df[["vpin_50", "vpin_50_z", "ofi_d", "ofi_d_w", "aggressor_ratio"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--cache-dir", default="data/microstructure_raw")
    parser.add_argument("--out-dir", default="data/microstructure")
    parser.add_argument(
        "--use-vision",
        action="store_true",
        default=False,
        help="Use Binance Vision archive instead of REST API pagination",
    )
    parser.add_argument(
        "--no-raw-cache",
        action="store_true",
        default=False,
        help=(
            "Process day-by-day in memory without caching raw parquets. "
            "Saves disk space (recommended for Vision with limited disk)."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Fetcher: %s | cache: %s",
        "Vision" if args.use_vision else "REST API",
        "disabled (no-raw-cache)" if args.no_raw_cache else str(cache_dir),
    )

    dates = pd.date_range(args.start, args.end, freq="D", tz="UTC")

    for coin in args.coins:
        symbol = _COIN_TO_SYMBOL[coin]
        logger.info("Processing %s (%s) for %d days", coin, symbol, len(dates))
        t_start = time.time()

        if args.no_raw_cache:
            # Day-by-day streaming: fetch → aggregate → discard raw trades
            day_rows: list[dict] = []
            day_dates: list[pd.Timestamp] = []
            skipped = 0
            failed = 0

            for i, d in enumerate(dates):
                day_str = d.strftime("%Y-%m-%d")
                try:
                    if args.use_vision:
                        # Fetch without caching: use a temp dir that we clean up
                        import io
                        import zipfile
                        import requests

                        url = (
                            f"https://data.binance.vision/data/spot/daily/aggTrades"
                            f"/{symbol}/{symbol}-aggTrades-{day_str}.zip"
                        )
                        resp = requests.get(url, timeout=60)
                        if resp.status_code == 404:
                            logger.warning("404: %s %s", symbol, day_str)
                            skipped += 1
                            continue
                        resp.raise_for_status()
                        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                            csv_name = f"{symbol}-aggTrades-{day_str}.csv"
                            with zf.open(csv_name) as f:
                                df_raw = pd.read_csv(
                                    f,
                                    header=None,
                                    names=[
                                        "aggTradeId", "price", "qty",
                                        "firstTradeId", "lastTradeId",
                                        "timestamp", "is_buyer_maker", "isBestMatch",
                                    ],
                                )
                        if df_raw["timestamp"].iloc[-1] > 10**14:
                            ts = pd.to_datetime(
                                df_raw["timestamp"], unit="us", utc=True
                            )
                        else:
                            ts = pd.to_datetime(
                                df_raw["timestamp"], unit="ms", utc=True
                            )
                        day_df = pd.DataFrame(
                            {
                                "price": df_raw["price"].values.astype(float),
                                "qty": df_raw["qty"].values.astype(float),
                                "is_buyer_maker": df_raw[
                                    "is_buyer_maker"
                                ].values.astype(bool),
                            },
                            index=ts,
                        )
                    else:
                        day_df = fetch_aggtrades(
                            symbol=symbol, date=d, cache_dir=cache_dir
                        )
                except Exception as exc:
                    logger.exception("Failed %s %s: %s", symbol, day_str, exc)
                    failed += 1
                    continue

                if day_df.empty:
                    skipped += 1
                    continue

                stats = _aggregate_one_day(day_df)
                day_rows.append(stats)
                day_dates.append(d.normalize())
                del day_df  # explicit free

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t_start
                    logger.info(
                        "  %s: %d/%d done in %.0fs, %d skipped, %d failed",
                        symbol, i + 1, len(dates), elapsed, skipped, failed,
                    )

            logger.info(
                "%s: %d days aggregated, %d skipped, %d failed (%.0fs)",
                symbol, len(day_rows), skipped, failed, time.time() - t_start,
            )
            if not day_rows:
                logger.warning("No data for %s — skipping", coin)
                continue

            features = _finalize_features(day_rows, day_dates)

        else:
            # Classic approach: load all trades → concat → build features
            fetcher = fetch_aggtrades_vision if args.use_vision else fetch_aggtrades
            all_trades = []
            skipped = 0
            for d in dates:
                try:
                    day_df = fetcher(symbol=symbol, date=d, cache_dir=cache_dir)
                    if day_df.empty:
                        skipped += 1
                        continue
                    all_trades.append(day_df)
                except Exception:
                    logger.exception("Failed %s %s", symbol, d.strftime("%Y-%m-%d"))
                    skipped += 1
            logger.info(
                "%s: fetched %d days, skipped %d",
                symbol, len(all_trades), skipped,
            )
            if not all_trades:
                logger.warning("No data for %s — skipping", coin)
                continue
            trades = pd.concat(all_trades).sort_index()
            features = build_daily_microstructure_features(
                trades, as_of=trades.index.max()
            )

        out_file = out_dir / f"{coin}.parquet"
        features.to_parquet(out_file)
        logger.info(
            "Wrote %s (%d rows, cols=%s)",
            out_file, len(features), list(features.columns),
        )


if __name__ == "__main__":
    main()
