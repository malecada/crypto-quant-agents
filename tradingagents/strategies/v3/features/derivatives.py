"""Derivatives features: funding rate, basis, OI, liquidation asymmetry.

Look-ahead-safe by construction: ``build_daily_derivatives_features`` slices
input to ``df.index <= as_of`` before any rolling op. Tests assert this.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from tradingagents.strategies.v3.features._http import RateLimitError, with_backoff

logger = logging.getLogger(__name__)

_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def _fetch_funding_page(symbol: str, start_ms: int, limit: int = 1000) -> list[dict]:
    """One funding-rate page from Binance Futures."""
    resp = requests.get(
        _BINANCE_FUNDING_URL,
        params={"symbol": symbol, "startTime": start_ms, "limit": limit},
        timeout=10,
    )
    if resp.status_code == 429:
        raise RateLimitError("Binance Futures 429")
    resp.raise_for_status()
    return resp.json()


def fetch_funding_rate(
    symbol: str,
    cache_dir: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch full funding-rate history, paginated, cached to parquet.

    Returns DataFrame indexed by funding-time (UTC) with column ``funding_rate``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_funding.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    if start is None:
        start = pd.Timestamp("2020-01-01", tz="UTC")
    if end is None:
        end = pd.Timestamp.utcnow().tz_convert("UTC")

    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    all_rows: list[dict] = []

    while cursor_ms < end_ms:
        page = with_backoff(
            lambda: _fetch_funding_page(symbol, cursor_ms, limit)
        )
        if not page:
            break
        all_rows.extend(page)
        last_time = page[-1]["fundingTime"]
        if last_time <= cursor_ms:
            break
        cursor_ms = last_time + 1

    if not all_rows:
        df = pd.DataFrame(columns=["funding_rate"])
    else:
        df = pd.DataFrame(
            {
                "funding_rate": [float(r["fundingRate"]) for r in all_rows],
            },
            index=pd.to_datetime(
                [r["fundingTime"] for r in all_rows], unit="ms", utc=True
            ),
        )
        df.index.name = "ts"
        df = df.sort_index()
    df.to_parquet(cache_file)
    return df


_BINANCE_OI_HIST_URL = "https://fapi.binance.com/fapi/v1/openInterestHist"
_BINANCE_PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


def _fetch_oi_page(symbol: str, period: str, start_ms: int, limit: int = 500) -> list[dict]:
    """One open-interest history page. ``period`` is one of 5m, 15m, 30m, 1h, 2h, 4h, 6h, 12h, 1d."""
    resp = requests.get(
        _BINANCE_OI_HIST_URL,
        params={
            "symbol": symbol,
            "period": period,
            "startTime": start_ms,
            "limit": limit,
        },
        timeout=10,
    )
    if resp.status_code == 429:
        raise RateLimitError("Binance Futures 429 (OI)")
    resp.raise_for_status()
    return resp.json()


def fetch_open_interest_history(
    symbol: str,
    cache_dir: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    period: str = "1d",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch full OI history, paginated, cached to ``{symbol}_oi.parquet``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_oi.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    if start is None:
        start = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=730)
    if end is None:
        end = pd.Timestamp.utcnow().tz_convert("UTC")

    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict] = []

    while cursor_ms < end_ms:
        page = with_backoff(
            lambda: _fetch_oi_page(symbol, period, cursor_ms, limit)
        )
        if not page:
            break
        rows.extend(page)
        last_time = page[-1]["timestamp"]
        if last_time <= cursor_ms:
            break
        cursor_ms = last_time + 1

    if not rows:
        df = pd.DataFrame(columns=["open_interest", "open_interest_value"])
    else:
        df = pd.DataFrame(
            {
                "open_interest": [float(r["sumOpenInterest"]) for r in rows],
                "open_interest_value": [float(r["sumOpenInterestValue"]) for r in rows],
            },
            index=pd.to_datetime(
                [r["timestamp"] for r in rows], unit="ms", utc=True
            ),
        )
        df.index.name = "ts"
        df = df.sort_index()
    df.to_parquet(cache_file)
    return df


def _fetch_premium_index_raw(symbol: str) -> dict:
    resp = requests.get(_BINANCE_PREMIUM_URL, params={"symbol": symbol}, timeout=10)
    if resp.status_code == 429:
        raise RateLimitError("Binance Futures 429 (premium)")
    resp.raise_for_status()
    return resp.json()


def fetch_premium_index(symbol: str) -> dict:
    """Snapshot mark price + index price + last funding rate for ``symbol``.

    Returns dict with keys: ``mark_price``, ``index_price``, ``basis``,
    ``last_funding_rate``, ``timestamp`` (pd.Timestamp UTC). ``basis`` is
    ``(mark - index) / index``.
    """
    raw = with_backoff(lambda: _fetch_premium_index_raw(symbol))
    mark = float(raw["markPrice"])
    index = float(raw["indexPrice"])
    return {
        "mark_price": mark,
        "index_price": index,
        "basis": (mark - index) / index if index > 0 else 0.0,
        "last_funding_rate": float(raw["lastFundingRate"]),
        "timestamp": pd.Timestamp(raw["time"], unit="ms", tz="UTC"),
    }


import os

_COINGLASS_LIQ_URL = "https://open-api-v3.coinglass.com/api/futures/liquidation/v3/aggregated-history"


def _fetch_liquidations_page(
    symbol: str, start_ms: int, end_ms: int, api_key: str
) -> list[dict]:
    """One page from Coinglass aggregated liquidation history.

    Free tier returns daily aggregates; ``interval=1d``.
    """
    resp = requests.get(
        _COINGLASS_LIQ_URL,
        params={
            "symbol": symbol,
            "interval": "1d",
            "startTime": start_ms,
            "endTime": end_ms,
        },
        headers={"coinglassSecret": api_key},
        timeout=10,
    )
    if resp.status_code == 429:
        raise RateLimitError("Coinglass 429 (liquidations)")
    resp.raise_for_status()
    payload = resp.json()
    return payload.get("data", []) if isinstance(payload, dict) else []


def fetch_liquidations(
    symbol: str,
    cache_dir: Path,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Daily long/short liquidation asymmetry for ``symbol``.

    Returns DataFrame indexed by date (UTC) with column ``liq_asym_24h`` =
    ``(long - short) / (long + short)``. ``df.attrs["proxy"]`` is False on real
    fetch, True when ``COINGLASS_API_KEY`` is unset (zero-filled fallback).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{symbol}_liquidations.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    api_key = os.environ.get("COINGLASS_API_KEY")
    if not api_key:
        logger.warning(
            "COINGLASS_API_KEY not set — liq_asym_24h will be zeroed (proxy=True)"
        )
        if start is None:
            start = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=730)
        if end is None:
            end = pd.Timestamp.utcnow().tz_convert("UTC")
        idx = pd.date_range(start.normalize(), end.normalize(), freq="D", tz="UTC")
        df = pd.DataFrame({"liq_asym_24h": [0.0] * len(idx)}, index=idx)
        df.attrs["proxy"] = True
        df.to_parquet(cache_file)
        return df

    if start is None:
        start = pd.Timestamp.utcnow().tz_convert("UTC") - pd.Timedelta(days=730)
    if end is None:
        end = pd.Timestamp.utcnow().tz_convert("UTC")

    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    rows: list[dict] = []
    while cursor_ms < end_ms:
        page = with_backoff(
            lambda: _fetch_liquidations_page(symbol, cursor_ms, end_ms, api_key)
        )
        if not page:
            break
        rows.extend(page)
        last_t = page[-1]["t"]
        if last_t <= cursor_ms:
            break
        cursor_ms = last_t + 1

    if not rows:
        df = pd.DataFrame(columns=["liq_asym_24h"])
    else:
        long_vol = [float(r["longLiquidationUsd"]) for r in rows]
        short_vol = [float(r["shortLiquidationUsd"]) for r in rows]
        asym = []
        for lv, sv in zip(long_vol, short_vol):
            total = lv + sv
            asym.append((lv - sv) / total if total > 0 else 0.0)
        df = pd.DataFrame(
            {"liq_asym_24h": asym},
            index=pd.to_datetime([r["t"] for r in rows], unit="ms", utc=True),
        )
        df.index.name = "ts"
        df = df.sort_index()
    df.attrs["proxy"] = False
    df.to_parquet(cache_file)
    return df


def build_daily_derivatives_features(
    funding_df: pd.DataFrame,
    oi_df: pd.DataFrame,
    liq_df: pd.DataFrame,
    spot_price_series: pd.Series,
    perp_price_series: pd.Series,
    as_of: pd.Timestamp,
    z_window: int = 30,
    slope_window: int = 7,
) -> pd.DataFrame:
    """Aggregate raw derivative inputs into daily features.

    Columns produced (per spec §4.2):
      - ``funding_8h_level``  : last 8h funding rate (resampled to daily, last)
      - ``funding_z_30``      : 30-day Z-score of daily funding
      - ``funding_slope_7``   : current funding − 7-day mean
      - ``basis_annual``      : (perp − spot) / spot × 365 (annualized)
      - ``oi_change_1d``      : log change in OI vs previous day
      - ``oi_change_7d``      : 7-day log change in OI
      - ``liq_asym_24h``      : passthrough from ``liq_df``

    Look-ahead guard: every input is sliced to ``df.index <= as_of`` before
    rolling ops.
    """
    import numpy as np

    if not isinstance(as_of, pd.Timestamp):
        raise TypeError("as_of must be a pandas Timestamp")

    funding_df = funding_df[funding_df.index <= as_of].copy()
    oi_df = oi_df[oi_df.index <= as_of].copy()
    liq_df = liq_df[liq_df.index <= as_of].copy()
    spot_price_series = spot_price_series[spot_price_series.index <= as_of].copy()
    perp_price_series = perp_price_series[perp_price_series.index <= as_of].copy()

    # Resample 8h funding to daily (last value of day)
    funding_daily = (
        funding_df["funding_rate"]
        .resample("D")
        .last()
        .rename("funding_8h_level")
    )
    funding_z = (
        (funding_daily - funding_daily.rolling(z_window).mean())
        / funding_daily.rolling(z_window).std()
    ).rename("funding_z_30")
    funding_slope = (
        funding_daily - funding_daily.rolling(slope_window).mean()
    ).rename("funding_slope_7")

    # Basis (annualized perpetual premium)
    basis = ((perp_price_series - spot_price_series) / spot_price_series) * 365.0
    basis = basis.rename("basis_annual")

    # OI changes (log)
    oi = oi_df["open_interest"]
    oi_log = np.log(oi.replace(0.0, np.nan))
    oi_change_1d = oi_log.diff(1).rename("oi_change_1d")
    oi_change_7d = oi_log.diff(7).rename("oi_change_7d")

    # Liquidations passthrough (already daily, indexed by date)
    liq = liq_df["liq_asym_24h"].rename("liq_asym_24h")

    out = pd.concat(
        [funding_daily, funding_z, funding_slope, basis, oi_change_1d, oi_change_7d, liq],
        axis=1,
    )
    out = out.sort_index()
    return out
