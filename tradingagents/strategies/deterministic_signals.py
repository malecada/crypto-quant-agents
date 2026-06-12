"""Deterministic signal pack — populates ``QuantSignal.deterministic_signals``.

Signals (per the plan's Tier B3):
  funding_z       cross-sectional Z-score of latest funding rate
  usdt_netflow    CoinMetrics FlowInExUSD - FlowOutExUSD (BTC/ETH only)
  ndf             whale concentration — NOT in CM community tier; None
  unlock_flag     bool: insider unlock within next 30 days
  kimchi          deferred (needs CryptoQuant Advanced) — None

The pack carries explicit ``None`` for signals that are unavailable for
the given coin so the modulator prompt can degrade gracefully instead
of guessing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.dataflows.unlocks import next_unlock

logger = logging.getLogger(__name__)

# Cross-sectional reference universe for funding-rate Z-scores
_FUNDING_REF_SYMBOLS = {
    "bitcoin": "BTCUSDT",
    "ethereum": "ETHUSDT",
    "binancecoin": "BNBUSDT",
    "solana": "SOLUSDT",
}

# CoinMetrics asset codes
_CM_ASSET = {
    "bitcoin": "btc",
    "ethereum": "eth",
}


def _funding_z(coin: str, date: str) -> Optional[float]:
    """Cross-sectional Z-score of the target coin's latest funding rate.

    We pull the latest single funding observation for BTC / ETH / BNB / SOL
    (Binance perpetuals) and compute Z over that 4-coin set. Cheap and
    PIT-safe via the existing ``_scrape_funding_rates``.
    """
    try:
        from tradingagents.dataflows.onchain import _scrape_funding_rates

        target_dt = pd.to_datetime(date).date()
        past = target_dt - timedelta(days=3)
        rates = {}
        for asset, sym in _FUNDING_REF_SYMBOLS.items():
            df = _scrape_funding_rates(past, target_dt, symbol=sym)
            if df.empty:
                continue
            rates[asset] = float(df["funding_rate"].iloc[-1])
        if coin not in rates or len(rates) < 2:
            return None
        vals = np.array(list(rates.values()))
        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        if sd == 0.0:
            return 0.0
        return (rates[coin] - m) / sd
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"funding_z failed for {coin} @ {date}: {exc}")
        return None


def _usdt_netflow(coin: str, date: str) -> Optional[float]:
    """Net USD flow into exchanges from the on-chain bitemporal store.

    CoinMetrics community has FlowInExUSD / FlowOutExUSD for BTC/ETH only.
    Returns ``netflow = inflow - outflow`` for the 7 days ending at ``date``,
    PIT-filtered. None for unsupported coins.
    """
    asset = _CM_ASSET.get(coin)
    if asset is None:
        return None
    try:
        from tradingagents.dataflows.onchain_store import query_metrics

        as_of = pd.to_datetime(date).tz_localize("UTC")
        ts_start = as_of - pd.Timedelta(days=7)
        rows = query_metrics(
            coin=asset,
            ts_start=ts_start,
            ts_end=as_of,
            as_of=as_of,
            metrics=["FlowInExUSD", "FlowOutExUSD"],
        )
        if rows.empty:
            return None
        wide = (
            rows.groupby(["event_ts", "metric"])["value"].last().unstack("metric")
        )
        if "FlowInExUSD" not in wide.columns or "FlowOutExUSD" not in wide.columns:
            return None
        netflow = float(
            (wide["FlowInExUSD"] - wide["FlowOutExUSD"]).fillna(0.0).sum()
        )
        return netflow
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"usdt_netflow failed for {coin} @ {date}: {exc}")
        return None


def _unlock_flag(coin: str, date: str) -> bool:
    """True iff insider unlock (team/vc/treasury) within next 30 days."""
    try:
        as_of = pd.to_datetime(date).tz_localize("UTC")
        nxt = next_unlock(coin, as_of, max_days=30, insider_only=True)
        return nxt is not None
    except Exception as exc:  # noqa: BLE001
        logger.debug(f"unlock_flag failed for {coin} @ {date}: {exc}")
        return False


def compute_deterministic_pack(coin: str, date: str) -> dict:
    """Return the deterministic signal pack consumed by Layer 1 + Layer 2."""
    return {
        "funding_z": _funding_z(coin, date),
        "usdt_netflow": _usdt_netflow(coin, date),
        "ndf": None,  # not in CM community tier
        "unlock_flag": _unlock_flag(coin, date),
        "kimchi": None,  # deferred (CryptoQuant Advanced $29/mo)
    }
