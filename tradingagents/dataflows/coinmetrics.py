"""CoinMetrics Community API client.

Free-tier network data for BTC + ETH (and a handful of other majors).
Rate limit: 10 req / 6s sliding window per IP (no key required).

Supported community metrics (verified 2026-04):
  AdrActCnt, TxCnt, HashRate, CapMVRVCur, CapMrktCurUSD, FeeTotNtv,
  FlowInExUSD, FlowOutExUSD, IssTotUSD, SplyCur, PriceUSD

Not in community tier: CapRealUSD, NVTAdj, SOPR, TxTfrValAdjUSD.

Flow* metrics carry a `-status: flash` flag indicating up-to-~3mo revision
window. Callers needing PIT correctness must apply a wider as_of_ts lag
for flow metrics (see :data:`FLASH_METRICS`).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Iterable, Iterator

import pandas as pd
import requests

log = logging.getLogger(__name__)

BASE_URL = "https://community-api.coinmetrics.io/v4"
DEFAULT_TIMEOUT = 30

# Per CM docs: 10 requests per 6s sliding window. Stay well under.
_MIN_REQUEST_INTERVAL = 0.7  # ~8 req / 6s

FLASH_METRICS = frozenset({"FlowInExUSD", "FlowOutExUSD"})

# Supported community metrics per (asset, metric) confirmed by probe + catalog
# (https://community-api.coinmetrics.io/v4/catalog/metrics, 1d frequency).
# Both btc and eth expose the same 28 free metrics.
_COMMON_COMMUNITY_METRICS = frozenset({
    "AdrActCnt", "AdrBalCnt", "BlkCnt",
    "CapMVRVCur", "CapMrktCurUSD", "CapMrktEstUSD",
    "FeeTotNtv",
    "FlowInExNtv", "FlowInExUSD", "FlowOutExNtv", "FlowOutExUSD",
    "HashRate",
    "IssTotNtv", "IssTotUSD",
    "PriceBTC", "PriceUSD",
    "ROI1yr", "ROI30d",
    "SplyCur",
    "SplyExNtv", "SplyExUSD", "SplyExpFut10yr",
    "TxCnt", "TxTfrCnt",
    "volume_reported_spot_usd_1d",
})

SUPPORTED = {
    "btc": _COMMON_COMMUNITY_METRICS,
    "eth": _COMMON_COMMUNITY_METRICS,
}

# Stablecoin supply tracking via CM Community: SplyCur per chain-token reveals
# mint/burn dynamics that historically would need direct Web3 log scraping.
_STABLE_SUPPLY_METRICS = frozenset({"SplyCur", "PriceUSD"})
STABLECOIN_SUPPORTED = {
    "usdt": _STABLE_SUPPLY_METRICS,
    "usdc": _STABLE_SUPPLY_METRICS,
    "dai":  _STABLE_SUPPLY_METRICS,
    "usdt_eth": _STABLE_SUPPLY_METRICS,
    "usdc_eth": _STABLE_SUPPLY_METRICS,
    "usdt_trx": _STABLE_SUPPLY_METRICS,
}
SUPPORTED.update(STABLECOIN_SUPPORTED)


class CoinMetricsError(RuntimeError):
    pass


def _throttled_get(url: str, params: dict, last_call: list[float]) -> requests.Response:
    elapsed = time.monotonic() - last_call[0]
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    backoff = 2.0
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        last_call[0] = time.monotonic()
        if resp.status_code == 429:
            log.warning("CM 429 — sleeping %.1fs", backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
            continue
        if resp.status_code >= 500:
            log.warning("CM %d — retry %d", resp.status_code, attempt + 1)
            time.sleep(backoff)
            continue
        return resp
    raise CoinMetricsError(f"CM request failed after 5 attempts: {url}")


def iter_asset_metrics(
    asset: str,
    metrics: Iterable[str],
    start: datetime,
    end: datetime,
    frequency: str = "1d",
    page_size: int = 10000,
) -> Iterator[dict]:
    """Yield raw asset-metric rows from CM community API, paginated."""
    asset = asset.lower()
    supported = SUPPORTED.get(asset)
    if supported is None:
        raise CoinMetricsError(
            f"Asset {asset!r} not verified for community tier. "
            f"Add to SUPPORTED after probing."
        )
    req_metrics = [m for m in metrics if m in supported]
    unknown = [m for m in metrics if m not in supported]
    if unknown:
        log.warning("%s: skipping metrics not in community tier: %s",
                    asset, unknown)
    if not req_metrics:
        return
    params = {
        "assets": asset,
        "metrics": ",".join(req_metrics),
        "frequency": frequency,
        "start_time": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "end_time": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "page_size": page_size,
    }
    url = f"{BASE_URL}/timeseries/asset-metrics"
    last_call = [0.0]
    while True:
        resp = _throttled_get(url, params, last_call)
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise CoinMetricsError(f"CM HTTP error: {e} body={resp.text[:400]}") from e
        payload = resp.json()
        for row in payload.get("data", []) or []:
            yield row
        nxt = payload.get("next_page_url")
        if not nxt:
            return
        # next_page_url is absolute; swap params for None so requests doesn't
        # re-append them.
        url = nxt
        params = None  # type: ignore[assignment]


def normalize_rows(
    rows: Iterable[dict],
    asset: str,
    metrics: Iterable[str],
    ingest_lag_days_stable: int = 1,
    ingest_lag_days_flash: int = 7,
) -> pd.DataFrame:
    """Convert raw CM rows into long-format bitemporal frame.

    Columns: event_ts, as_of_ts, coin, metric, value, source, status.

    as_of_ts = event_ts + lag. Flash metrics (flow) get a wider lag to
    respect the ~3-month revision window conservatively.
    """
    records: list[dict] = []
    metric_list = list(metrics)
    for row in rows:
        event_ts = pd.to_datetime(row["time"], utc=True).to_pydatetime()
        for m in metric_list:
            if m not in row:
                continue
            raw_val = row[m]
            if raw_val is None or raw_val == "":
                # CoinMetrics returns null for metrics not yet published or
                # not applicable on a given date — skip silently.
                continue
            try:
                value = float(raw_val)
            except (TypeError, ValueError):
                continue
            status = row.get(f"{m}-status", "final")
            lag_days = (
                ingest_lag_days_flash
                if (m in FLASH_METRICS or status == "flash")
                else ingest_lag_days_stable
            )
            as_of_ts = event_ts + pd.Timedelta(days=lag_days)
            records.append({
                "event_ts": event_ts,
                "as_of_ts": as_of_ts.to_pydatetime() if isinstance(as_of_ts, pd.Timestamp) else as_of_ts,
                "coin": asset.lower(),
                "metric": m,
                "value": value,
                "source": "coinmetrics_community",
                "status": status,
            })
    return pd.DataFrame.from_records(
        records,
        columns=["event_ts", "as_of_ts", "coin", "metric", "value", "source", "status"],
    )


def fetch_asset_metrics_df(
    asset: str,
    metrics: Iterable[str],
    start: datetime,
    end: datetime,
    frequency: str = "1d",
) -> pd.DataFrame:
    """Convenience: fetch + normalize in one call."""
    rows = list(iter_asset_metrics(asset, metrics, start, end, frequency))
    return normalize_rows(rows, asset, metrics)
