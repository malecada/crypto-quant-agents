"""Tests for incremental OHLCV cache in coingecko_binance._load_crypto_ohlcv.

Validates the per-cycle full-reload bug (cache filename embedded curr_date,
so every cycle missed the cache and refetched 7 years of klines) is fixed:
canonical filename + cache_last detection + tail-only fetch.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache_dir config at a tmp dir and reset session cache."""
    from tradingagents.dataflows import coingecko_binance, config as cfg_mod

    coingecko_binance._session_cache.clear()

    def _patched_get_config():
        return {
            "data_cache_dir": str(tmp_path),
            "ohlcv_lookback_years": 7,
        }

    monkeypatch.setattr(coingecko_binance, "get_config", _patched_get_config)
    return tmp_path


def _build_klines(start_iso: str, n_days: int) -> list[list]:
    """Generate `n_days` of synthetic daily klines starting at `start_iso`."""
    start_ms = int(pd.Timestamp(start_iso).timestamp() * 1000)
    klines = []
    for i in range(n_days):
        ts_ms = start_ms + i * 86_400_000
        klines.append([
            ts_ms,
            "100.0", "110.0", "90.0", "105.0",  # OHLC
            "1000.0",  # volume
            ts_ms + 86_399_999, "0", 0, "0", "0", "0",
        ])
    return klines


def test_cold_cache_writes_canonical_filename(isolated_cache):
    """First fetch creates `{coin}-crypto-ohlcv.csv`, NOT a dated filename."""
    from tradingagents.dataflows import coingecko_binance

    klines = _build_klines("2026-05-15", 5)
    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=klines) as mock_fetch:
        df = coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-19")

    assert not df.empty
    assert mock_fetch.call_count == 1
    canonical = isolated_cache / "bitcoin-crypto-ohlcv.csv"
    assert canonical.exists()
    # No legacy dated file should be created.
    dated = [f for f in os.listdir(isolated_cache)
             if f.startswith("bitcoin-crypto-") and f != "bitcoin-crypto-ohlcv.csv"]
    assert dated == []


def test_warm_cache_skips_fetch_when_curr_date_covered(isolated_cache):
    """Same-day re-call hits the cache and makes ZERO network calls."""
    from tradingagents.dataflows import coingecko_binance

    klines = _build_klines("2026-05-15", 5)
    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=klines) as mock_fetch:
        coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-19")

    coingecko_binance.clear_session_cache()  # force disk-cache path

    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=[]) as mock_fetch2:
        df = coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-19")

    assert not df.empty
    assert mock_fetch2.call_count == 0, "warm cache must not hit Binance"


def test_next_day_appends_only_tail_bars(isolated_cache):
    """curr_date advances 1 day → fetch only the new bar, not 7 years."""
    from tradingagents.dataflows import coingecko_binance

    initial = _build_klines("2026-05-15", 5)  # 15→19
    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=initial):
        coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-19")

    coingecko_binance.clear_session_cache()

    tail = _build_klines("2026-05-20", 1)
    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=tail) as mock_fetch:
        df = coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-20")

    # Exactly one chunked fetch, and the range must start ON or AFTER the day
    # following the cached last date — not 7 years back.
    assert mock_fetch.call_count == 1
    from_ms = mock_fetch.call_args[0][1]
    # Production code uses time.mktime (local-TZ); match its convention here
    # to avoid spurious offsets from naive-pandas-UTC vs local-mktime mismatch.
    import time as _t
    expected_min_ms = int(
        _t.mktime(pd.Timestamp("2026-05-20").to_pydatetime().timetuple()) * 1000
    )
    assert from_ms >= expected_min_ms, (
        f"Tail fetch started too early: from_ms={from_ms}, "
        f"expected ≥ {expected_min_ms} (2026-05-20 local)"
    )
    # Window must be small (1-2 day tail), NOT 7 years.
    span_days = (pd.Timestamp("2026-05-20") - pd.Timestamp(from_ms, unit="ms")).days
    assert abs(span_days) <= 2, f"tail window too wide: {span_days} days"
    assert len(df) == 6  # 5 cached + 1 new


def test_legacy_dated_cache_seeds_canonical(isolated_cache):
    """First run after upgrade: existing dated file seeds the canonical cache."""
    from tradingagents.dataflows import coingecko_binance

    legacy = isolated_cache / "bitcoin-crypto-2019-05-20-2026-05-19.csv"
    pd.DataFrame({
        "Date": pd.date_range("2026-05-15", periods=5, freq="D"),
        "Open": [100.0] * 5, "High": [110.0] * 5, "Low": [90.0] * 5,
        "Close": [105.0] * 5, "Volume": [1000.0] * 5,
    }).to_csv(legacy, index=False)

    # Same-day request: should seed canonical from legacy, no fetch needed.
    with patch.object(coingecko_binance, "_binance_klines_chunked",
                      return_value=[]) as mock_fetch:
        df = coingecko_binance._load_crypto_ohlcv("bitcoin", "2026-05-19")

    assert not df.empty
    assert mock_fetch.call_count == 0
    canonical = isolated_cache / "bitcoin-crypto-ohlcv.csv"
    assert canonical.exists()
