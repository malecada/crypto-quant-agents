from __future__ import annotations

import pandas as pd
import pytest


def test_fetch_funding_rate_uses_cache(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.derivatives import fetch_funding_rate

    cache_file = tmp_path / "BTCUSDT_funding.parquet"
    df_cached = pd.DataFrame(
        {"funding_rate": [0.0001]},
        index=pd.date_range("2026-01-01", periods=1, freq="8h", tz="UTC"),
    )
    df_cached.to_parquet(cache_file)

    def _fail(*args, **kwargs):
        raise AssertionError("network must not be hit when cache present")

    monkeypatch.setattr(
        "tradingagents.strategies.v3.features.derivatives._fetch_funding_page",
        _fail,
    )
    df = fetch_funding_rate(symbol="BTCUSDT", cache_dir=tmp_path)
    assert len(df) == 1
    assert df["funding_rate"].iloc[0] == 0.0001


def test_fetch_funding_rate_paginates(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features import derivatives

    pages = [
        [
            {"fundingTime": 1735689600000, "fundingRate": "0.0001"},  # 2025-01-01 UTC
            {"fundingTime": 1735718400000, "fundingRate": "0.0002"},
        ],
        [
            {"fundingTime": 1735747200000, "fundingRate": "0.0003"},
        ],
        [],  # empty signals end
    ]

    call_count = {"n": 0}

    def _fake(symbol, start_ms, limit):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(pages):
            return pages[idx]
        return []

    monkeypatch.setattr(derivatives, "_fetch_funding_page", _fake)
    df = derivatives.fetch_funding_rate(
        symbol="BTCUSDT",
        cache_dir=tmp_path,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-02", tz="UTC"),
    )
    assert len(df) == 3
    assert df["funding_rate"].iloc[0] == 0.0001


def test_fetch_funding_rate_backoff_on_429(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features import derivatives
    from tradingagents.strategies.v3.features._http import RateLimitError

    calls = {"n": 0}

    def _fake(symbol, start_ms, limit):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError("429")
        return [{"fundingTime": 1735689600000, "fundingRate": "0.0001"}]

    # Avoid actual sleeping
    monkeypatch.setattr(
        "tradingagents.strategies.v3.features._http.time.sleep", lambda _s: None
    )
    monkeypatch.setattr(derivatives, "_fetch_funding_page", _fake)
    df = derivatives.fetch_funding_rate(
        symbol="BTCUSDT",
        cache_dir=tmp_path,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-02", tz="UTC"),
    )
    assert len(df) >= 1
    assert calls["n"] >= 3


def test_fetch_open_interest_uses_cache(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.derivatives import fetch_open_interest_history

    cache_file = tmp_path / "BTCUSDT_oi.parquet"
    df_cached = pd.DataFrame(
        {"open_interest": [1000.0]},
        index=pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC"),
    )
    df_cached.to_parquet(cache_file)

    def _fail(*args, **kwargs):
        raise AssertionError("network must not be hit when cache present")

    monkeypatch.setattr(
        "tradingagents.strategies.v3.features.derivatives._fetch_oi_page",
        _fail,
    )
    df = fetch_open_interest_history(symbol="BTCUSDT", cache_dir=tmp_path)
    assert len(df) == 1
    assert df["open_interest"].iloc[0] == 1000.0


def test_fetch_open_interest_paginates(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features import derivatives

    pages = [
        [
            {"timestamp": 1735689600000, "sumOpenInterest": "1000.0", "sumOpenInterestValue": "30000000.0"},
            {"timestamp": 1735776000000, "sumOpenInterest": "1100.0", "sumOpenInterestValue": "33000000.0"},
        ],
        [
            {"timestamp": 1735862400000, "sumOpenInterest": "1200.0", "sumOpenInterestValue": "36000000.0"},
        ],
        [],
    ]
    call_count = {"n": 0}

    def _fake(symbol, period, start_ms, limit):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(pages):
            return pages[idx]
        return []

    monkeypatch.setattr(derivatives, "_fetch_oi_page", _fake)
    df = derivatives.fetch_open_interest_history(
        symbol="BTCUSDT",
        cache_dir=tmp_path,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-04", tz="UTC"),
    )
    assert len(df) == 3
    assert df["open_interest"].iloc[0] == 1000.0
    assert df["open_interest"].iloc[2] == 1200.0


def test_fetch_premium_index(monkeypatch):
    from tradingagents.strategies.v3.features import derivatives

    def _fake(symbol):
        return {
            "symbol": "BTCUSDT",
            "markPrice": "50000.0",
            "indexPrice": "49950.0",
            "lastFundingRate": "0.0001",
            "time": 1735689600000,
        }

    monkeypatch.setattr(derivatives, "_fetch_premium_index_raw", _fake)
    out = derivatives.fetch_premium_index(symbol="BTCUSDT")
    assert out["mark_price"] == 50000.0
    assert out["index_price"] == 49950.0
    assert out["basis"] == pytest.approx(50.0 / 49950.0, rel=1e-6)


def test_fetch_liquidations_missing_api_key(tmp_path, monkeypatch, caplog):
    import logging
    from tradingagents.strategies.v3.features.derivatives import fetch_liquidations

    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)
    with caplog.at_level(logging.WARNING):
        df = fetch_liquidations(symbol="BTCUSDT", cache_dir=tmp_path)
    assert "COINGLASS_API_KEY" in caplog.text
    assert df.attrs.get("proxy") is True
    assert "liq_asym_24h" in df.columns
    assert (df["liq_asym_24h"] == 0.0).all()


def test_fetch_liquidations_with_api_key(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features import derivatives

    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")

    pages = [
        [
            {
                "t": 1735689600000,
                "longLiquidationUsd": "1000000",
                "shortLiquidationUsd": "500000",
            },
            {
                "t": 1735776000000,
                "longLiquidationUsd": "2000000",
                "shortLiquidationUsd": "3000000",
            },
        ],
        [],
    ]
    call_count = {"n": 0}

    def _fake(symbol, start_ms, end_ms, api_key):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(pages):
            return pages[idx]
        return []

    monkeypatch.setattr(derivatives, "_fetch_liquidations_page", _fake)
    df = derivatives.fetch_liquidations(
        symbol="BTCUSDT",
        cache_dir=tmp_path,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-03", tz="UTC"),
    )
    assert df.attrs.get("proxy") is False
    assert len(df) == 2
    # Day 0: long=1M, short=0.5M → asym = (1-0.5)/(1+0.5) = 0.333
    # Day 1: long=2M, short=3M → asym = (2-3)/(2+3) = -0.2
    assert df["liq_asym_24h"].iloc[0] == pytest.approx(1.0 / 3.0, rel=1e-3)
    assert df["liq_asym_24h"].iloc[1] == pytest.approx(-0.2, rel=1e-3)


def test_fetch_liquidations_uses_cache(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.derivatives import fetch_liquidations

    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")

    cache_file = tmp_path / "BTCUSDT_liquidations.parquet"
    df_cached = pd.DataFrame(
        {"liq_asym_24h": [0.5]},
        index=pd.date_range("2026-01-01", periods=1, freq="D", tz="UTC"),
    )
    df_cached.attrs["proxy"] = False
    df_cached.to_parquet(cache_file)

    def _fail(*args, **kwargs):
        raise AssertionError("network must not be hit when cache present")

    monkeypatch.setattr(
        "tradingagents.strategies.v3.features.derivatives._fetch_liquidations_page",
        _fail,
    )
    df = fetch_liquidations(symbol="BTCUSDT", cache_dir=tmp_path)
    assert len(df) == 1
    assert df["liq_asym_24h"].iloc[0] == 0.5


def test_build_daily_derivatives_features_columns():
    from tradingagents.strategies.v3.features.derivatives import (
        build_daily_derivatives_features,
    )

    funding_idx = pd.date_range("2025-01-01", periods=200, freq="8h", tz="UTC")
    funding_df = pd.DataFrame(
        {"funding_rate": [0.0001 + 0.00001 * (i % 10) for i in range(200)]},
        index=funding_idx,
    )

    oi_idx = pd.date_range("2025-01-01", periods=70, freq="D", tz="UTC")
    oi_df = pd.DataFrame(
        {
            "open_interest": [1000.0 + 10.0 * i for i in range(70)],
            "open_interest_value": [30000.0 + 300.0 * i for i in range(70)],
        },
        index=oi_idx,
    )

    liq_df = pd.DataFrame(
        {"liq_asym_24h": [0.1 * (i % 5) for i in range(70)]},
        index=oi_idx,
    )

    df = build_daily_derivatives_features(
        funding_df=funding_df,
        oi_df=oi_df,
        liq_df=liq_df,
        spot_price_series=pd.Series(
            [50000.0 + 100.0 * i for i in range(70)], index=oi_idx
        ),
        perp_price_series=pd.Series(
            [50050.0 + 100.0 * i for i in range(70)], index=oi_idx
        ),
        as_of=oi_idx.max(),
    )
    expected_cols = {
        "funding_8h_level",
        "funding_z_30",
        "funding_slope_7",
        "basis_annual",
        "oi_change_1d",
        "oi_change_7d",
        "liq_asym_24h",
    }
    assert expected_cols.issubset(df.columns)
    assert df.index.max() <= oi_idx.max()


def test_build_daily_derivatives_look_ahead_guard():
    from tradingagents.strategies.v3.features.derivatives import (
        build_daily_derivatives_features,
    )

    idx = pd.date_range("2025-01-01", periods=50, freq="D", tz="UTC")
    funding_idx = pd.date_range("2025-01-01", periods=150, freq="8h", tz="UTC")
    df = build_daily_derivatives_features(
        funding_df=pd.DataFrame({"funding_rate": [0.0001] * 150}, index=funding_idx),
        oi_df=pd.DataFrame(
            {
                "open_interest": [1000.0] * 50,
                "open_interest_value": [30000.0] * 50,
            },
            index=idx,
        ),
        liq_df=pd.DataFrame({"liq_asym_24h": [0.0] * 50}, index=idx),
        spot_price_series=pd.Series([50000.0] * 50, index=idx),
        perp_price_series=pd.Series([50050.0] * 50, index=idx),
        as_of=pd.Timestamp("2025-01-25", tz="UTC"),
    )
    assert df.index.max() <= pd.Timestamp("2025-01-25", tz="UTC")
