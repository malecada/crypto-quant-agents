from __future__ import annotations

import pandas as pd

from tradingagents.strategies.v3.features.microstructure import (
    compute_vpin,
    volume_buckets,
)


def test_volume_buckets_split_correctly():
    trades = pd.DataFrame(
        {
            "price": [100.0] * 10,
            "qty": [1.0] * 10,
            "is_buyer_maker": [True, False] * 5,
        },
        index=pd.date_range("2026-01-01", periods=10, freq="min", tz="UTC"),
    )
    buckets = list(volume_buckets(trades, bucket_size=2.5))
    assert len(buckets) == 4
    for b in buckets:
        assert abs(b["qty"].sum() - 2.5) < 1e-6


def test_vpin_zero_when_balanced():
    trades = pd.DataFrame(
        {
            "price": [100.0] * 100,
            "qty": [1.0] * 100,
            "is_buyer_maker": [True, False] * 50,
        },
        index=pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC"),
    )
    vpin = compute_vpin(trades, n_buckets=10)
    assert vpin < 0.05


def test_vpin_high_when_imbalanced():
    trades = pd.DataFrame(
        {
            "price": [100.0] * 100,
            "qty": [1.0] * 100,
            "is_buyer_maker": [True] * 100,  # all sells
        },
        index=pd.date_range("2026-01-01", periods=100, freq="min", tz="UTC"),
    )
    vpin = compute_vpin(trades, n_buckets=10)
    assert vpin > 0.9


import numpy as np


def test_build_daily_features_basic():
    from tradingagents.strategies.v3.features.microstructure import (
        build_daily_microstructure_features,
    )

    trades = pd.DataFrame(
        {
            "price": np.random.uniform(99, 101, 5000),
            "qty": np.random.uniform(0.1, 1.0, 5000),
            "is_buyer_maker": np.random.choice([True, False], 5000),
        },
        index=pd.date_range("2026-01-01", periods=5000, freq="min", tz="UTC"),
    )
    df = build_daily_microstructure_features(
        trades, as_of=pd.Timestamp("2026-01-04", tz="UTC")
    )
    assert "vpin_50" in df.columns
    assert "ofi_d" in df.columns
    assert "ofi_d_w" in df.columns
    assert "aggressor_ratio" in df.columns
    assert df.index.max() <= pd.Timestamp("2026-01-04", tz="UTC")


def test_build_daily_features_look_ahead_guard():
    from tradingagents.strategies.v3.features.microstructure import (
        build_daily_microstructure_features,
    )

    trades = pd.DataFrame(
        {
            "price": np.random.uniform(99, 101, 100),
            "qty": np.random.uniform(0.1, 1.0, 100),
            "is_buyer_maker": np.random.choice([True, False], 100),
        },
        index=pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
    )
    df = build_daily_microstructure_features(
        trades, as_of=pd.Timestamp("2026-01-02 12:00", tz="UTC")
    )
    assert df.index.max() <= pd.Timestamp("2026-01-02 12:00", tz="UTC")


def test_fetch_aggtrades_uses_cache(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.microstructure import fetch_aggtrades

    cache_file = tmp_path / "BTCUSDT_2026-01-01.parquet"
    df_cached = pd.DataFrame(
        {
            "price": [100.0],
            "qty": [1.0],
            "is_buyer_maker": [True],
        },
        index=pd.date_range("2026-01-01", periods=1, freq="min", tz="UTC"),
    )
    df_cached.to_parquet(cache_file)

    def _fail_call(*args, **kwargs):
        raise AssertionError("network must not be hit when cache present")

    monkeypatch.setattr(
        "tradingagents.strategies.v3.features.microstructure._fetch_one_day",
        _fail_call,
    )

    out = fetch_aggtrades(
        symbol="BTCUSDT",
        date=pd.Timestamp("2026-01-01", tz="UTC"),
        cache_dir=tmp_path,
    )
    assert len(out) == 1
    assert out["price"].iloc[0] == 100.0


def test_fetch_aggtrades_backoff_on_429(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features import microstructure

    calls = {"n": 0}

    def _fake(symbol, start_ms, end_ms):
        calls["n"] += 1
        if calls["n"] < 3:
            raise microstructure.RateLimitError("429")
        return pd.DataFrame(
            {
                "price": [100.0],
                "qty": [1.0],
                "is_buyer_maker": [True],
            },
            index=pd.date_range("2026-01-01", periods=1, freq="min", tz="UTC"),
        )

    monkeypatch.setattr(microstructure, "_fetch_one_day", _fake)
    monkeypatch.setattr(microstructure.time, "sleep", lambda _s: None)

    out = microstructure.fetch_aggtrades(
        symbol="BTCUSDT",
        date=pd.Timestamp("2026-01-01", tz="UTC"),
        cache_dir=tmp_path,
        max_retries=5,
    )
    assert len(out) == 1
    assert calls["n"] == 3


def test_proxy_features_from_klines(synthetic_ohlcv):
    from tradingagents.strategies.v3.features.microstructure import (
        build_proxy_microstructure_features,
    )

    df = build_proxy_microstructure_features(
        synthetic_ohlcv, as_of=synthetic_ohlcv.index.max()
    )
    expected_cols = {"ofi_proxy", "ofi_proxy_w", "vol_dispersion"}
    assert expected_cols.issubset(df.columns)
    assert df.index.max() <= synthetic_ohlcv.index.max()
    assert len(df) > 0


def test_fetch_aggtrades_vision_uses_cache(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.microstructure import fetch_aggtrades_vision

    cache_file = tmp_path / "BTCUSDT_2026-01-01.parquet"
    df_cached = pd.DataFrame(
        {"price": [100.0], "qty": [1.0], "is_buyer_maker": [True]},
        index=pd.date_range("2026-01-01", periods=1, freq="min", tz="UTC"),
    )
    df_cached.to_parquet(cache_file)

    def _fail(*args, **kwargs):
        raise AssertionError("network must not be hit when cache present")
    monkeypatch.setattr("requests.get", _fail)

    df = fetch_aggtrades_vision(
        symbol="BTCUSDT",
        date=pd.Timestamp("2026-01-01", tz="UTC"),
        cache_dir=tmp_path,
    )
    assert len(df) == 1
    assert df["price"].iloc[0] == 100.0


def test_fetch_aggtrades_vision_404_returns_empty(tmp_path, monkeypatch):
    from tradingagents.strategies.v3.features.microstructure import fetch_aggtrades_vision

    class FakeResp:
        status_code = 404
        content = b""
        def raise_for_status(self):
            raise RuntimeError("should not be called for 404")
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp())

    df = fetch_aggtrades_vision(
        symbol="BTCUSDT",
        date=pd.Timestamp("2099-01-01", tz="UTC"),
        cache_dir=tmp_path,
    )
    assert df.empty
    assert list(df.columns) == ["price", "qty", "is_buyer_maker"]
