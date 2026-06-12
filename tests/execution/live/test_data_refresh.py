from unittest.mock import patch, MagicMock

import pandas as pd
import pytest


@pytest.fixture
def fake_cm_df():
    return pd.DataFrame({
        "coin": ["BTC"], "metric": ["MVRV"], "valid_from": ["2026-05-12"],
        "value": [1.5],
    })


@pytest.fixture
def fake_defillama_df():
    return pd.DataFrame({
        "coin": ["BTC"], "metric": ["TVL"], "valid_from": ["2026-05-12"],
        "value": [50e9],
    })


def test_refresh_coinmetrics_calls_fetch_and_upsert(tmp_path, fake_cm_df):
    from tradingagents.execution.live import data_refresh

    with patch.object(data_refresh, "fetch_coinmetrics_incremental",
                      return_value=fake_cm_df) as mock_fetch, \
         patch.object(data_refresh, "upsert_onchain_rows") as mock_upsert:
        data_refresh.refresh_coinmetrics(coins=["BTC"], store_root=tmp_path)
        mock_fetch.assert_called_once()
        mock_upsert.assert_called_once()
        df_arg, root_arg = mock_upsert.call_args.args
        assert root_arg == tmp_path
        assert "MVRV" in df_arg["metric"].values


def test_refresh_handles_empty_response(tmp_path):
    from tradingagents.execution.live import data_refresh

    empty = pd.DataFrame(columns=["coin", "metric", "valid_from", "value"])
    with patch.object(data_refresh, "fetch_coinmetrics_incremental",
                      return_value=empty), \
         patch.object(data_refresh, "upsert_onchain_rows") as mock_upsert:
        data_refresh.refresh_coinmetrics(coins=["BTC"], store_root=tmp_path)
        mock_upsert.assert_not_called()


def test_refresh_defillama_uses_correct_args(tmp_path, fake_defillama_df):
    from tradingagents.execution.live import data_refresh

    with patch.object(data_refresh, "fetch_defillama_incremental",
                      return_value=fake_defillama_df), \
         patch.object(data_refresh, "upsert_onchain_rows") as mock_upsert:
        data_refresh.refresh_defillama(coins=["BTC", "ETH"], store_root=tmp_path)
        mock_upsert.assert_called_once()


def test_refresh_binance_ohlcv_appends_yesterday(tmp_path):
    from tradingagents.execution.live import data_refresh

    fake_bar = pd.DataFrame({
        "date": ["2026-05-11"], "open": [60000], "high": [61000],
        "low": [59000], "close": [60500], "volume": [1000],
    })
    with patch.object(data_refresh, "fetch_binance_daily",
                      return_value=fake_bar) as mock_f, \
         patch.object(data_refresh, "append_ohlcv") as mock_app:
        data_refresh.refresh_ohlcv(coin="bitcoin", cache_root=tmp_path)
        mock_f.assert_called_once()
        mock_app.assert_called_once()
        # Cold-start: no cache file → fetches the full min_history window.
        kwargs = mock_f.call_args.kwargs
        assert kwargs.get("symbol") == "BTCUSDT"


def test_refresh_binance_ohlcv_incremental_when_cache_warm(tmp_path):
    """When cache has >= min_history rows, refresh fetches only 2 days."""
    from tradingagents.execution.live import data_refresh

    # Seed a warm cache with 100 rows (>= default min_history=60).
    cache_file = tmp_path / "BTCUSDT_1d.parquet"
    seed = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=100, freq="D"),
        "open": [60000] * 100, "high": [61000] * 100,
        "low": [59000] * 100, "close": [60500] * 100, "volume": [1000] * 100,
    })
    seed.to_parquet(cache_file, index=False)

    fake_bar = pd.DataFrame({
        "date": ["2026-05-11"], "open": [60000], "high": [61000],
        "low": [59000], "close": [60500], "volume": [1000],
    })
    with patch.object(data_refresh, "fetch_binance_daily",
                      return_value=fake_bar) as mock_f, \
         patch.object(data_refresh, "append_ohlcv") as mock_app:
        data_refresh.refresh_ohlcv(coin="bitcoin", cache_root=tmp_path)
        mock_f.assert_called_once()
        mock_app.assert_called_once()
        kwargs = mock_f.call_args.kwargs
        assert kwargs.get("days") == 2
        assert kwargs.get("symbol") == "BTCUSDT"


def test_refresh_coinglass_idempotent(monkeypatch, tmp_path):
    """Two calls in one cycle produce identical parquet (no row duplication)."""
    import pandas as pd
    from tradingagents.execution.live import data_refresh

    # Stub the underlying Coinglass fetch helpers — return a small known frame.
    calls = []

    def fake_fetch_oi_agg(symbol, key):
        calls.append(("oi", symbol))
        idx = pd.to_datetime(["2026-05-13", "2026-05-14"], utc=True)
        return pd.DataFrame({
            "oi_open": [1.0, 2.0], "oi_high": [1.0, 2.0],
            "oi_low": [1.0, 2.0], "oi_close": [1.0, 2.0],
        }, index=idx)

    monkeypatch.setattr(
        "scripts.fetch_coinglass_history.fetch_oi_agg", fake_fetch_oi_agg
    )
    # Stub other endpoints similarly (return empty DataFrames so the test stays focused)
    for fn in ("fetch_liq_agg", "fetch_ls_ratio", "fetch_taker_vol", "fetch_funding_weighted"):
        monkeypatch.setattr(f"scripts.fetch_coinglass_history.{fn}",
                             lambda *a, **k: pd.DataFrame())

    deriv_dir = tmp_path / "derivatives"
    deriv_dir.mkdir()
    raw_dir = tmp_path / "derivatives_raw"
    raw_dir.mkdir()

    data_refresh.refresh_coinglass(
        coins=["bitcoin"], derivatives_dir=deriv_dir, raw_dir=raw_dir,
        api_key="test", structured_log=None,
    )
    out1 = pd.read_parquet(deriv_dir / "bitcoin.parquet").copy()
    data_refresh.refresh_coinglass(
        coins=["bitcoin"], derivatives_dir=deriv_dir, raw_dir=raw_dir,
        api_key="test", structured_log=None,
    )
    out2 = pd.read_parquet(deriv_dir / "bitcoin.parquet")
    pd.testing.assert_frame_equal(out1, out2)


def test_refresh_coinglass_raises_on_missing_key(tmp_path):
    from tradingagents.execution.live import data_refresh
    with pytest.raises(RuntimeError, match="COINGLASS_API_KEY"):
        data_refresh.refresh_coinglass(
            coins=["bitcoin"], derivatives_dir=tmp_path / "d", raw_dir=tmp_path / "r",
            api_key="", structured_log=None,
        )


def test_refresh_deribit_dvol_appends_yesterday(monkeypatch, tmp_path):
    """One-day pull appends a single new row to the per-currency parquet."""
    import pandas as pd
    from tradingagents.execution.live import data_refresh

    def fake_fetch_dvol(currency, start, end):
        idx = pd.to_datetime(["2026-05-14"], utc=True)
        return pd.DataFrame({
            "dvol_open": [60.0], "dvol_high": [62.0],
            "dvol_low": [59.0], "dvol_close": [61.5],
        }, index=idx)

    monkeypatch.setattr(
        "scripts.fetch_deribit_dvol.fetch_dvol", fake_fetch_dvol
    )

    options_dir = tmp_path / "options"
    options_dir.mkdir()

    data_refresh.refresh_deribit_dvol(
        currencies=["BTC"], options_dir=options_dir, structured_log=None,
    )
    out = pd.read_parquet(options_dir / "btc_dvol.parquet")
    assert len(out) == 1
    assert out["dvol_close"].iloc[0] == 61.5


def test_refresh_deribit_dvol_idempotent(monkeypatch, tmp_path):
    import pandas as pd
    from tradingagents.execution.live import data_refresh

    def fake_fetch_dvol(currency, start, end):
        idx = pd.to_datetime(["2026-05-14"], utc=True)
        return pd.DataFrame({
            "dvol_open": [60.0], "dvol_high": [62.0],
            "dvol_low": [59.0], "dvol_close": [61.5],
        }, index=idx)

    monkeypatch.setattr(
        "scripts.fetch_deribit_dvol.fetch_dvol", fake_fetch_dvol
    )
    options_dir = tmp_path / "options"
    options_dir.mkdir()
    data_refresh.refresh_deribit_dvol(["BTC"], options_dir, None)
    data_refresh.refresh_deribit_dvol(["BTC"], options_dir, None)
    out = pd.read_parquet(options_dir / "btc_dvol.parquet")
    assert len(out) == 1  # not 2


def test_refresh_perp_spot_basis_appends_basis(monkeypatch, tmp_path):
    """Daily refresher adds basis_annual column to per-coin derivatives parquet."""
    import pandas as pd
    from tradingagents.execution.live import data_refresh

    def fake_fetch_klines(url, symbol, start, end):
        idx = pd.to_datetime(["2026-05-14"], utc=True)
        return pd.DataFrame({
            "open": [50000.0], "high": [50500.0],
            "low": [49500.0], "close": [50100.0],
            "volume": [1000.0],
        }, index=idx)

    monkeypatch.setattr(
        "scripts.build_perp_spot_basis.fetch_klines", fake_fetch_klines
    )
    raw = tmp_path / "raw"
    raw.mkdir()
    daily = tmp_path / "daily"
    daily.mkdir()

    data_refresh.refresh_perp_spot_basis(
        symbols=["BTCUSDT"], raw_dir=raw, daily_dir=daily,
        structured_log=None,
    )
    out = pd.read_parquet(daily / "bitcoin.parquet")
    assert "basis_annual" in out.columns
    assert "perp_price" in out.columns
    assert "spot_price" in out.columns


def test_refresh_all_critical_fail_raises(monkeypatch, tmp_path):
    """If OHLCV or CoinMetrics fail, refresh_all raises CriticalDataRefreshError."""
    from tradingagents.execution.live import data_refresh
    from tradingagents.execution.live.data_refresh import CriticalDataRefreshError

    monkeypatch.setattr(data_refresh, "refresh_ohlcv",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("OHLCV API down")))
    monkeypatch.setattr(data_refresh, "refresh_coinmetrics", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_defillama", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_coinglass", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_deribit_dvol", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_perp_spot_basis", lambda *a, **k: None)

    class FakeLog:
        def __init__(self): self.events = []
        def warn(self, event, **kw): self.events.append(("warn", event, kw))
        def info(self, event, **kw): self.events.append(("info", event, kw))

    cfg = _fake_cfg(tmp_path)
    log = FakeLog()
    with pytest.raises(CriticalDataRefreshError) as exc_info:
        data_refresh.refresh_all(cfg, log)
    assert "ohlcv" in str(exc_info.value)


def test_refresh_all_supplementary_fail_continues(monkeypatch, tmp_path):
    """Supplementary failure logs warning, does not raise."""
    from tradingagents.execution.live import data_refresh

    monkeypatch.setattr(data_refresh, "refresh_ohlcv", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_coinmetrics", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_defillama", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_coinglass",
                         lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Coinglass 429")))
    monkeypatch.setattr(data_refresh, "refresh_deribit_dvol", lambda *a, **k: None)
    monkeypatch.setattr(data_refresh, "refresh_perp_spot_basis", lambda *a, **k: None)

    class FakeLog:
        def __init__(self): self.warns = []
        def warn(self, event, **kw): self.warns.append((event, kw))
        def info(self, event, **kw): pass

    cfg = _fake_cfg(tmp_path)
    log = FakeLog()
    result = data_refresh.refresh_all(cfg, log)
    assert result["critical_ok"] is True
    assert "coinglass" in [src for src, _err in result["supplementary_failures"]]
    assert any(e[0] == "supplementary_data_stale" for e in log.warns)


def _fake_cfg(tmp_path):
    """Minimal LiveConfig-like for refresh_all tests."""
    class C:
        coin_universe = ["bitcoin", "ethereum", "binancecoin", "solana"]
        coinmetrics_api_key = "k"
        coinglass_api_key = "k"
        data_root = str(tmp_path)
        data_refresh_critical = {"ohlcv", "coinmetrics"}
    return C()


def test_satellite_symbol_maps_present():
    """8-coin expansion: reverse basis map covers the 4 satellites."""
    import tradingagents.execution.live.data_refresh as dr
    assert dr._BASIS_SYM_TO_COIN["XRPUSDT"] == "ripple"
    assert dr._BASIS_SYM_TO_COIN["DOGEUSDT"] == "dogecoin"
    assert dr._BASIS_SYM_TO_COIN["ADAUSDT"] == "cardano"
    assert dr._BASIS_SYM_TO_COIN["TRXUSDT"] == "tron"
