"""Cycle-level position snapshot (exsym-3/exsym-4 audit fixes).

The runner must (1) never trade against an *assumed* position when Binance
cannot report the real one — a fetch failure treated as "flat" can double a
live position or skip a required close — and (2) read positions from one
whole-account ``get_open_positions()`` call per cycle instead of ~2 per-symbol
calls per coin plus an O(N²) max-positions sweep.
"""
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def env_setup(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_MODE", "false")
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
    monkeypatch.setenv("COINGLASS_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    return tmp_path


def _seed_ohlcv_cache(data_dir: Path, seed: int) -> None:
    cache_dir = data_dir / "ohlcv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-03-01", periods=60, freq="D")
    for coin, base_price in [("BTC", 60000.0), ("ETH", 3000.0), ("BNB", 400.0)]:
        prices = base_price * np.exp(np.cumsum(rng.normal(0, 0.02, 60)))
        df = pd.DataFrame({"date": dates, "close": prices})
        df.to_parquet(cache_dir / f"{coin}USDT_1d.parquet", index=False)


_PREDS = pd.DataFrame([
    {"coin": "bitcoin", "horizon": 7, "prediction": 63000.0,
     "ref_price": 60000.0, "bundle_route": "bitcoin_78f"},
    {"coin": "bitcoin", "horizon": 14, "prediction": 66000.0,
     "ref_price": 60000.0, "bundle_route": "bitcoin_78f"},
    {"coin": "ethereum", "horizon": 7, "prediction": 3050.0,
     "ref_price": 3000.0, "bundle_route": "ethereum_193f"},
    {"coin": "ethereum", "horizon": 14, "prediction": 3100.0,
     "ref_price": 3000.0, "bundle_route": "ethereum_193f"},
    {"coin": "binancecoin", "horizon": 7, "prediction": 410.0,
     "ref_price": 400.0, "bundle_route": "binancecoin_78f"},
    {"coin": "binancecoin", "horizon": 14, "prediction": 420.0,
     "ref_price": 400.0, "bundle_route": "binancecoin_78f"},
])


def _run(env_setup, mock_ex, seed=7, cycle_id="2026-05-12"):
    from tradingagents.execution.live import runner

    data_dir = env_setup / "data"
    _seed_ohlcv_cache(data_dir, seed=seed)

    with patch("tradingagents.execution.live.data_refresh.refresh_all",
               return_value={"critical_ok": True, "supplementary_failures": []}), \
         patch("tradingagents.execution.live.retrain.run_retrain_with_fallback") as mock_retrain, \
         patch("tradingagents.execution.live.predict.run_predict",
               return_value=_PREDS), \
         patch("tradingagents.execution.live.runner.ExchangeClient",
               return_value=mock_ex), \
         patch("tradingagents.execution.live.notify.send_daily_summary"), \
         patch("tradingagents.execution.live.notify.send_alert") as mock_alert:
        mock_retrain.return_value = MagicMock(
            path=Path("/tmp/m.pkl"), sha="c" * 64, retrain_id=cycle_id,
            routes=["bitcoin_78f", "ethereum_193f", "binancecoin_78f"],
            n_train_rows=100, train_window_start="2024-01-01",
            train_dir_acc=0.0,
        )
        result = runner.run_cycle(cycle_id=cycle_id, dry_run=True)
    return result, mock_alert


def _trades(env_setup):
    import sqlite3
    db = env_setup / "data" / "trade_journal.db"
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT coin, status FROM trades").fetchall()
    conn.close()
    return rows


def test_snapshot_failure_skips_trading(env_setup):
    """get_open_positions failing for the cycle ⇒ ZERO trades — positions are
    unknown and trading against an assumed-flat book is forbidden."""
    mock_ex = MagicMock()
    mock_ex.get_total_portfolio_value.return_value = 10000.0
    mock_ex.get_usdt_balance.return_value = 10000.0
    mock_ex.get_open_positions.side_effect = RuntimeError("binance 5xx")
    mock_ex.get_current_position.return_value = 0.0  # snapshot fallback only
    mock_ex.get_ticker_price.return_value = 60000.0

    result, mock_alert = _run(env_setup, mock_ex)

    assert _trades(env_setup) == []  # nothing traded, not even DRY_RUN rows
    severities = [
        kw.get("severity") or (a[2] if len(a) > 2 else None)
        for a, kw in mock_alert.call_args_list
    ]
    assert any(s and "POSITION" in str(s) for s in severities), (
        f"expected a POSITION_SNAPSHOT_FAILED alert, got {severities}"
    )


def test_positions_come_from_single_snapshot_call(env_setup):
    """With a healthy snapshot, the execute path must not fall back to
    per-symbol get_current_position calls (the O(N²) / ~80-call pattern)."""
    mock_ex = MagicMock()
    mock_ex.get_total_portfolio_value.return_value = 10000.0
    mock_ex.get_usdt_balance.return_value = 10000.0
    mock_ex.get_open_positions.return_value = [
        {"symbol": "BTCUSDT", "qty": 0.001, "usd": 60.0},
    ]
    mock_ex.get_ticker_price.return_value = 60000.0

    result, _ = _run(env_setup, mock_ex)

    assert result.status == "ok"
    mock_ex.get_current_position.assert_not_called()
    # dry-run still records intended trades — the snapshot is used, not bypassed
    assert all(status == "DRY_RUN" for _, status in _trades(env_setup))


def test_journal_snapshot_uses_single_call(env_setup):
    """Step-9 portfolio snapshot derives per-coin qty from the same
    single-call source: position present in get_open_positions appears in the
    journal row; absent coins are zero."""
    import sqlite3, json

    mock_ex = MagicMock()
    mock_ex.get_total_portfolio_value.return_value = 10000.0
    mock_ex.get_usdt_balance.return_value = 10000.0
    mock_ex.get_open_positions.return_value = [
        {"symbol": "ETHUSDT", "qty": -0.5, "usd": -1500.0},
    ]
    mock_ex.get_ticker_price.return_value = 3000.0

    _run(env_setup, mock_ex)

    db = env_setup / "data" / "trade_journal.db"
    conn = sqlite3.connect(db)
    raw = conn.execute(
        "SELECT position_qty_per_coin FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
    ).fetchone()[0]
    conn.close()
    qty = json.loads(raw)
    assert qty.get("ethereum") == -0.5
    assert qty.get("bitcoin", 0.0) == 0.0
    mock_ex.get_current_position.assert_not_called()
