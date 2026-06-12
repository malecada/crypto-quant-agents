"""Runner orchestrator tests — end-to-end dry-run with mocked boundaries."""
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


def test_run_cycle_refuses_to_trade_when_halted(env_setup):
    """R4: a persistent HALT sentinel short-circuits the cycle before any data
    refresh or exchange access, so a tripped halt does not auto-resume."""
    from tradingagents.execution.live import runner, halt

    data_dir = env_setup / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    halt.write_halt("cycle 2026-05-11: daily PnL -16% — KILL SWITCH",
                    data_dir=data_dir)

    with patch("tradingagents.execution.live.data_refresh.refresh_all") as mock_refresh, \
         patch("tradingagents.execution.live.runner.ExchangeClient") as mock_ex_cls, \
         patch("tradingagents.execution.live.notify.send_alert"):
        result = runner.run_cycle(dry_run=True)

    assert result.status == "halted"
    assert "KILL SWITCH" in result.error_msg
    mock_refresh.assert_not_called()
    mock_ex_cls.assert_not_called()


def test_dry_run_completes_full_pipeline(env_setup):
    """End-to-end dry run: no real orders, all steps execute, summary sent."""
    from tradingagents.execution.live import runner

    data_dir = env_setup / "data"
    _seed_ohlcv_cache(data_dir, seed=42)

    with patch("tradingagents.execution.live.data_refresh.refresh_all",
                return_value={"critical_ok": True, "supplementary_failures": []}), \
         patch("tradingagents.execution.live.retrain.run_retrain_with_fallback") as mock_retrain, \
         patch("tradingagents.execution.live.predict.run_predict") as mock_pred, \
         patch("tradingagents.execution.live.runner.ExchangeClient") as mock_ex_cls, \
         patch("tradingagents.execution.live.notify.send_daily_summary") as mock_notify:
        mock_retrain.return_value = MagicMock(
            path=Path("/tmp/m.pkl"),
            sha="a" * 64,
            retrain_id="2026-05-12",
            routes=["bitcoin_78f", "ethereum_193f", "binancecoin_78f"],
            n_train_rows=100,
            train_window_start="2024-01-01",
            train_dir_acc=0.0,
        )
        mock_pred.return_value = pd.DataFrame([
            {"coin": "bitcoin", "horizon": 7, "prediction": 63000.0,
             "ref_price": 60000.0, "bundle_route": "bitcoin_78f"},
            {"coin": "bitcoin", "horizon": 14, "prediction": 66000.0,
             "ref_price": 60000.0, "bundle_route": "bitcoin_78f"},
            {"coin": "ethereum", "horizon": 7, "prediction": 2950.0,
             "ref_price": 3000.0, "bundle_route": "ethereum_193f"},
            {"coin": "ethereum", "horizon": 14, "prediction": 2900.0,
             "ref_price": 3000.0, "bundle_route": "ethereum_193f"},
            {"coin": "binancecoin", "horizon": 7, "prediction": 410.0,
             "ref_price": 400.0, "bundle_route": "binancecoin_78f"},
            {"coin": "binancecoin", "horizon": 14, "prediction": 405.0,
             "ref_price": 400.0, "bundle_route": "binancecoin_78f"},
        ])
        mock_ex_cls.return_value.get_total_portfolio_value.return_value = 10000.0
        mock_ex_cls.return_value.get_usdt_balance.return_value = 10000.0
        mock_ex_cls.return_value.get_current_position.return_value = 0.0
        mock_ex_cls.return_value.get_ticker_price.return_value = 60000.0

        result = runner.run_cycle(cycle_id="2026-05-12", dry_run=True)

    assert result.status == "ok"
    assert result.n_executed == 0  # dry-run skips real orders
    mock_notify.assert_called_once()


def test_run_cycle_logs_to_journal(env_setup):
    """After a cycle, the journal DB has rows in cycles/sizing/shadow/snapshots."""
    from tradingagents.execution.live import runner
    import sqlite3

    data_dir = env_setup / "data"
    _seed_ohlcv_cache(data_dir, seed=7)

    with patch("tradingagents.execution.live.data_refresh.refresh_all",
                return_value={"critical_ok": True, "supplementary_failures": []}), \
         patch("tradingagents.execution.live.retrain.run_retrain_with_fallback") as mock_retrain, \
         patch("tradingagents.execution.live.predict.run_predict") as mock_pred, \
         patch("tradingagents.execution.live.runner.ExchangeClient") as mock_ex_cls, \
         patch("tradingagents.execution.live.notify.send_daily_summary"):
        mock_retrain.return_value = MagicMock(
            path=Path("/tmp/m.pkl"),
            sha="b" * 64,
            retrain_id="2026-05-12",
            routes=["bitcoin_78f", "ethereum_193f", "binancecoin_78f"],
            n_train_rows=100,
            train_window_start="2024-01-01",
            train_dir_acc=0.0,
        )
        mock_pred.return_value = pd.DataFrame([
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
        mock_ex_cls.return_value.get_total_portfolio_value.return_value = 10000.0
        mock_ex_cls.return_value.get_usdt_balance.return_value = 10000.0
        mock_ex_cls.return_value.get_current_position.return_value = 0.0
        mock_ex_cls.return_value.get_ticker_price.return_value = 60000.0

        runner.run_cycle(cycle_id="2026-05-12", dry_run=True)

    db = data_dir / "trade_journal.db"
    assert db.exists()
    conn = sqlite3.connect(db)
    cycles = conn.execute("SELECT cycle_id, status FROM cycles").fetchall()
    assert cycles == [("2026-05-12", "ok")]
    sizing_rows = conn.execute("SELECT COUNT(*) FROM sizing").fetchone()[0]
    assert sizing_rows == 3  # BTC, ETH, BNB
    shadow_rows = conn.execute("SELECT COUNT(*) FROM shadow_decisions").fetchone()[0]
    assert shadow_rows == 3
    snapshot_rows = conn.execute(
        "SELECT COUNT(*) FROM portfolio_snapshots"
    ).fetchone()[0]
    assert snapshot_rows >= 1
    conn.close()


def test_runner_uses_v5_routing(monkeypatch, tmp_path):
    """Runner threads routing through retrain + predict."""
    from tradingagents.execution.live import runner, data_refresh, retrain, predict
    import pandas as pd

    monkeypatch.setenv("COINGLASS_API_KEY", "test")
    monkeypatch.setenv("BINANCE_API_KEY", "x")
    monkeypatch.setenv("BINANCE_API_SECRET", "y")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))

    monkeypatch.setattr(data_refresh, "refresh_all",
                         lambda cfg, log: {"critical_ok": True, "supplementary_failures": []})

    captured_routing = []

    def fake_retrain_with_fallback(**kw):
        captured_routing.append(kw.get("routing"))
        from tradingagents.execution.live.retrain import CheckpointArtifact
        from pathlib import Path
        p = Path(tmp_path) / "ckpt" / "lgb_v5_mix_X.pkl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("")
        return CheckpointArtifact(
            path=p, sha="x", retrain_id="x",
            routes=["bitcoin_78f", "ethereum_193f", "binancecoin_78f", "solana_193f"],
        )
    monkeypatch.setattr(retrain, "run_retrain_with_fallback", fake_retrain_with_fallback)

    captured_predict_routing = []

    def fake_run_predict(**kw):
        captured_predict_routing.append(kw.get("routing"))
        return pd.DataFrame([])  # empty preds → no trades
    monkeypatch.setattr(predict, "run_predict", fake_run_predict)

    result = runner.run_cycle(cycle_id="20260514-test", dry_run=True)

    assert captured_routing[0] is not None
    # Default universe is the 8-coin V5 MIX (4 core + 4 satellite).
    assert set(captured_routing[0].keys()) == {
        "bitcoin", "ethereum", "binancecoin", "solana",
        "ripple", "dogecoin", "cardano", "tron",
    }
    assert captured_predict_routing[0] == captured_routing[0]


_BULLISH_3COIN_PREDS = [
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
]


def _retrain_artifact():
    return MagicMock(
        path=Path("/tmp/m.pkl"), sha="c" * 64, retrain_id="2026-05-12",
        routes=["bitcoin_78f", "ethereum_193f", "binancecoin_78f"],
        n_train_rows=100, train_window_start="2024-01-01", train_dir_acc=0.0,
    )


def test_below_min_notional_opening_order_skipped(env_setup):
    """A: a fresh (non-reduce-only) order whose notional is under the symbol's
    MIN_NOTIONAL is skipped cleanly, not sent — Binance rejects such dust and
    the runner logged FAILED rows for them on the live testnet."""
    from tradingagents.execution.live import runner
    import sqlite3

    data_dir = env_setup / "data"
    _seed_ohlcv_cache(data_dir, seed=7)

    with patch("tradingagents.execution.live.data_refresh.refresh_all",
                return_value={"critical_ok": True, "supplementary_failures": []}), \
         patch("tradingagents.execution.live.retrain.run_retrain_with_fallback") as mock_retrain, \
         patch("tradingagents.execution.live.predict.run_predict") as mock_pred, \
         patch("tradingagents.execution.live.runner.ExchangeClient") as mock_ex_cls, \
         patch("tradingagents.execution.live.notify.send_daily_summary"):
        mock_retrain.return_value = _retrain_artifact()
        mock_pred.return_value = pd.DataFrame(_BULLISH_3COIN_PREDS)
        ex = mock_ex_cls.return_value
        ex.get_total_portfolio_value.return_value = 10000.0
        ex.get_usdt_balance.return_value = 10000.0
        ex.get_current_position.return_value = 0.0
        ex.get_ticker_price.return_value = 60000.0
        ex.min_notional.return_value = 1e12   # every opening order is "dust"

        result = runner.run_cycle(cycle_id="2026-05-12", dry_run=True)

    assert result.status == "ok"
    conn = sqlite3.connect(data_dir / "trade_journal.db")
    n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    conn.close()
    assert n_trades == 0   # all below-min-notional opens skipped, none attempted


def test_shadow_logs_stateless_size_not_held_fraction(env_setup):
    """B: shadow_decisions.live_size is the STATELESS sizing result (comparable
    to the stateless backtest_size), not the min-hold-adjusted held_fraction.
    On a held bar the two differ; logging held_fraction made the size-parity
    diagnostic cry wolf every hold cycle."""
    from tradingagents.execution.live import runner
    from tradingagents.execution.live.journal import Journal
    import sqlite3

    data_dir = env_setup / "data"
    _seed_ohlcv_cache(data_dir, seed=11)
    data_dir.mkdir(parents=True, exist_ok=True)
    # Freeze a long hold for bitcoin with an entry_base distinct from what fresh
    # sizing produces this bar, so held_fraction != stateless final_size_notional.
    j = Journal(str(data_dir / "trade_journal.db"))
    j.upsert_hold_state(coin="bitcoin", current_dir=1, bars_held=2,
                        entry_price=60000.0, entry_base=0.123456,
                        entry_cycle="2026-05-11")
    j.close()

    with patch("tradingagents.execution.live.data_refresh.refresh_all",
                return_value={"critical_ok": True, "supplementary_failures": []}), \
         patch("tradingagents.execution.live.retrain.run_retrain_with_fallback") as mock_retrain, \
         patch("tradingagents.execution.live.predict.run_predict") as mock_pred, \
         patch("tradingagents.execution.live.runner.ExchangeClient") as mock_ex_cls, \
         patch("tradingagents.execution.live.notify.send_daily_summary"):
        mock_retrain.return_value = _retrain_artifact()
        mock_pred.return_value = pd.DataFrame(_BULLISH_3COIN_PREDS)
        ex = mock_ex_cls.return_value
        ex.get_total_portfolio_value.return_value = 10000.0
        ex.get_usdt_balance.return_value = 10000.0
        ex.get_current_position.return_value = 0.0
        ex.get_ticker_price.return_value = 60000.0

        runner.run_cycle(cycle_id="2026-05-12", dry_run=True)

    conn = sqlite3.connect(data_dir / "trade_journal.db")
    rows = conn.execute(
        "SELECT live_size, backtest_size FROM shadow_decisions WHERE coin='bitcoin'"
    ).fetchall()
    conn.close()
    assert rows, "expected a shadow_decisions row for the held bitcoin position"
    for live_size, backtest_size in rows:
        assert live_size == pytest.approx(backtest_size)
