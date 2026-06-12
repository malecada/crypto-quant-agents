"""Shared fixtures for monitor UI tests.

Builds a temporary SQLite journal from the live schema and inserts
representative rows, plus a sample structured-log file.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

_SCHEMA = Path(__file__).resolve().parents[2] / "tradingagents/execution/live/schema.sql"


@pytest.fixture
def journal_path(tmp_path) -> str:
    """A populated journal DB. Returns the file path as a string."""
    db = tmp_path / "trade_journal.db"
    conn = sqlite3.connect(str(db))
    with open(_SCHEMA) as f:
        conn.executescript(f.read())

    conn.executemany(
        "INSERT INTO cycles (cycle_id, start_ts, end_ts, status, n_trades, "
        "critical_data_fail_sources, supplementary_stale_sources) VALUES (?,?,?,?,?,?,?)",
        [
            ("c1", "2026-05-19T07:00:00+00:00", "2026-05-19T07:05:00+00:00", "ok", 2, "", ""),
            ("c2", "2026-05-20T07:00:00+00:00", "2026-05-20T07:05:00+00:00", "ok", 1, "", "gdelt"),
        ],
    )
    conn.executemany(
        "INSERT INTO predictions (cycle_id, coin, horizon, pred_value, "
        "pred_quantile_low, pred_quantile_high, ref_price, signal_h7, signal_h14, "
        "consensus_signal, bundle_route) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("c2", "bitcoin", 7, 0.021, 0.005, 0.040, 68000.0, 1, 1, 1, "78f"),
            ("c2", "ethereum", 7, -0.010, -0.030, 0.008, 3800.0, -1, 0, 0, "193f"),
        ],
    )
    conn.executemany(
        "INSERT INTO sizing (cycle_id, coin, realized_vol, target_vol, kelly, "
        "confidence, base_size, leverage, sma30_multiplier, final_size_notional) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            ("c2", "bitcoin", 0.45, 0.30, 0.25, 0.62, 1000.0, 1.5, 1.0, 1500.0),
            ("c2", "ethereum", 0.55, 0.30, 0.25, 0.40, 0.0, 0.0, 1.0, 0.0),
        ],
    )
    conn.executemany(
        "INSERT INTO risk_checks (cycle_id, coin, check_name, passed, value, "
        "threshold, reason) VALUES (?,?,?,?,?,?,?)",
        [
            ("c2", "bitcoin", "max_leverage", 1, 1.5, 3.0, "ok"),
            ("c2", "ethereum", "min_confidence", 0, 0.40, 0.50, "below threshold"),
        ],
    )
    conn.executemany(
        "INSERT INTO trades (cycle_id, coin, side, qty, entry_price, exit_price, "
        "pnl, fees, slippage, order_id, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        # The live runner logs one row per executed order; exit_price/pnl/fees
        # are never back-filled (V5 is a rebalancing strategy). Real status
        # values are EXECUTED / FAILED / UNPROTECTED.
        [
            ("c1", "bitcoin", "BUY", 0.05, 65000.0, None, None, None, 1.1, "o1", "EXECUTED"),
            ("c1", "ethereum", "SELL", 1.0, 3600.0, None, None, None, 0.6, "o2", "EXECUTED"),
            ("c2", "bitcoin", "BUY", 0.06, 68000.0, None, None, None, 1.4, "o3", "FAILED"),
        ],
    )
    conn.executemany(
        "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, usdt_balance, "
        "position_qty_per_coin, unrealized_pnl) VALUES (?,?,?,?,?,?)",
        [
            ("c1", "2026-05-19T07:05:00+00:00", 10150.0, 6000.0, '{"bitcoin": 0.05}', 0.0),
            ("c2", "2026-05-20T07:05:00+00:00", 10280.0, 4000.0, '{"bitcoin": 0.06, "ethereum": 1.4}', 80.0),
        ],
    )
    conn.executemany(
        "INSERT INTO retrains (retrain_id, cycle_id, n_train_rows, "
        "train_window_start, train_dir_acc, status, routes) VALUES (?,?,?,?,?,?,?)",
        [
            ("r1", "c2", 2500, "2019-05-01", 0.58, "ok", "78f,193f"),
        ],
    )
    conn.executemany(
        "INSERT INTO shadow_decisions (cycle_id, coin, live_signal, "
        "backtest_signal, agree, live_size, backtest_size, size_delta_pct) "
        "VALUES (?,?,?,?,?,?,?,?)",
        [
            ("c2", "bitcoin", 1, 1, 1, 1500.0, 1520.0, -1.3),
            ("c2", "ethereum", 0, -1, 0, 0.0, 200.0, -100.0),
        ],
    )
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def empty_journal_path(tmp_path) -> str:
    """An empty but schema-valid journal DB."""
    db = tmp_path / "empty_journal.db"
    conn = sqlite3.connect(str(db))
    with open(_SCHEMA) as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    return str(db)


@pytest.fixture
def log_dir(tmp_path) -> str:
    """A log directory with one sample cycle structured-log file."""
    d = tmp_path / "logs"
    d.mkdir()
    records = [
        {"ts": "2026-05-20T07:00:00+00:00", "cycle_id": "c2", "step": "data_refresh",
         "status": "ok", "duration_ms": 120, "payload": {}},
        {"ts": "2026-05-20T07:01:00+00:00", "cycle_id": "c2", "step": "predict",
         "status": "ok", "duration_ms": 30, "payload": {}},
        {"ts": "2026-05-20T07:02:00+00:00", "cycle_id": "c2", "step": "execute",
         "status": "error", "duration_ms": 90, "payload": {"error": "binance timeout"}},
    ]
    with open(d / "cycle_c2.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(d)


@pytest.fixture
def hybrid_journal_path(tmp_path) -> str:
    """A small hybrid journal: 1 overlapping cycle + modulator rows."""
    db = tmp_path / "hybrid" / "trade_journal.db"
    db.parent.mkdir()
    conn = sqlite3.connect(str(db))
    with open(_SCHEMA) as f:
        conn.executescript(f.read())
    conn.execute(
        "INSERT INTO cycles (cycle_id, start_ts, end_ts, status, n_trades) "
        "VALUES ('c2','2026-05-20T08:00:00+00:00','2026-05-20T08:20:00+00:00','ok',1)")
    conn.execute(
        "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, usdt_balance, "
        "position_qty_per_coin, unrealized_pnl) VALUES "
        "('c2','2026-05-20T08:20:00+00:00',10100.0,5000.0,'{\"ethereum\": 1.0}',20.0)")
    conn.execute(
        "INSERT INTO modulator_outputs (cycle_id, coin, multiplier, "
        "effective_weight, llm_confidence, regime, fallback) VALUES "
        "('c2','ethereum',1.2,0.35,0.7,'trend_up',0)")
    conn.execute(
        "INSERT INTO trades (cycle_id, coin, side, qty, entry_price, slippage, "
        "order_id, status) VALUES ('c2','ethereum','BUY',1.0,3800.0,0.4,'h1','EXECUTED')")
    conn.commit()
    conn.close()
    return str(db)


def _fake_snapshot(positions=None, equity=10350.0, usdt=7000.0):
    def snap():
        return {"positions": positions or [], "usdt_free": usdt,
                "equity": equity, "income": None}
    return snap


@pytest.fixture
def dual_app(journal_path, hybrid_journal_path, log_dir, monkeypatch):
    """create_app with quant+hybrid sources and fake snapshot providers."""
    from tradingagents.monitor.app import create_app
    from tradingagents.monitor.sources import StrategySource
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    quant = StrategySource("quant", journal_path, _fake_snapshot(positions=[
        {"symbol": "BTCUSDT", "qty": 0.05, "entry_price": 65000.0,
         "mark_price": 66000.0, "upnl": 50.0, "leverage": 3.0,
         "liq_price": 30000.0, "notional": 3300.0}]))
    hybrid = StrategySource("hybrid", hybrid_journal_path, _fake_snapshot(
        positions=[{"symbol": "ETHUSDT", "qty": 1.0, "entry_price": 3800.0,
                    "mark_price": 3900.0, "upnl": 100.0, "leverage": 2.0,
                    "liq_price": 1900.0, "notional": 3900.0}], equity=10100.0))
    return create_app(quant=quant, hybrid=hybrid, log_dir=log_dir,
                      start_capital=10000.0)


@pytest.fixture
def dual_client(dual_app):
    from fastapi.testclient import TestClient
    c = TestClient(dual_app)
    c.auth = ("admin", "pw")
    return c
