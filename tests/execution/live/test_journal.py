import os
import sqlite3
from datetime import datetime, timezone

import pytest


@pytest.fixture
def journal(tmp_path):
    from tradingagents.execution.live.journal import Journal
    db_path = tmp_path / "j.db"
    j = Journal(str(db_path))
    yield j
    j.close()


def test_creates_all_tables(journal):
    cur = journal._conn.cursor()
    rows = cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {r[0] for r in rows}
    assert {"cycles", "predictions", "sizing", "risk_checks", "trades",
            "portfolio_snapshots", "feature_snapshots",
            "model_artifacts", "shadow_decisions"}.issubset(names)


def test_log_cycle_round_trip(journal):
    journal.log_cycle_start("2026-05-12", git_sha="abc1234")
    journal.log_cycle_end("2026-05-12", status="ok")

    rows = journal._conn.execute("SELECT cycle_id, status FROM cycles").fetchall()
    assert rows == [("2026-05-12", "ok")]


def test_log_prediction_round_trip(journal):
    journal.log_cycle_start("2026-05-12", git_sha="abc")
    journal.log_prediction(cycle_id="2026-05-12", coin="BTC",
                            horizon=7, model_path_sha="sha7",
                            pred_value=70000.0, ref_price=68000.0,
                            signal_h7=1, signal_h14=1, consensus_signal=1)
    rows = journal._conn.execute(
        "SELECT coin, horizon, pred_value, consensus_signal FROM predictions"
    ).fetchall()
    assert rows == [("BTC", 7, 70000.0, 1)]


def test_log_risk_check_passed_and_failed(journal):
    journal.log_cycle_start("2026-05-12", git_sha="abc")
    journal.log_risk_check("2026-05-12", "BTC", "leverage_cap", True, 2.0, 3.0, "OK")
    journal.log_risk_check("2026-05-12", "BTC", "daily_loss", False, -0.20, -0.15, "kill")
    rows = journal._conn.execute(
        "SELECT check_name, passed FROM risk_checks ORDER BY id"
    ).fetchall()
    assert rows == [("leverage_cap", 1), ("daily_loss", 0)]


def test_idempotent_cycle_start(journal):
    journal.log_cycle_start("2026-05-12", git_sha="abc")
    journal.log_cycle_start("2026-05-12", git_sha="abc")  # safe re-call
    rows = journal._conn.execute("SELECT COUNT(*) FROM cycles").fetchone()
    assert rows[0] == 1


# ── Trade fill reconciliation (V5 parity gap #2) ──────────────────────


def test_log_trade_returns_inserted_row_id(journal):
    """log_trade must return the autoincrement id so callers can later
    update fees/realized_pnl once Binance reports fills."""
    journal.log_cycle_start("2026-05-18", git_sha="abc")
    trade_id = journal.log_trade(
        cycle_id="2026-05-18", coin="bitcoin", side="BUY", qty=0.001,
        entry_price=77000.0, exit_price=None, pnl=None, fees=None,
        slippage=0.0, order_id="orderA", stop_loss_id=None, status="EXECUTED",
    )
    assert isinstance(trade_id, int) and trade_id > 0
    row = journal._conn.execute(
        "SELECT id, coin FROM trades WHERE id=?", (trade_id,)
    ).fetchone()
    assert row == (trade_id, "bitcoin")


def test_update_trade_fills_writes_fees_and_realized_pnl(journal):
    """update_trade_fills must populate fees + pnl for an existing trade row
    and leave all other columns unchanged."""
    journal.log_cycle_start("2026-05-18", git_sha="abc")
    trade_id = journal.log_trade(
        cycle_id="2026-05-18", coin="bitcoin", side="BUY", qty=0.001,
        entry_price=77000.0, exit_price=None, pnl=None, fees=None,
        slippage=0.0042, order_id="orderA", stop_loss_id="stopA",
        status="EXECUTED",
    )

    journal.update_trade_fills(trade_id, fees=0.0782, realized_pnl=2.31)

    row = journal._conn.execute(
        "SELECT coin, side, qty, entry_price, slippage, order_id, "
        "stop_loss_id, status, fees, pnl FROM trades WHERE id=?",
        (trade_id,),
    ).fetchone()
    coin, side, qty, entry_price, slippage, order_id, stop_loss_id, status, fees, pnl = row
    assert (coin, side, status) == ("bitcoin", "BUY", "EXECUTED")
    assert qty == 0.001 and entry_price == 77000.0
    assert slippage == 0.0042
    assert (order_id, stop_loss_id) == ("orderA", "stopA")
    assert fees == 0.0782
    assert pnl == 2.31


def test_update_trade_fills_unknown_id_does_not_raise(journal):
    """SQL UPDATE on a non-existent trade_id should be a no-op (callers may
    race with a journal that was rotated/migrated)."""
    journal.update_trade_fills(99999, fees=0.1, realized_pnl=0.0)
    n = journal._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert n == 0


def test_journal_uses_wal_and_busy_timeout(journal):
    """J1: WAL mode + busy_timeout so concurrent connections don't abort cycles."""
    mode = journal._conn.execute("PRAGMA journal_mode").fetchone()[0]
    busy = journal._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert mode.lower() == "wal"
    assert busy >= 10000
