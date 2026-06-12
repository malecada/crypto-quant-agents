import sqlite3

import pytest

from tradingagents.monitor import db


def test_open_journal_is_read_only(journal_path):
    conn = db.open_journal(journal_path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("INSERT INTO cycles (cycle_id, start_ts) VALUES ('x', 'y')")
    conn.close()


def test_open_journal_missing_file_raises():
    with pytest.raises(sqlite3.OperationalError):
        db.open_journal("/nonexistent/path/trade_journal.db")


def test_list_cycles_newest_first(journal_path):
    conn = db.open_journal(journal_path)
    cycles = db.list_cycles(conn)
    assert [c["cycle_id"] for c in cycles] == ["c2", "c1"]
    assert cycles[0]["status"] == "ok"
    conn.close()


def test_latest_cycle(journal_path):
    conn = db.open_journal(journal_path)
    assert db.latest_cycle(conn)["cycle_id"] == "c2"
    conn.close()


def test_latest_cycle_empty(empty_journal_path):
    conn = db.open_journal(empty_journal_path)
    assert db.latest_cycle(conn) is None
    conn.close()


def test_cycle_detail(journal_path):
    conn = db.open_journal(journal_path)
    detail = db.cycle_detail(conn, "c2")
    assert len(detail["predictions"]) == 2
    assert len(detail["sizing"]) == 2
    assert len(detail["risk_checks"]) == 2
    assert len(detail["shadow_decisions"]) == 2
    assert detail["predictions"][0]["coin"] == "bitcoin"
    conn.close()


def test_all_trades(journal_path):
    conn = db.open_journal(journal_path)
    trades = db.all_trades(conn)
    assert len(trades) == 3
    assert trades[0]["status"] == "FAILED"  # newest first
    conn.close()


def test_portfolio_snapshots(journal_path):
    conn = db.open_journal(journal_path)
    snaps = db.portfolio_snapshots(conn)
    assert [s["total_value"] for s in snaps] == [10150.0, 10280.0]  # oldest first
    conn.close()


def test_retrains(journal_path):
    conn = db.open_journal(journal_path)
    rows = db.retrains(conn)
    assert rows[0]["retrain_id"] == "r1"
    conn.close()


def test_modulator_outputs_missing_table_is_empty(journal_path):
    # Simulate an OLD journal (pre-modulator table) by dropping it.
    import sqlite3 as sq
    conn = sq.connect(journal_path)
    conn.execute("DROP TABLE IF EXISTS modulator_outputs")
    conn.commit()
    conn.close()
    ro = db.open_journal(journal_path)
    assert db.modulator_outputs(ro, "c2") == []
    ro.close()


def test_modulator_outputs_rows(journal_path):
    import sqlite3 as sq
    conn = sq.connect(journal_path)
    conn.execute(
        "INSERT INTO modulator_outputs (cycle_id, coin, multiplier, "
        "effective_weight, llm_confidence, regime, fallback) "
        "VALUES ('c2','ethereum',1.2,0.35,0.7,'trend_up',0)")
    conn.commit()
    conn.close()
    ro = db.open_journal(journal_path)
    rows = db.modulator_outputs(ro, "c2")
    ro.close()
    assert rows[0]["coin"] == "ethereum"
    assert rows[0]["multiplier"] == 1.2
    assert rows[0]["fallback"] == 0
