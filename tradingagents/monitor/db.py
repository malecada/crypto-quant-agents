"""Read-only access to the live bot's SQLite forensic journal.

Every connection is opened with the SQLite ``mode=ro`` URI so the UI can
never write to or lock the bot's database.
"""
from __future__ import annotations

import sqlite3


def open_journal(db_path: str) -> sqlite3.Connection:
    """Open the journal DB read-only. Raises sqlite3.OperationalError if the
    file is missing or unreadable."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def list_cycles(conn: sqlite3.Connection, limit: int = 200) -> list[dict]:
    """Cycles, newest first."""
    return _rows(
        conn,
        "SELECT cycle_id, start_ts, end_ts, status, error_msg, n_trades, "
        "critical_data_fail_sources, supplementary_stale_sources "
        "FROM cycles ORDER BY start_ts DESC LIMIT ?",
        (limit,),
    )


def latest_cycle(conn: sqlite3.Connection) -> dict | None:
    rows = list_cycles(conn, limit=1)
    return rows[0] if rows else None


def cycle_detail(conn: sqlite3.Connection, cycle_id: str) -> dict:
    """Predictions, sizing, risk checks and shadow decisions for one cycle."""
    return {
        "predictions": _rows(
            conn, "SELECT * FROM predictions WHERE cycle_id = ? ORDER BY coin",
            (cycle_id,)),
        "sizing": _rows(
            conn, "SELECT * FROM sizing WHERE cycle_id = ? ORDER BY coin",
            (cycle_id,)),
        "risk_checks": _rows(
            conn, "SELECT * FROM risk_checks WHERE cycle_id = ? ORDER BY coin",
            (cycle_id,)),
        "shadow_decisions": _rows(
            conn, "SELECT * FROM shadow_decisions WHERE cycle_id = ? ORDER BY coin",
            (cycle_id,)),
    }


def all_trades(conn: sqlite3.Connection) -> list[dict]:
    """All trades, newest first (by row id)."""
    return _rows(conn, "SELECT * FROM trades ORDER BY id DESC")


def portfolio_snapshots(conn: sqlite3.Connection) -> list[dict]:
    """Portfolio snapshots, oldest first (chronological for charting)."""
    return _rows(conn, "SELECT * FROM portfolio_snapshots ORDER BY ts ASC")


def retrains(conn: sqlite3.Connection) -> list[dict]:
    """Retrain history, newest first."""
    return _rows(conn, "SELECT * FROM retrains ORDER BY rowid DESC")


def modulator_outputs(conn: sqlite3.Connection, cycle_id: str) -> list[dict]:
    """Hybrid modulator rows for one cycle. Empty list when the journal
    predates the modulator_outputs table (quant journals never have it)."""
    try:
        return _rows(
            conn,
            "SELECT * FROM modulator_outputs WHERE cycle_id = ? ORDER BY coin",
            (cycle_id,))
    except sqlite3.OperationalError:
        return []
