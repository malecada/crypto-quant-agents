"""Tests for the wired-up live + backtest metric computations."""
import pytest


def test_compute_live_metrics_with_real_db(tmp_path, monkeypatch):
    """Live metrics read portfolio_snapshots and compute a return."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    db = tmp_path / "trade_journal.db"
    db.parent.mkdir(parents=True, exist_ok=True)

    from tradingagents.execution.live.journal import Journal
    j = Journal(str(db))
    j.log_cycle_start("2026-05-12", git_sha="abc")
    for day, val in [
        ("2026-05-12", 10000),
        ("2026-05-13", 10100),
        ("2026-05-14", 10250),
    ]:
        j._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, "
            "usdt_balance, position_qty_per_coin, unrealized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (day, f"{day}T00:05:00+00:00", val, val, "{}", 0),
        )
    j._conn.commit()
    j.close()

    from tradingagents.execution.live.rebacktest import compute_live_metrics
    metrics = compute_live_metrics("2026-05-12", "2026-05-14")
    assert metrics["return_pct"] == pytest.approx(0.025)
    assert metrics["n_trades"] == 3


def test_compute_live_metrics_handles_missing_db(tmp_path, monkeypatch):
    """No DB → safe defaults rather than a crash."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from tradingagents.execution.live.rebacktest import compute_live_metrics
    metrics = compute_live_metrics("2026-05-12", "2026-05-14")
    assert metrics["n_trades"] == 0
