"""L1 fix — the daily-loss kill switch was dead under once-daily cadence.

`compute_live_metrics(today, today)` only ever sees today's single snapshot
(`len(rows) < 2` -> return_pct=0.0), so `check_daily_loss` never tripped.
The fix computes today's PnL against the prior-day close (using the live
current equity, which the runner already reads) and adds a drawdown-from-peak
halt mirroring the backtest's 15% portfolio circuit breaker.
"""
import pytest


def _seed(db_path, rows):
    """rows: list of (cycle_id_date, total_value)."""
    from tradingagents.execution.live.journal import Journal
    j = Journal(str(db_path))
    for day, val in rows:
        j._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, "
            "usdt_balance, position_qty_per_coin, unrealized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (day, f"{day}T00:06:00+00:00", val, val, "{}", 0),
        )
    j._conn.commit()
    j.close()


def test_daily_pnl_vs_prior_day_close(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "trade_journal.db", [("2026-05-28", 10_000)])
    from tradingagents.execution.live.rebacktest import compute_daily_pnl_pct

    # current live equity 9000 vs yesterday close 10000 -> -10%
    pnl = compute_daily_pnl_pct(9_000.0, "2026-05-29")
    assert pnl == pytest.approx(-0.10)


def test_daily_pnl_zero_when_no_prior_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "trade_journal.db", [])  # creates empty schema only
    from tradingagents.execution.live.rebacktest import compute_daily_pnl_pct

    assert compute_daily_pnl_pct(9_000.0, "2026-05-29") == 0.0


def test_daily_loss_gate_fires_on_real_input(tmp_path, monkeypatch):
    """End-to-end of the dead path: a real -16% day must now trip the gate."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "trade_journal.db", [("2026-05-28", 10_000)])
    from tradingagents.execution.live.rebacktest import compute_daily_pnl_pct
    from tradingagents.execution.live.risk import check_daily_loss

    pnl = compute_daily_pnl_pct(8_400.0, "2026-05-29")  # -16%
    ok, why = check_daily_loss(pnl, 0.15)
    assert ok is False
    assert "KILL SWITCH" in why


def test_drawdown_from_peak(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "trade_journal.db",
          [("2026-05-26", 10_000), ("2026-05-27", 12_000), ("2026-05-28", 11_000)])
    from tradingagents.execution.live.rebacktest import compute_drawdown_from_peak

    # current 9600 vs running peak 12000 -> 20% drawdown
    dd = compute_drawdown_from_peak(9_600.0, "2026-05-29")
    assert dd == pytest.approx(0.20)


def test_drawdown_zero_at_new_peak(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed(tmp_path / "trade_journal.db",
          [("2026-05-27", 12_000), ("2026-05-28", 11_000)])
    from tradingagents.execution.live.rebacktest import compute_drawdown_from_peak

    assert compute_drawdown_from_peak(13_000.0, "2026-05-29") == 0.0


def test_check_drawdown_gate():
    from tradingagents.execution.live.risk import check_drawdown

    ok, why = check_drawdown(0.16, 0.15)
    assert ok is False
    assert "KILL SWITCH" in why
    ok2, _ = check_drawdown(0.10, 0.15)
    assert ok2 is True
