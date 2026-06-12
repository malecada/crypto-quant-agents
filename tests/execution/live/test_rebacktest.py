import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _seed_journal(db_path):
    """Two portfolio snapshots so compute_live_metrics returns real numbers."""
    from tradingagents.execution.live.journal import Journal
    j = Journal(str(db_path))
    j.log_cycle_start("2026-05-13", git_sha="abc")
    for day, val in [("2026-05-13", 10000), ("2026-05-20", 10300)]:
        j._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, "
            "usdt_balance, position_qty_per_coin, unrealized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (day, f"{day}T00:05:00+00:00", val, val, "{}", 0),
        )
    j._conn.commit()
    j.close()


def test_run_weekly_parity_parses_verdict(tmp_path, monkeypatch):
    """run_weekly_parity captures the parity script's VERDICT line."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed_journal(tmp_path / "trade_journal.db")
    from tradingagents.execution.live import rebacktest

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(
            stdout="...\nVERDICT: PASS\nREPORT: /tmp/sandbox/parity_report.md\n",
            stderr="", returncode=0,
        )

    with patch.object(rebacktest.subprocess, "run", side_effect=fake_run):
        out = rebacktest.run_weekly_parity(
            week_end="2026-W21",
            live_start_date="2026-05-13", live_end_date="2026-05-20",
            output_dir=tmp_path / "reports",
        )

    data = json.loads(out.read_text())
    assert data["week_end"] == "2026-W21"
    assert data["verdict"] == "PASS"
    assert data["parity_report"] == "/tmp/sandbox/parity_report.md"
    assert data["live"]["return_pct"] == pytest.approx(0.03)


def test_run_weekly_parity_uses_sys_executable_and_iso_dates(tmp_path, monkeypatch):
    """Subprocess launches via sys.executable; ISO dates passed straight through."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from tradingagents.execution.live import rebacktest

    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        return SimpleNamespace(stdout="VERDICT: PASS\n", stderr="", returncode=0)

    with patch.object(rebacktest.subprocess, "run", side_effect=fake_run):
        rebacktest.run_weekly_parity(
            week_end="2026-W21",
            live_start_date="2026-05-13", live_end_date="2026-05-20",
            output_dir=tmp_path / "reports",
        )

    cmd = captured["cmd"]
    assert cmd[0] == sys.executable
    assert cmd[0] != "python"
    assert "parity_refetch_and_replay.py" in cmd[1]
    # ISO dates match the live runner cycle_id format — no YYYYMMDD conversion.
    assert "--start-date" in cmd and "2026-05-13" in cmd
    assert "--end-date" in cmd and "2026-05-20" in cmd


def test_run_weekly_parity_writes_error_summary_on_failure(tmp_path, monkeypatch):
    """A failed parity subprocess still produces a JSON summary with ERROR."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from tradingagents.execution.live import rebacktest

    err = rebacktest.subprocess.CalledProcessError(
        returncode=1, cmd=["x"], output="partial out", stderr="boom",
    )
    with patch.object(rebacktest.subprocess, "run", side_effect=err):
        out = rebacktest.run_weekly_parity(
            week_end="2026-W21",
            live_start_date="2026-05-13", live_end_date="2026-05-20",
            output_dir=tmp_path / "reports",
        )

    data = json.loads(out.read_text())
    assert data["verdict"] == "ERROR"
    assert "boom" in data["stdout_tail"]


def _seed_reset_journal(db_path):
    """Mimic the deploy-day testnet reset: a pre-reset midnight snapshot and a
    post-reset top-up snapshot share the SAME daily cycle_id, plus a manual
    `deploy-*` cycle, followed by a slightly declining week. The buggy metric
    started from the 4594.78 pre-reset value and reported the +405 faucet
    top-up as +7.8% profit."""
    from tradingagents.execution.live.journal import Journal
    j = Journal(str(db_path))
    rows = [
        ("2026-05-31", "2026-05-31T00:06:45+00:00", 4594.78),   # pre-reset
        ("2026-05-31", "2026-05-31T07:45:51+00:00", 4999.91),   # post-reset top-up
        ("deploy-20260531", "2026-05-31T07:57:58+00:00", 4999.66),  # manual, exclude
        ("2026-06-01", "2026-06-01T00:07:58+00:00", 4993.59),
        ("2026-06-07", "2026-06-07T00:07:38+00:00", 4954.04),
    ]
    for cid, ts, val in rows:
        j._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, "
            "usdt_balance, position_qty_per_coin, unrealized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, ts, val, val, "{}", 0),
        )
    j._conn.commit()
    j.close()


def test_compute_live_metrics_ignores_deploy_day_reset(tmp_path, monkeypatch):
    """The clean baseline is the post-reset 4999.91, not the pre-reset 4594.78,
    so the week is -0.9% — not the spurious +7.8% the reset top-up produced."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    _seed_reset_journal(tmp_path / "trade_journal.db")
    from tradingagents.execution.live import rebacktest
    m = rebacktest.compute_live_metrics("2026-05-31", "2026-06-07")
    assert m["return_pct"] < 0
    assert m["return_pct"] == pytest.approx(-0.00917, abs=2e-3)


def test_compute_live_metrics_excludes_non_daily_cycles(tmp_path, monkeypatch):
    """Manual / non-scheduled cycle snapshots (cycle_id not YYYY-MM-DD) are not
    part of the equity curve; only one point per scheduled trading day counts."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from tradingagents.execution.live.journal import Journal
    j = Journal(str(tmp_path / "trade_journal.db"))
    rows = [
        ("2026-06-01", "2026-06-01T00:06:00+00:00", 5000.0),
        ("manual-xyz", "2026-06-01T12:00:00+00:00", 9999.0),   # blip, must NOT count
        ("2026-06-02", "2026-06-02T00:06:00+00:00", 5050.0),
    ]
    for cid, ts, val in rows:
        j._conn.execute(
            "INSERT INTO portfolio_snapshots (cycle_id, ts, total_value, "
            "usdt_balance, position_qty_per_coin, unrealized_pnl) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, ts, val, val, "{}", 0),
        )
    j._conn.commit()
    j.close()
    from tradingagents.execution.live import rebacktest
    m = rebacktest.compute_live_metrics("2026-06-01", "2026-06-02")
    assert m["n_trades"] == 2          # two daily points, blip excluded
    assert m["return_pct"] == pytest.approx(0.01, abs=1e-6)   # 5000 -> 5050
