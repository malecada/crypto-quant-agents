# tests/execution/live/test_hybrid_runner_skeleton.py
from pathlib import Path
from tradingagents.execution.live import hybrid_runner, halt
from tradingagents.execution.live.runner import CycleResult


def test_halt_short_circuits(tmp_path, monkeypatch):
    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(tmp_path))
    halt.write_halt("test", data_dir=tmp_path)
    res = hybrid_runner.run_hybrid_cycle(cycle_id="2026-06-11", dry_run=True)
    assert isinstance(res, CycleResult)
    assert res.status == "halted"
    assert (tmp_path / "last_cycle_heartbeat.txt").exists()
