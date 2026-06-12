"""Runner aborts cycle on critical data refresh failure."""
from __future__ import annotations

import pytest


def test_critical_data_fail_aborts_cycle(monkeypatch, tmp_path):
    from tradingagents.execution.live import runner, data_refresh
    from tradingagents.execution.live.data_refresh import CriticalDataRefreshError

    monkeypatch.setenv("COINGLASS_API_KEY", "test")
    monkeypatch.setenv("BINANCE_API_KEY", "x")
    monkeypatch.setenv("BINANCE_API_SECRET", "y")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CHECKPOINT_DIR", str(tmp_path / "ckpt"))

    def fake_refresh_all(cfg, log):
        raise CriticalDataRefreshError([("ohlcv", RuntimeError("API down"))])
    monkeypatch.setattr(data_refresh, "refresh_all", fake_refresh_all)

    result = runner.run_cycle(cycle_id="20260514-test", dry_run=True)
    assert result.status == "critical_data_fail"
    assert result.n_executed == 0
