"""R4 fix — persistent trading halt sentinel.

A KILL_SWITCH trip or operator --kill-all writes a HALT file under $DATA_DIR;
run_cycle refuses to trade while it exists, so a tripped halt does not silently
auto-resume on the next systemd timer fire. The operator clears it with
--resume after investigating.
"""


def test_write_and_detect_halt(tmp_path):
    from tradingagents.execution.live import halt

    assert halt.is_halted(data_dir=tmp_path) is False
    halt.write_halt("daily PnL -16% — KILL SWITCH", data_dir=tmp_path)
    assert halt.is_halted(data_dir=tmp_path) is True
    assert "KILL SWITCH" in halt.halt_reason(data_dir=tmp_path)


def test_clear_halt_is_idempotent(tmp_path):
    from tradingagents.execution.live import halt

    halt.write_halt("x", data_dir=tmp_path)
    assert halt.clear_halt(data_dir=tmp_path) is True
    assert halt.is_halted(data_dir=tmp_path) is False
    assert halt.clear_halt(data_dir=tmp_path) is False


def test_halt_path_defaults_to_data_dir_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    from tradingagents.execution.live import halt

    assert halt.halt_path() == tmp_path / "HALT"
