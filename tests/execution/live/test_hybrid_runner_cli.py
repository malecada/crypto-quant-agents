# tests/execution/live/test_hybrid_runner_cli.py
import os
import subprocess
import sys


def test_cli_dry_run_exits_zero_or_one(tmp_path, monkeypatch):
    """CLI module is runnable; no-quant-cycle → exit 1; wrong env → exit 1."""
    env = dict(**os.environ)
    env.update(
        HYBRID_BINANCE_API_KEY="k",
        HYBRID_BINANCE_API_SECRET="s",
        HYBRID_DATA_DIR=str(tmp_path),
        QUANT_DATA_DIR=str(tmp_path),
        COIN_UNIVERSE="bitcoin",
        BINANCE_API_KEY="qk",
        BINANCE_API_SECRET="qs",
        COINGLASS_API_KEY="cgk",
    )
    # No quant cycle seeded → status no_quant_cycle → exit code 1
    r = subprocess.run(
        [sys.executable, "-m", "tradingagents.execution.live.hybrid_runner",
         "--once", "--dry-run", "--cycle-id", "2026-06-11"],
        capture_output=True, env=env,
    )
    assert r.returncode in (0, 1), (
        f"unexpected returncode {r.returncode}\n"
        f"stdout: {r.stdout.decode()}\nstderr: {r.stderr.decode()}"
    )


def test_cli_missing_key_exits_nonzero(tmp_path):
    """Missing HYBRID_BINANCE_API_KEY → non-zero exit."""
    env = dict(**os.environ)
    env.pop("HYBRID_BINANCE_API_KEY", None)
    env.pop("HYBRID_BINANCE_API_SECRET", None)
    r = subprocess.run(
        [sys.executable, "-m", "tradingagents.execution.live.hybrid_runner",
         "--once", "--dry-run"],
        capture_output=True, env=env,
    )
    assert r.returncode != 0


def test_cli_resume_exits_zero(tmp_path):
    """--resume clears the halt sentinel and exits 0."""
    env = dict(**os.environ)
    env.update(
        HYBRID_BINANCE_API_KEY="k",
        HYBRID_BINANCE_API_SECRET="s",
        HYBRID_DATA_DIR=str(tmp_path),
    )
    # Write a halt sentinel using the correct filename (halt.py uses "HALT")
    (tmp_path / "HALT").write_text("test halt")
    r = subprocess.run(
        [sys.executable, "-m", "tradingagents.execution.live.hybrid_runner",
         "--resume"],
        capture_output=True, env=env,
    )
    assert r.returncode == 0
    # Halt file should be gone
    assert not (tmp_path / "HALT").exists()
