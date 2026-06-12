"""Persistent trading-halt sentinel (R4).

A KILL_SWITCH trip (daily-loss / drawdown breach) or an operator `--kill-all`
writes a ``HALT`` file under ``$DATA_DIR``. ``run_cycle`` refuses to trade while
it exists, so a tripped halt survives process restarts and the systemd timer
instead of silently auto-resuming on the next cycle. The operator clears it
explicitly with ``--resume`` after investigating.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def halt_path(data_dir: str | os.PathLike | None = None) -> Path:
    root = Path(data_dir) if data_dir is not None else Path(os.environ.get("DATA_DIR", "data"))
    return root / "HALT"


def write_halt(reason: str, *, data_dir: str | os.PathLike | None = None) -> Path:
    """Create the halt sentinel, stamping the UTC time and reason."""
    p = halt_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    p.write_text(f"{ts}\n{reason}\n")
    return p


def is_halted(*, data_dir: str | os.PathLike | None = None) -> bool:
    return halt_path(data_dir).exists()


def halt_reason(*, data_dir: str | os.PathLike | None = None) -> str:
    p = halt_path(data_dir)
    return p.read_text().strip() if p.exists() else ""


def clear_halt(*, data_dir: str | os.PathLike | None = None) -> bool:
    """Remove the sentinel. Returns True if one existed, False otherwise."""
    p = halt_path(data_dir)
    if p.exists():
        p.unlink()
        return True
    return False
