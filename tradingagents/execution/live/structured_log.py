"""Append-only JSONL structured log for forensic reconstruction.

Each cycle writes a per-cycle JSONL file with one record per pipeline step.
Used as a complement to the SQLite journal: the journal stores the *what*
(trades, sizing, risk checks) while the structured log captures the *when*
and *how-long* (per-step timing + error context) for debugging.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class StructuredLogger:
    """Append-only JSONL emitter for cycle pipeline events."""

    def __init__(self, path: Path, cycle_id: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cycle_id = cycle_id

    def event(self, step: str, status: str, payload: dict | None = None,
              duration_ms: int | None = None) -> None:
        """Write one JSONL record describing a pipeline event."""
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle_id": self.cycle_id,
            "step": step,
            "status": status,
            "duration_ms": duration_ms,
            "payload": payload or {},
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def info(self, event: str, **payload) -> None:
        """Emit an informational JSONL record (used by data_refresh.refresh_all)."""
        self.event(event, "info", payload or None)

    def warn(self, event: str, **payload) -> None:
        """Emit a warning JSONL record (used by data_refresh.refresh_all)."""
        self.event(event, "warn", payload or None)

    @contextmanager
    def step(self, step: str, payload: dict | None = None):
        """Context manager that emits ok/error events bracketing a block."""
        start = time.monotonic()
        try:
            yield
            self.event(step, "ok", payload, int((time.monotonic() - start) * 1000))
        except Exception as e:
            self.event(
                step, "error",
                {"error": str(e), **(payload or {})},
                int((time.monotonic() - start) * 1000),
            )
            raise
