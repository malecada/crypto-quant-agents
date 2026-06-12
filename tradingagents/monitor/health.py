"""Parse the live bot's structured JSONL logs for the System Health view.

The journal stores *what* happened (trades, sizing); these per-cycle JSONL
files capture *when* and *how long* each pipeline step ran.
"""
from __future__ import annotations

import json
from pathlib import Path


def read_structured_log(log_dir: str) -> list[dict]:
    """Parse the newest ``cycle_*.jsonl`` file in ``log_dir``.

    Returns one dict per step record, in file order. Returns an empty list
    if the directory or any matching file is missing. Malformed lines are
    skipped rather than raising.
    """
    d = Path(log_dir)
    if not d.is_dir():
        return []
    files = sorted(d.glob("cycle_*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        return []
    records: list[dict] = []
    with open(files[-1]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def recent_errors(records: list[dict]) -> list[dict]:
    """Step records whose status is not ``ok``."""
    return [r for r in records if r.get("status") != "ok"]
