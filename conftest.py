"""Ensure pytest imports the worktree's tradingagents package.

The editable install (`pip install -e .`) points at the primary repo, so
without this shim pytest run from a worktree would import the primary
repo's package and miss any local changes (e.g. new submodules added on a
feature branch).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
