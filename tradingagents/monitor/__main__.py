"""Entrypoint: ``python -m tradingagents.monitor``.

Reads QUANT_DATA_DIR|DATA_DIR, HYBRID_DATA_DIR, LOG_DIR, TA_MONITOR_PASSWORD,
TA_MONITOR_START_CAPITAL from the environment (same env contract as the
runners). Binds 127.0.0.1 only — a reverse proxy terminates TLS in production.
"""
from __future__ import annotations

import os

import uvicorn

from tradingagents.monitor.app import create_app
from tradingagents.monitor.sources import resolve_sources


def main() -> None:
    quant, hybrid = resolve_sources()
    app = create_app(
        quant=quant,
        hybrid=hybrid,
        log_dir=os.environ.get("LOG_DIR", "logs"),
        start_capital=float(os.environ.get("TA_MONITOR_START_CAPITAL", "10000")),
    )
    uvicorn.run(app, host=os.environ.get("TA_MONITOR_HOST", "127.0.0.1"),
                port=int(os.environ.get("TA_MONITOR_PORT", "8800")))


if __name__ == "__main__":
    main()
