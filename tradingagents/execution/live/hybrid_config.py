# tradingagents/execution/live/hybrid_config.py
"""Second-account + data-dir resolution for the hybrid cycle.

V5 sizing/risk knobs come from the shared config.load_config(); only the
Binance account credentials and the data dir are overridden so the hybrid
book is fully isolated from the quant book.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HybridAccount:
    api_key: str
    api_secret: str
    data_dir: str
    quant_db_path: str


def load_hybrid_account() -> HybridAccount:
    key = os.environ.get("HYBRID_BINANCE_API_KEY", "")
    secret = os.environ.get("HYBRID_BINANCE_API_SECRET", "")
    if not key or not secret:
        raise ValueError("HYBRID_BINANCE_API_KEY / _SECRET must be set")
    data_dir = os.environ.get("HYBRID_DATA_DIR", "data-hybrid")
    quant_data = os.environ.get("QUANT_DATA_DIR", os.environ.get("DATA_DIR", "data"))
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    return HybridAccount(
        api_key=key, api_secret=secret, data_dir=data_dir,
        quant_db_path=str(Path(quant_data) / "trade_journal.db"),
    )
