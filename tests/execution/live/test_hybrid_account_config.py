# tests/execution/live/test_hybrid_account_config.py
import os
import pytest
from tradingagents.execution.live.hybrid_config import load_hybrid_account


def test_reads_hybrid_env(monkeypatch):
    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "hk")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "hs")
    monkeypatch.setenv("HYBRID_DATA_DIR", "/tmp/data-hybrid")
    monkeypatch.setenv("QUANT_DATA_DIR", "/tmp/data")
    acct = load_hybrid_account()
    assert acct.api_key == "hk" and acct.api_secret == "hs"
    assert acct.data_dir.endswith("data-hybrid")
    assert acct.quant_db_path.endswith("data/trade_journal.db")


def test_missing_hybrid_key_raises(monkeypatch):
    monkeypatch.delenv("HYBRID_BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("HYBRID_BINANCE_API_SECRET", raising=False)
    with pytest.raises(ValueError):
        load_hybrid_account()
