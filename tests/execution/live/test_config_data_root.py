"""P5 — single data-root source of truth.

On the VPS, TRADINGAGENTS_DATA_ROOT was unset (config default "data" -> relative
repo/data, where data_refresh + retrain WRITE) while the systemd unit set
DATA_DIR=/opt/tradingagents/data (where the runner READS the OHLCV cache + the
journal). The two diverged: the read-side cache froze while the write-side cache
updated daily. Now data_root falls back to DATA_DIR when TRADINGAGENTS_DATA_ROOT
is unset, and load_config hard-fails if both are set and disagree.
"""
import pytest


def _base_env(monkeypatch):
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("COINGLASS_API_KEY", "c")
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin,ethereum,binancecoin,solana")


def test_data_root_falls_back_to_data_dir(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("TRADINGAGENTS_DATA_ROOT", raising=False)
    monkeypatch.setenv("DATA_DIR", "/opt/tradingagents/data")
    from tradingagents.execution.live.config import load_config
    assert load_config().data_root == "/opt/tradingagents/data"


def test_data_root_explicit_takes_precedence_when_equal(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", "/opt/tradingagents/data")
    monkeypatch.setenv("TRADINGAGENTS_DATA_ROOT", "/opt/tradingagents/data")
    from tradingagents.execution.live.config import load_config
    assert load_config().data_root == "/opt/tradingagents/data"


def test_data_root_conflict_raises(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("DATA_DIR", "/opt/tradingagents/data")
    monkeypatch.setenv("TRADINGAGENTS_DATA_ROOT", "data")  # relative, diverges
    from tradingagents.execution.live.config import load_config
    with pytest.raises(ValueError, match="(?i)data_root|DATA_DIR"):
        load_config()


def test_data_root_defaults_to_data_when_neither_set(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("TRADINGAGENTS_DATA_ROOT", raising=False)
    monkeypatch.delenv("DATA_DIR", raising=False)
    from tradingagents.execution.live.config import load_config
    assert load_config().data_root == "data"
