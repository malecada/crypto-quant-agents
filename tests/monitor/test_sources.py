"""StrategySource resolution from env + TTL-cached account snapshots."""
from __future__ import annotations

import pytest

from tradingagents.monitor import sources


def test_resolve_quant_only(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("QUANT_DATA_DIR", raising=False)
    monkeypatch.delenv("HYBRID_DATA_DIR", raising=False)
    quant, hybrid = sources.resolve_sources()
    assert quant.name == "quant"
    assert quant.journal_path == str(tmp_path / "trade_journal.db")
    assert hybrid is None


def test_resolve_both(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path / "q"))
    monkeypatch.setenv("HYBRID_DATA_DIR", str(tmp_path / "h"))
    quant, hybrid = sources.resolve_sources()
    assert hybrid is not None and hybrid.name == "hybrid"
    assert hybrid.journal_path == str(tmp_path / "h" / "trade_journal.db")


def test_resolve_hybrid_equal_dirs_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HYBRID_DATA_DIR", str(tmp_path))
    _, hybrid = sources.resolve_sources()
    assert hybrid is None


def test_cached_provider_caches_success_and_failure():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            return {"ok": True}
        raise RuntimeError("ban")

    t = {"now": 0.0}
    cached = sources.ttl_cached(flaky, ttl=30.0, clock=lambda: t["now"])
    assert cached() == {"ok": True}
    assert cached() == {"ok": True} and calls["n"] == 1  # cached
    t["now"] = 31.0
    with pytest.raises(RuntimeError):
        cached()
    with pytest.raises(RuntimeError):  # failure cached too
        cached()
    assert calls["n"] == 2


def test_account_snapshot_shape(monkeypatch):
    class FakeEx:
        def get_position_details(self):
            return [{"symbol": "BTCUSDT", "qty": 0.05, "entry_price": 65000.0,
                     "mark_price": 66000.0, "upnl": 50.0, "leverage": 3.0,
                     "liq_price": 30000.0, "notional": 3300.0}]

        def get_balances(self):
            return {"USDT": 7000.0}

        def get_total_portfolio_value(self):
            return 10350.0

        def income_history(self, **kw):
            return [{"incomeType": "REALIZED_PNL", "income": "5", "symbol": "BTCUSDT"}]

    snap = sources.account_snapshot(FakeEx())
    assert snap["equity"] == 10350.0
    assert snap["usdt_free"] == 7000.0
    assert snap["positions"][0]["symbol"] == "BTCUSDT"
    assert snap["income"][0]["income"] == "5"


def test_account_snapshot_income_failure_is_none():
    class FakeEx:
        def get_position_details(self):
            return []

        def get_balances(self):
            return {"USDT": 1.0}

        def get_total_portfolio_value(self):
            return 1.0

        def income_history(self, **kw):
            raise RuntimeError("weight limit")

    assert sources.account_snapshot(FakeEx())["income"] is None
