"""Dual-strategy monitor API tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tradingagents.monitor.app import create_app
from tradingagents.monitor.sources import StrategySource


def _quant_only_app(journal_path, log_dir, monkeypatch, snapshot=None):
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    def boom():
        raise RuntimeError("no creds")
    quant = StrategySource("quant", journal_path, snapshot or boom)
    return create_app(quant=quant, hybrid=None, log_dir=log_dir,
                      start_capital=10000.0)


def test_create_app_requires_password(journal_path, log_dir, monkeypatch):
    monkeypatch.delenv("TA_MONITOR_PASSWORD", raising=False)
    quant = StrategySource("quant", journal_path, lambda: {})
    with pytest.raises(RuntimeError):
        create_app(quant=quant, hybrid=None, log_dir=log_dir)


def test_all_routes_require_auth(dual_app):
    # Use a fresh client with NO auth set so the middleware returns 401.
    from fastapi.testclient import TestClient
    c = TestClient(dual_app, raise_server_exceptions=False)
    for path in ("/", "/api/performance", "/api/positions", "/api/trades",
                 "/api/cycles", "/api/health", "/api/compare"):
        assert c.get(path).status_code == 401, path


def test_performance_dual(dual_client):
    r = dual_client.get("/api/performance", auth=dual_client.auth)
    assert r.status_code == 200
    body = r.json()
    q, h = body["quant"], body["hybrid"]
    assert q["cards"]["equity"] == 10280.0          # last snapshot total_value
    assert q["cards"]["total_upnl"] == 50.0          # live snapshot
    assert q["cards"]["open_positions"] == 1
    assert len(q["equity"]) == 2 and len(q["drawdown"]) == 2
    assert q["rolling_sharpe"] == []                 # < 31 points
    assert h["cards"]["equity"] == 10100.0
    assert body["anchors"]["quant"] == 3.18
    assert "compare" in body                          # delta block present


def test_performance_hybrid_none(journal_path, log_dir, monkeypatch):
    app = _quant_only_app(journal_path, log_dir, monkeypatch)
    c = TestClient(app)
    body = c.get("/api/performance", auth=("admin", "pw")).json()
    assert body["hybrid"] is None
    assert body["compare"] is None
    # live snapshot failed -> uPnL falls back to journal snapshot value
    assert body["quant"]["cards"]["total_upnl"] == 80.0
    assert body["quant"]["cards"]["upnl_stale"] is True


def test_positions_dual(dual_client):
    body = dual_client.get("/api/positions", auth=dual_client.auth).json()
    q = body["quant"]
    assert q["positions"][0]["coin"] == "bitcoin"
    assert q["positions"][0]["upnl_usd"] == 50.0
    assert q["totals"]["upnl"] == 50.0 and q["totals"]["equity"] == 10350.0
    assert {"label": "USDT (free)", "usd": 7000.0} in q["allocation"]
    assert q["stale"] is False
    h = body["hybrid"]
    assert h["positions"][0]["coin"] == "ethereum"
    assert h["positions"][0]["upnl_pct"] == pytest.approx(100.0 / 3800.0 * 100, rel=1e-3)


def test_positions_fallback_when_live_fails(journal_path, log_dir, monkeypatch):
    app = _quant_only_app(journal_path, log_dir, monkeypatch)
    c = TestClient(app)
    q = c.get("/api/positions", auth=("admin", "pw")).json()["quant"]
    assert q["stale"] is True and "no creds" in q["error"]
    coins = {p["coin"] for p in q["positions"]}
    assert coins == {"bitcoin", "ethereum"}          # journal snapshot qty map
    assert q["as_of"] == "2026-05-20T07:05:00+00:00"


def test_trades_strategy_param_and_analytics(dual_client):
    body = dual_client.get("/api/trades?strategy=hybrid",
                           auth=dual_client.auth).json()
    assert len(body["executions"]) == 1
    assert body["executions"][0]["coin"] == "ethereum"
    assert body["analytics"]["slippage"] == {"mean": 0.4, "max": 0.4, "n": 1}
    assert body["analytics"]["income"] is None       # fake snapshot has no income
    quant = dual_client.get("/api/trades?strategy=quant",
                            auth=dual_client.auth).json()
    assert len(quant["executions"]) == 3


def test_trades_bad_strategy_400(dual_client):
    r = dual_client.get("/api/trades?strategy=nope", auth=dual_client.auth)
    assert r.status_code == 400


def test_cycles_and_cycle_detail_strategy(dual_client):
    cycles = dual_client.get("/api/cycles?strategy=hybrid",
                             auth=dual_client.auth).json()["cycles"]
    assert [c["cycle_id"] for c in cycles] == ["c2"]
    detail = dual_client.get("/api/cycle/c2?strategy=hybrid",
                             auth=dual_client.auth).json()
    assert detail["modulator"][0]["multiplier"] == 1.2
    quant_detail = dual_client.get("/api/cycle/c2?strategy=quant",
                                   auth=dual_client.auth).json()
    assert quant_detail["modulator"] == []
    assert len(quant_detail["predictions"]) == 2


def test_health_dual(dual_client):
    body = dual_client.get("/api/health", auth=dual_client.auth).json()
    assert body["timeline"]["quant"][0]["cycle_id"] == "c2"
    assert body["timeline"]["hybrid"][0]["cycle_id"] == "c2"
    assert body["steps"]                                # quant JSONL only
    assert body["errors"][0]["step"] == "execute"


def test_missing_db_returns_503(log_dir, monkeypatch):
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    quant = StrategySource("quant", "/nonexistent/x.db", lambda: {})
    app = create_app(quant=quant, hybrid=None, log_dir=log_dir)
    c = TestClient(app)
    r = c.get("/api/cycles", auth=("admin", "pw"))
    assert r.status_code == 503 and "error" in r.json()


def test_auth_non_ascii_header_is_401(dual_app):
    c = TestClient(dual_app, raise_server_exceptions=False)
    # Pass raw bytes so the HTTP layer doesn't ASCII-encode and reject first
    r = c.get("/api/performance", headers={"Authorization": b"Basic \xff\xfe"})
    assert r.status_code == 401


def test_auth_wrong_password_is_401(dual_app):
    c = TestClient(dual_app, raise_server_exceptions=False)
    r = c.get("/api/performance", auth=("admin", "wrong"))
    assert r.status_code == 401


def test_performance_isolates_missing_hybrid_db(journal_path, log_dir, monkeypatch):
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    quant = StrategySource("quant", journal_path, lambda: (_ for _ in ()).throw(RuntimeError("x")))
    hybrid = StrategySource("hybrid", "/nonexistent/h.db", lambda: {})
    app = create_app(quant=quant, hybrid=hybrid, log_dir=log_dir, start_capital=10000.0)
    c = TestClient(app)
    r = c.get("/api/performance", auth=("admin", "pw"))
    assert r.status_code == 200
    body = r.json()
    assert body["quant"]["cards"]["equity"] == 10280.0
    assert body["hybrid"] is None


def test_health_isolates_missing_hybrid_db(journal_path, log_dir, monkeypatch):
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    quant = StrategySource("quant", journal_path, lambda: {})
    hybrid = StrategySource("hybrid", "/nonexistent/h.db", lambda: {})
    app = create_app(quant=quant, hybrid=hybrid, log_dir=log_dir, start_capital=10000.0)
    c = TestClient(app)
    body = c.get("/api/health", auth=("admin", "pw")).json()
    assert body["timeline"]["quant"][0]["cycle_id"] == "c2"
    assert body["timeline"]["hybrid"] is None


def test_positions_isolates_missing_hybrid_db(journal_path, log_dir, monkeypatch):
    monkeypatch.setenv("TA_MONITOR_PASSWORD", "pw")
    def boom():
        raise RuntimeError("no creds")
    quant = StrategySource("quant", journal_path, boom)
    hybrid = StrategySource("hybrid", "/nonexistent/h.db", boom)
    app = create_app(quant=quant, hybrid=hybrid, log_dir=log_dir, start_capital=10000.0)
    c = TestClient(app)
    r = c.get("/api/positions", auth=("admin", "pw"))
    assert r.status_code == 200
    body = r.json()
    assert body["quant"]["stale"] is True            # journal fallback worked
    assert body["hybrid"] is None                     # isolated, not 503


def test_index_serves_react_dist(dual_client):
    import pathlib
    dist = (pathlib.Path(__file__).resolve().parents[2]
            / "tradingagents/monitor/frontend/dist")
    if not dist.is_dir():
        import pytest as _pytest
        _pytest.skip("frontend dist not built")
    r = dual_client.get("/", auth=dual_client.auth)
    assert r.status_code == 200
    assert '<div id="root">' in r.text
