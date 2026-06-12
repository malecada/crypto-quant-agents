"""FastAPI app for the dual-strategy (quant + hybrid) live bot monitor.

Read-only. Serves the built React SPA at ``/`` and JSON at ``/api/*``.
HTTP basic auth is enforced by middleware on EVERY path (including static
assets). All endpoints tolerate an empty or missing journal: empty DBs
yield empty payloads, an unreadable DB yields HTTP 503 on per-strategy
endpoints (/api/cycles, /api/cycle, /api/trades). Combined endpoints
(/api/performance, /api/health) degrade per-strategy instead — a locked
or missing journal for one source yields ``null`` for that strategy and
continues serving the other. A missing hybrid source yields
``hybrid: null`` blocks, never an error.
"""
from __future__ import annotations

import base64
import json
import math
import os
import secrets
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from tradingagents.execution.live.config import from_binance_symbol
from tradingagents.execution.live.rebacktest import compare_quant_hybrid
from tradingagents.monitor import analytics, db, health, metrics
from tradingagents.monitor.sources import StrategySource

_DIR = Path(__file__).parent
_DIST = _DIR / "frontend" / "dist"
_AUTH_USER = "admin"
_ROLLING_WINDOW = 30


def _sanitize_floats(obj):
    """Recursively replace NaN/±Inf with None so FastAPI can JSON-serialize."""
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj


def create_app(
    *,
    quant: StrategySource,
    hybrid: StrategySource | None = None,
    log_dir: str = "logs",
    start_capital: float = 10000.0,
) -> FastAPI:
    """Build the monitor app. Raises RuntimeError if TA_MONITOR_PASSWORD
    is unset — the UI must never run without a password."""
    password = os.environ.get("TA_MONITOR_PASSWORD", "")
    if not password:
        raise RuntimeError("TA_MONITOR_PASSWORD environment variable is not set")

    app = FastAPI(title="Live Monitor", docs_url=None, redoc_url=None)

    # ── parse anchor env vars once at startup (fail fast on bad values) ────
    _anchor_quant = float(os.environ.get("TA_MONITOR_ANCHOR_SR_QUANT", "3.18"))
    _anchor_hybrid_env = os.environ.get("TA_MONITOR_ANCHOR_SR_HYBRID")
    _anchor_hybrid = float(_anchor_hybrid_env) if _anchor_hybrid_env else None

    # ── auth middleware (covers /api AND static SPA assets) ────────────────
    expected = base64.b64encode(f"{_AUTH_USER}:{password}".encode())

    @app.middleware("http")
    async def basic_auth(request: Request, call_next):
        header = request.headers.get("authorization", "")
        ok = header.startswith("Basic ") and secrets.compare_digest(
            header[6:].encode("latin-1", "replace"), expected)
        if not ok:
            return Response(status_code=401,
                            headers={"WWW-Authenticate": "Basic"})
        return await call_next(request)

    # ── helpers ────────────────────────────────────────────────────────────
    def _sources() -> list[StrategySource]:
        return [quant] + ([hybrid] if hybrid else [])

    def _source(name: str) -> StrategySource:
        for s in _sources():
            if s.name == name:
                return s
        raise HTTPException(status_code=400, detail=f"unknown strategy {name!r}")

    def _conn(s: StrategySource) -> sqlite3.Connection:
        return db.open_journal(s.journal_path)

    def _snapshot_rows(conn: sqlite3.Connection) -> tuple:
        """(portfolio_snapshots, latest ref_price per coin)."""
        snaps = db.portfolio_snapshots(conn)
        ref_prices: dict = {}
        latest = db.latest_cycle(conn)
        if latest:
            for p in db.cycle_detail(conn, latest["cycle_id"])["predictions"]:
                if p.get("ref_price") is not None:
                    ref_prices[p["coin"]] = p["ref_price"]
        return snaps, ref_prices

    def _live_block(s: StrategySource) -> tuple:
        """(snapshot|None, error|None) from the TTL-cached provider."""
        try:
            return s.snapshot(), None
        except Exception as exc:
            return None, str(exc)

    # ── endpoints ──────────────────────────────────────────────────────────
    @app.get("/api/performance")
    def api_performance():
        out: dict = {"quant": None, "hybrid": None}
        for s in _sources():
            try:
                conn = _conn(s)
            except sqlite3.OperationalError:
                out[s.name] = None
                continue
            try:
                snaps, _ = _snapshot_rows(conn)
                trades = db.all_trades(conn)
            finally:
                conn.close()
            equity = metrics.equity_series(snaps, trades, start_capital)
            values = [pt["value"] for pt in equity]
            live, _live_err = _live_block(s)
            if live is not None:
                total_upnl = sum(p["upnl"] for p in live["positions"])
                n_open = len(live["positions"])
                upnl_stale = False
            else:
                total_upnl = snaps[-1].get("unrealized_pnl") if snaps else None
                n_open = None
                upnl_stale = True
            out[s.name] = {
                "cards": {
                    "equity": values[-1] if values else start_capital,
                    "sharpe": round(metrics.sharpe(values), 2),
                    "max_drawdown": round(metrics.max_drawdown(values), 4),
                    "total_upnl": total_upnl,
                    "upnl_stale": upnl_stale,
                    "open_positions": n_open,
                },
                "equity": equity,
                "drawdown": metrics.drawdown_series(equity),
                "rolling_sharpe": metrics.rolling_sharpe(equity, _ROLLING_WINDOW),
            }
        compare = None
        if hybrid is not None:
            try:
                compare = _sanitize_floats(compare_quant_hybrid(
                    Path(quant.journal_path), Path(hybrid.journal_path), coins=[]))
            except Exception as exc:
                compare = {"error": str(exc)}
        out["compare"] = compare
        out["anchors"] = {
            "quant": _anchor_quant,
            "hybrid": _anchor_hybrid,
        }
        return out

    @app.get("/api/positions")
    def api_positions():
        out: dict = {"quant": None, "hybrid": None}
        for s in _sources():
            live, live_err = _live_block(s)
            if live is not None:
                positions = []
                for p in sorted(live["positions"], key=lambda x: x["symbol"]):
                    entry_notional = abs(p["qty"]) * p["entry_price"]
                    positions.append({
                        "coin": from_binance_symbol(p["symbol"]),
                        "side": "LONG" if p["qty"] > 0 else "SHORT",
                        "qty": p["qty"],
                        "entry": p["entry_price"],
                        "mark": p["mark_price"],
                        "leverage": p["leverage"],
                        "notional": p["notional"],
                        "upnl_usd": p["upnl"],
                        "upnl_pct": (p["upnl"] / entry_notional * 100.0
                                     if entry_notional else None),
                        "liq_price": p["liq_price"] or None,
                    })
                allocation = [{"label": pos["coin"], "usd": abs(pos["notional"])}
                              for pos in positions]
                allocation.append({"label": "USDT (free)", "usd": live["usdt_free"]})
                out[s.name] = {
                    "positions": positions,
                    "totals": {
                        "upnl": sum(p["upnl_usd"] for p in positions),
                        "notional": sum(abs(p["notional"]) for p in positions),
                        "equity": live["equity"],
                    },
                    "allocation": allocation,
                    "stale": False, "as_of": None, "error": None,
                }
            else:  # journal fallback (same pattern as v2.3.1 holdings fix)
                try:
                    conn = _conn(s)
                except sqlite3.OperationalError:
                    out[s.name] = None
                    continue
                try:
                    snaps, ref_prices = _snapshot_rows(conn)
                finally:
                    conn.close()
                positions = []
                as_of = None
                if snaps:
                    as_of = snaps[-1].get("ts")
                    try:
                        qty_map = json.loads(
                            snaps[-1].get("position_qty_per_coin") or "{}")
                    except (json.JSONDecodeError, TypeError):
                        qty_map = {}
                    for coin, qty in sorted(qty_map.items()):
                        if not qty:
                            continue
                        price = ref_prices.get(coin)
                        positions.append({
                            "coin": coin,
                            "side": "LONG" if qty > 0 else "SHORT",
                            "qty": qty, "entry": None, "mark": price,
                            "leverage": None,
                            "notional": qty * price if price else None,
                            "upnl_usd": None, "upnl_pct": None,
                            "liq_price": None,
                        })
                allocation = [{"label": p["coin"], "usd": abs(p["notional"])}
                              for p in positions if p["notional"] is not None]
                out[s.name] = {
                    "positions": positions,
                    "totals": {
                        "upnl": snaps[-1].get("unrealized_pnl") if snaps else None,
                        "notional": sum(a["usd"] for a in allocation) or None,
                        "equity": snaps[-1].get("total_value") if snaps else None,
                    },
                    "allocation": allocation,
                    "stale": True, "as_of": as_of, "error": live_err,
                }
        return out

    @app.get("/api/trades")
    def api_trades(strategy: str = "quant"):
        s = _source(strategy)
        conn = _conn(s)
        try:
            executions = db.all_trades(conn)
        finally:
            conn.close()
        income_block = None
        live, _err = _live_block(s)
        if live is not None and live.get("income") is not None:
            income_block = analytics.income_summary(live["income"])
        return {
            "executions": executions,
            "analytics": {
                "income": income_block,
                "slippage": analytics.slippage_stats(executions),
            },
        }

    @app.get("/api/cycles")
    def api_cycles(strategy: str = "quant"):
        conn = _conn(_source(strategy))
        try:
            return {"cycles": db.list_cycles(conn)}
        finally:
            conn.close()

    @app.get("/api/cycle/{cycle_id}")
    def api_cycle(cycle_id: str, strategy: str = "quant"):
        conn = _conn(_source(strategy))
        try:
            detail = db.cycle_detail(conn, cycle_id)
            detail["modulator"] = db.modulator_outputs(conn, cycle_id)
            return detail
        finally:
            conn.close()

    @app.get("/api/health")
    def api_health():
        timeline: dict = {}
        retrains: dict = {}
        for s in _sources():
            try:
                conn = _conn(s)
            except sqlite3.OperationalError:
                timeline[s.name] = None
                retrains[s.name] = None
                continue
            try:
                timeline[s.name] = db.list_cycles(conn)
                retrains[s.name] = db.retrains(conn)
            finally:
                conn.close()
        if hybrid is None:
            timeline.setdefault("hybrid", None)
            retrains.setdefault("hybrid", None)
        steps = health.read_structured_log(log_dir)  # quant runner only
        return {
            "timeline": timeline,
            "steps": steps,
            "errors": health.recent_errors(steps),
            "retrains": retrains,
        }

    @app.get("/api/compare")
    def api_compare():
        if hybrid is None:
            return {"error": "hybrid not configured — HYBRID_DATA_DIR not set "
                             "or equals QUANT_DATA_DIR"}
        coins_env = os.environ.get("COMPARE_COINS", "")
        coins = [c.strip() for c in coins_env.split(",") if c.strip()]
        try:
            return _sanitize_floats(compare_quant_hybrid(
                Path(quant.journal_path), Path(hybrid.journal_path), coins=coins))
        except Exception as exc:
            return {"error": str(exc)}

    @app.exception_handler(sqlite3.OperationalError)
    def _db_error(request: Request, exc: sqlite3.OperationalError):
        return JSONResponse(status_code=503, content={"error": str(exc)})

    # ── React SPA (built dist committed to repo) ───────────────────────────
    if _DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")),
                  name="assets")

        @app.get("/")
        def index():
            return FileResponse(str(_DIST / "index.html"))
    else:  # pre-build / CI without dist: explicit 503, not a silent 404
        @app.get("/")
        def index_missing():
            return JSONResponse(status_code=503, content={
                "error": "frontend not built — run npm run build in "
                         "tradingagents/monitor/frontend"})

    return app
