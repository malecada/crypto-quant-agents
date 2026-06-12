"""Per-strategy data sources for the dual (quant + hybrid) monitor.

A StrategySource bundles everything the API layer needs for one strategy:
its journal path and a TTL-cached live-account snapshot provider. Hybrid is
optional — resolve_sources() returns (quant, hybrid|None) from the same env
contract the runners use (QUANT_DATA_DIR / HYBRID_DATA_DIR /
HYBRID_BINANCE_API_KEY).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class StrategySource:
    name: str
    journal_path: str
    # () -> {"positions": [...], "usdt_free": float, "equity": float, "income": [...]|None}
    snapshot: Callable[[], dict]


def account_snapshot(ex) -> dict:
    """One live snapshot of an account: positions, free USDT, equity, and
    raw income records (None when the income endpoint fails — positions
    must still render)."""
    try:
        income = ex.income_history()
    except Exception:
        income = None
    return {
        "positions": ex.get_position_details(),
        "usdt_free": ex.get_balances().get("USDT", 0.0),
        "equity": ex.get_total_portfolio_value(),
        "income": income,
    }


def ttl_cached(fn: Callable[[], dict], ttl: float = 30.0,
               clock: Callable[[], float] = time.monotonic) -> Callable[[], dict]:
    """Cache fn() results AND failures for ttl seconds (same semantics as the
    old live_positions cache: retries during an IP ban must not re-query)."""
    state: dict = {"exp": 0.0, "data": None, "error": None}

    def wrapped() -> dict:
        now = clock()
        if now < state["exp"]:
            if state["error"] is not None:
                raise state["error"]
            return state["data"]
        try:
            data = fn()
            state.update(exp=now + ttl, data=data, error=None)
            return data
        except Exception as exc:
            state.update(exp=now + ttl, data=None, error=exc)
            raise

    return wrapped


def _exchange_provider(
    api_key_env: str,
    api_secret_env: str,
    testnet: bool | None = None,
) -> Callable[[], dict]:
    """Lazy ExchangeClient bound to one account's env credentials.

    Args:
        api_key_env: Environment variable name for the Binance API key.
        api_secret_env: Environment variable name for the Binance API secret.
        testnet: When True, force testnet=True on the ExchangeClient regardless
            of the quant runner's LIVE_MODE config. When None, the ExchangeClient
            falls back to its own config (follows the quant runner's live_mode).
            The hybrid runner always runs testnet=True, so callers constructing a
            hybrid provider must pass testnet=True to avoid a live-venue mismatch
            if the quant runner is ever promoted to LIVE_MODE=True.
    """
    holder: dict = {"client": None}

    def provide() -> dict:
        if not os.environ.get(api_key_env):
            raise RuntimeError(f"{api_key_env} not set — live account unavailable")
        if holder["client"] is None:
            from tradingagents.execution.exchange import ExchangeClient
            kwargs: dict = {
                "api_key": os.environ.get(api_key_env),
                "api_secret": os.environ.get(api_secret_env),
            }
            if testnet is not None:
                kwargs["testnet"] = testnet
            holder["client"] = ExchangeClient(**kwargs)
        return account_snapshot(holder["client"])

    return provide


def resolve_sources(ttl: float = 30.0) -> tuple[StrategySource, StrategySource | None]:
    """Build (quant, hybrid|None) from the runners' env contract.

    Hybrid is enabled only when HYBRID_DATA_DIR is set AND differs from the
    quant dir (mirrors the /api/compare guard).
    """
    quant_dir = Path(os.environ.get(
        "QUANT_DATA_DIR", os.environ.get("DATA_DIR", "data")))
    quant = StrategySource(
        name="quant",
        journal_path=str(quant_dir / "trade_journal.db"),
        snapshot=ttl_cached(
            _exchange_provider("BINANCE_API_KEY", "BINANCE_API_SECRET"), ttl),
    )
    hybrid_env = os.environ.get("HYBRID_DATA_DIR")
    if not hybrid_env or Path(hybrid_env) == quant_dir:
        return quant, None
    hybrid = StrategySource(
        name="hybrid",
        journal_path=str(Path(hybrid_env) / "trade_journal.db"),
        snapshot=ttl_cached(
            _exchange_provider("HYBRID_BINANCE_API_KEY",
                               "HYBRID_BINANCE_API_SECRET",
                               testnet=True), ttl),
    )
    return quant, hybrid
