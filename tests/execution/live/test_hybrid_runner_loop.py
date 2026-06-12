# tests/execution/live/test_hybrid_runner_loop.py
import sqlite3
import numpy as np
import pandas as pd
import pytest
from tradingagents.execution.live import hybrid_runner
from tradingagents.execution.live.journal import Journal


class FakeExchange:
    def __init__(self):
        self.orders = []
        self.stops = []

    def set_leverage(self, *a, **k):
        pass

    def get_total_portfolio_value(self):
        return 10000.0

    def get_usdt_balance(self):
        return 10000.0

    def get_current_position(self, symbol):
        return 0.0

    def round_quantity(self, symbol, q):
        return round(q, 3)

    def min_notional(self, symbol):
        return 5.0

    def get_ticker_price(self, symbol):
        return 65000.0

    def place_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append((symbol, side, qty))
        return {"orderId": 1, "status": "FILLED"}

    def cancel_all_orders(self, symbol):
        return []

    def list_open_stops(self, symbol):
        return []

    def place_stop_loss(self, symbol, qty, stop_price, stop_side):
        stop_id = len(self.stops) + 100
        self.stops.append((symbol, qty, stop_price, stop_side))
        return {"orderId": stop_id}

    def cancel_order(self, symbol, order_id):
        pass


class StubGraph:
    """Returns a fixed modulator output: mult=1.4, eff_w=0.5.
    Importantly, position=-999.0 to assert it is DISCARDED."""

    def propagate_with_modulator(self, coin, date):
        mp = {
            "coin": coin,
            "llm_multiplier": 1.4,
            "effective_weight": 0.5,
            "position": -999.0,   # must be discarded; composition uses base×formula
            "regime": "bull",
            "llm_uncertainty": 0.1,
        }
        return ({}, mp, {"coin": coin, "direction": "long", "magnitude": 0.2}, "ok")


def _seed_quant_db(db_path: str, cycle_id: str) -> None:
    j = Journal(db_path)
    j.log_cycle_start(cycle_id, git_sha="x")
    preds_df = pd.DataFrame([
        {"coin": "bitcoin", "horizon": 7,  "prediction": 0.03,
         "ref_price": 65000.0, "bundle_route": "bitcoin_78f"},
        {"coin": "bitcoin", "horizon": 14, "prediction": 0.05,
         "ref_price": 65000.0, "bundle_route": "bitcoin_78f"},
    ])
    j.record_predictions(cycle_id=cycle_id, preds_df=preds_df)
    j.close()


def _seed_ohlcv_cache(data_dir, symbol: str) -> None:
    """Write 60-bar parquet to <data_dir>/ohlcv_cache/<symbol>_1d.parquet
    with lowercase columns: date, open, high, low, close, volume.
    60 bars ensures vol_lookback=20 + trend_sma=30 have sufficient history.
    The hybrid runner reads OHLCV from the QUANT data dir (populated by the
    quant cycle); pass quant_dir here, not hybrid_dir.
    """
    cache_dir = data_dir / "ohlcv_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2026-03-01", periods=60, freq="D")
    px = pd.Series(100.0 + np.arange(60) * 0.5)
    df = pd.DataFrame({
        "date": idx,
        "open": px,
        "high": px * 1.01,
        "low": px * 0.99,
        "close": px,
        "volume": 1000.0,
    })
    df.to_parquet(cache_dir / f"{symbol}_1d.parquet", index=False)


def test_loop_composes_and_executes_on_hybrid_only(tmp_path, monkeypatch):
    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"

    _seed_quant_db(str(quant_dir / "trade_journal.db"), "2026-06-11")

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    # Provide required quant config vars so config.load_config() succeeds
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    fake_ex = FakeExchange()
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=StubGraph(),
    )
    assert res.status == "ok", f"cycle returned {res.status}: {res.error_msg}"

    # An order was placed on the hybrid (fake) exchange
    assert len(fake_ex.orders) >= 1, "expected at least one order on the hybrid exchange"

    # Hybrid journal got a trade row
    hybrid_db = str(hybrid_dir / "trade_journal.db")
    hyb_conn = sqlite3.connect(hybrid_db)
    h_trades = hyb_conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    hyb_conn.close()
    assert h_trades >= 1, f"hybrid journal has {h_trades} trades (expected ≥ 1)"

    # Quant journal is UNTOUCHED (0 trades written from this cycle)
    qn = sqlite3.connect(str(quant_dir / "trade_journal.db")).execute(
        "SELECT COUNT(*) FROM trades"
    ).fetchone()[0]
    assert qn == 0, f"quant journal has {qn} trades — isolation breached!"


def test_loop_compose_uses_mult_not_position(tmp_path, monkeypatch):
    """Verify compose uses (mult, eff_w) from StubGraph, not mp['position']=-999."""
    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"

    _seed_quant_db(str(quant_dir / "trade_journal.db"), "2026-06-11")

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    # StubGraph returns position=-999 but the hybrid should use mult=1.4, eff_w=0.5
    fake_ex = FakeExchange()
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=StubGraph(),
    )
    assert res.status == "ok"
    # If position=-999 were used, the order quantity would be enormous/negative;
    # it's bounded to exchange round_quantity (a small positive number).
    for sym, side, qty in fake_ex.orders:
        assert qty > 0, f"negative qty {qty} for {sym}/{side} — position=-999 leaked"
        assert qty < 10.0, f"huge qty {qty} for {sym}/{side} — position=-999 leaked"


def test_modulator_failure_degrades_to_pure_quant(tmp_path, monkeypatch):
    """A crashing modulator must not block the cycle — pure-quant fallback."""
    class CrashGraph:
        def propagate_with_modulator(self, coin, date):
            raise RuntimeError("intentional modulator crash")

    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"

    _seed_quant_db(str(quant_dir / "trade_journal.db"), "2026-06-11")

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    fake_ex = FakeExchange()
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=CrashGraph(),
    )
    # Cycle must succeed (pure quant fallback, not crash)
    assert res.status == "ok", f"cycle failed with modulator crash: {res.error_msg}"
    # Still executed (pure-quant base is non-zero for a long signal)
    assert len(fake_ex.orders) >= 1


# ---------------------------------------------------------------------------
# FIX 1: portfolio-level daily-loss + drawdown kill-switch
# ---------------------------------------------------------------------------

class FakeExchangeLow:
    """Exchange whose portfolio value is 5 000 — 50 % below the seeded peak."""

    def __init__(self):
        self.orders = []

    def get_total_portfolio_value(self):
        return 5_000.0          # well below the 20 000 peak seeded in journal

    def get_usdt_balance(self):
        return 5_000.0

    def get_current_position(self, symbol):
        return 0.0

    def round_quantity(self, symbol, q):
        return round(q, 3)

    def min_notional(self, symbol):
        return 5.0

    def get_ticker_price(self, symbol):
        return 65_000.0

    def place_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append((symbol, side, qty, reduce_only))
        return {"orderId": 1, "status": "FILLED"}

    def cancel_all_orders(self, symbol):
        return []

    def list_open_stops(self, symbol):
        return []

    def place_stop_loss(self, symbol, qty, stop_price, stop_side):
        return {"orderId": 999}

    def cancel_order(self, symbol, order_id):
        pass


def _seed_quant_db_for_kill(db_path: str, cycle_id: str) -> None:
    j = Journal(db_path)
    j.log_cycle_start(cycle_id, git_sha="x")
    preds_df = pd.DataFrame([
        {"coin": "bitcoin", "horizon": 7,  "prediction": 0.03,
         "ref_price": 65_000.0, "bundle_route": "bitcoin_78f"},
        {"coin": "bitcoin", "horizon": 14, "prediction": 0.05,
         "ref_price": 65_000.0, "bundle_route": "bitcoin_78f"},
    ])
    j.record_predictions(cycle_id=cycle_id, preds_df=preds_df)
    j.close()


def _seed_hybrid_peak(hybrid_db_path: str, peak_value: float) -> None:
    """Pre-seed the hybrid journal with a portfolio_snapshot at *peak_value*
    so peak_total_value() returns a high number, triggering a drawdown kill."""
    j = Journal(hybrid_db_path)
    j.log_cycle_start("2026-01-01", git_sha="seed")
    j.log_portfolio_snapshot(
        cycle_id="2026-01-01",
        total_value=peak_value,
        usdt_balance=peak_value,
        position_qty_per_coin={},
        unrealized_pnl=0.0,
    )
    j.log_cycle_end("2026-01-01", status="ok")
    j.close()


def test_drawdown_kill_halts_and_places_no_orders(tmp_path, monkeypatch):
    """Hybrid cycle with a large drawdown fires the kill-switch and places NO orders."""
    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"
    hybrid_dir.mkdir()

    _seed_quant_db_for_kill(str(quant_dir / "trade_journal.db"), "2026-06-11")
    # Seed hybrid journal with a peak of 20 000; current portfolio = 5 000 → 75 % DD
    _seed_hybrid_peak(str(hybrid_dir / "trade_journal.db"), peak_value=20_000.0)

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    fake_ex = FakeExchangeLow()
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=StubGraph(),
    )

    # Cycle must report a halt/risk_halt status
    assert res.status in ("halted", "risk_halt", "error"), (
        f"expected a halt/risk_halt but got {res.status}: {res.error_msg}"
    )
    # NO trade orders should have been placed
    assert len(fake_ex.orders) == 0, (
        f"kill-switch fired but {len(fake_ex.orders)} orders were placed"
    )
    # Halt sentinel must be written so the NEXT cycle also refuses to trade
    from tradingagents.execution.live import halt as halt_mod
    assert halt_mod.is_halted(data_dir=hybrid_dir), (
        "kill-switch fired but halt sentinel was not written"
    )


# ---------------------------------------------------------------------------
# FIX 2: log_cycle_end called on ALL paths (including exception)
# ---------------------------------------------------------------------------

class ExchangeThatExplodes(FakeExchange):
    """Exchange whose place_market_order raises to simulate an unexpected error
    that escapes the per-coin loop (not the modulator try/except)."""
    def place_market_order(self, symbol, side, qty, reduce_only=False):
        raise RuntimeError("boom — unexpected exchange error outside modulator")


def test_cycle_end_logged_on_exception(tmp_path, monkeypatch):
    """On an exception inside the cycle that escapes per-coin handling,
    the `cycles` table row must have a non-NULL status (not left open-ended)."""
    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"

    _seed_quant_db(str(quant_dir / "trade_journal.db"), "2026-06-11")

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    fake_ex = ExchangeThatExplodes()
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=StubGraph(),
    )
    # Run returns an error result (not raised)
    assert res.status == "error", f"unexpected status: {res.status}"

    # The cycle row in the hybrid journal must have status NOT NULL
    hybrid_db = str(hybrid_dir / "trade_journal.db")
    conn = sqlite3.connect(hybrid_db)
    row = conn.execute(
        "SELECT status FROM cycles WHERE cycle_id = ?", ("2026-06-11",)
    ).fetchone()
    conn.close()
    assert row is not None, "cycle row missing entirely"
    assert row[0] is not None, "cycle row status is NULL — log_cycle_end not called on error path"


# ---------------------------------------------------------------------------
# FIX 3: reduce_only=True when flattening an existing position
# ---------------------------------------------------------------------------

class FakeExchangeWithPosition:
    """Exchange that reports an existing LONG position and records reduce_only."""

    def __init__(self, existing_qty: float = 0.5):
        self._existing_qty = existing_qty
        self.orders = []          # list of (symbol, side, qty, reduce_only)

    def get_total_portfolio_value(self):
        return 10_000.0

    def get_usdt_balance(self):
        return 10_000.0

    def get_current_position(self, symbol):
        return self._existing_qty   # long position

    def round_quantity(self, symbol, q):
        return round(q, 3)

    def min_notional(self, symbol):
        return 5.0

    def get_ticker_price(self, symbol):
        return 65_000.0

    def place_market_order(self, symbol, side, qty, reduce_only=False):
        self.orders.append((symbol, side, qty, reduce_only))
        return {"orderId": 2, "status": "FILLED"}

    def cancel_all_orders(self, symbol):
        return []

    def list_open_stops(self, symbol):
        return []

    def place_stop_loss(self, symbol, qty, stop_price, stop_side):
        return {"orderId": 998}

    def cancel_order(self, symbol, order_id):
        pass


class FlatSignalGraph:
    """Returns a zero-signal multiplier so the hybrid target is flat (0), forcing a
    reducing SELL against the existing long position."""
    def propagate_with_modulator(self, coin, date):
        mp = {
            "coin": coin,
            "llm_multiplier": 0.0,    # zero the position
            "effective_weight": 0.0,
            "position": 0.0,
            "regime": "bear",
            "llm_uncertainty": 0.5,
        }
        return ({}, mp, {"coin": coin, "direction": "short", "magnitude": 0.5}, "ok")


def test_reduce_only_set_when_flattening(tmp_path, monkeypatch):
    """When the hybrid closes (reduces) an existing position, reduce_only=True
    must be passed to place_market_order."""
    quant_dir = tmp_path / "data"
    quant_dir.mkdir()
    hybrid_dir = tmp_path / "data-hybrid"

    # Seed a flat/sell signal so both horizons predict DOWN
    j = Journal(str(quant_dir / "trade_journal.db"))
    j.log_cycle_start("2026-06-11", git_sha="x")
    preds_df = pd.DataFrame([
        {"coin": "bitcoin", "horizon": 7,  "prediction": -0.03,
         "ref_price": 65_000.0, "bundle_route": "bitcoin_78f"},
        {"coin": "bitcoin", "horizon": 14, "prediction": -0.05,
         "ref_price": 65_000.0, "bundle_route": "bitcoin_78f"},
    ])
    j.record_predictions(cycle_id="2026-06-11", preds_df=preds_df)
    j.close()

    # Pre-seed the hybrid hold-state to reflect the existing long position
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    hyb_j = Journal(str(hybrid_dir / "trade_journal.db"))
    hyb_j.log_cycle_start("2026-06-10", git_sha="seed")
    hyb_j.upsert_hold_state(
        coin="bitcoin", current_dir=1, bars_held=1,
        entry_price=65_000.0, entry_base=0.5, entry_cycle="2026-06-10",
    )
    hyb_j.log_cycle_end("2026-06-10", status="ok")
    hyb_j.close()

    monkeypatch.setenv("HYBRID_BINANCE_API_KEY", "k")
    monkeypatch.setenv("HYBRID_BINANCE_API_SECRET", "s")
    monkeypatch.setenv("HYBRID_DATA_DIR", str(hybrid_dir))
    monkeypatch.setenv("QUANT_DATA_DIR", str(quant_dir))
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin")
    monkeypatch.setenv("BINANCE_API_KEY", "qk")
    monkeypatch.setenv("BINANCE_API_SECRET", "qs")
    monkeypatch.setenv("COINGLASS_API_KEY", "cgk")

    _seed_ohlcv_cache(quant_dir, "BTCUSDT")

    # Exchange reports existing 0.5 BTC long
    fake_ex = FakeExchangeWithPosition(existing_qty=0.5)
    res = hybrid_runner.run_hybrid_cycle(
        cycle_id="2026-06-11", dry_run=False,
        _exchange=fake_ex, _graph=FlatSignalGraph(),
    )

    assert res.status == "ok", f"cycle failed: {res.status}: {res.error_msg}"
    assert len(fake_ex.orders) >= 1, "expected a closing/reducing SELL order"
    sell_orders = [(sym, side, qty, ro) for sym, side, qty, ro in fake_ex.orders
                   if side == "SELL"]
    assert sell_orders, "no SELL orders placed to reduce the long"
    # Every SELL that reduces the long must have reduce_only=True
    for sym, side, qty, ro in sell_orders:
        assert ro is True, (
            f"SELL order {sym} qty={qty} has reduce_only={ro}, expected True"
        )
