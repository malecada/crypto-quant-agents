"""Unit tests for the fills-reconciliation helper in live.runner.

Closes V5 parity gap #2: live trade journal had `trades.fees` and
`trades.pnl` permanently NULL because `place_market_order` returns only the
order envelope; per-fill `commission` and `realizedPnl` live on
`/fapi/v1/userTrades?orderId=...`. After each successful order placement the
runner now sums those fields and backfills the journal row.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_reconcile_fills_sums_commission_and_realized_pnl_into_journal():
    from tradingagents.execution.live.runner import _reconcile_fills

    ex = MagicMock()
    ex.get_user_trades.return_value = [
        {"commission": "0.0782", "commissionAsset": "USDT",
         "realizedPnl": "1.23", "qty": "0.001", "price": "77000"},
        {"commission": "0.0391", "commissionAsset": "USDT",
         "realizedPnl": "0.62", "qty": "0.0005", "price": "77010"},
    ]
    journal = MagicMock()

    _reconcile_fills(
        ex, journal, symbol="BTCUSDT", order_id="12345", trade_id=42,
    )

    ex.get_user_trades.assert_called_once_with("BTCUSDT", "12345")
    journal.update_trade_fills.assert_called_once_with(
        42, fees=pytest.approx(0.1173), realized_pnl=pytest.approx(1.85),
    )


def test_reconcile_fills_handles_negative_commission():
    """Binance commission rebates appear as negative values; magnitude is the cost."""
    from tradingagents.execution.live.runner import _reconcile_fills

    ex = MagicMock()
    ex.get_user_trades.return_value = [
        {"commission": "-0.0782", "commissionAsset": "USDT", "realizedPnl": "0.00"},
    ]
    journal = MagicMock()

    _reconcile_fills(
        ex, journal, symbol="BTCUSDT", order_id="111", trade_id=1,
    )

    # Cost is the magnitude — operator wants "fees paid", not signed bookkeeping.
    journal.update_trade_fills.assert_called_once_with(
        1, fees=pytest.approx(0.0782), realized_pnl=pytest.approx(0.0),
    )


def test_reconcile_fills_no_op_when_no_fills():
    """Empty fills list (e.g. dry-run sentinel order_id) must NOT update."""
    from tradingagents.execution.live.runner import _reconcile_fills

    ex = MagicMock()
    ex.get_user_trades.return_value = []
    journal = MagicMock()

    _reconcile_fills(
        ex, journal, symbol="BTCUSDT", order_id="0", trade_id=1,
    )

    journal.update_trade_fills.assert_not_called()


def test_reconcile_fills_swallows_exchange_error():
    """A fill-fetch failure must not propagate — the trade row already exists,
    we just lose the fees/pnl backfill (operator-visible via NULL columns)."""
    from tradingagents.execution.live.runner import _reconcile_fills

    ex = MagicMock()
    ex.get_user_trades.side_effect = RuntimeError("network blip")
    journal = MagicMock()

    # Must not raise.
    _reconcile_fills(
        ex, journal, symbol="BTCUSDT", order_id="999", trade_id=7,
    )

    journal.update_trade_fills.assert_not_called()


def test_reconcile_fills_skips_blank_order_id():
    """`order_id=""` or `"dry-run"` means no real order — skip the API hit."""
    from tradingagents.execution.live.runner import _reconcile_fills

    ex = MagicMock()
    journal = MagicMock()

    _reconcile_fills(ex, journal, symbol="BTCUSDT", order_id="", trade_id=1)
    _reconcile_fills(ex, journal, symbol="BTCUSDT", order_id="dry-run", trade_id=2)

    ex.get_user_trades.assert_not_called()
    journal.update_trade_fills.assert_not_called()
