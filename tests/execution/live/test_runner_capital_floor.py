"""S3265: portfolio_before=0 book-wipe floor.

get_total_portfolio_value must raise on a missing totalMarginBalance (rather
than silently returning 0.0), and the runner must abort the cycle when equity
is at/below the sanity floor instead of sizing every coin to zero.
"""
import types

import pytest


def _fake_exchange(account):
    """A bare ExchangeClient with _retry/_client stubbed to return `account`."""
    from tradingagents.execution.exchange import ExchangeClient
    ex = ExchangeClient.__new__(ExchangeClient)
    ex._client = types.SimpleNamespace(futures_account=lambda: account)
    ex._retry = lambda fn: fn()
    return ex


def test_get_total_portfolio_value_raises_on_missing_margin_balance():
    ex = _fake_exchange({})  # error envelope without totalMarginBalance
    with pytest.raises(ValueError, match="totalMarginBalance"):
        ex.get_total_portfolio_value()


def test_get_total_portfolio_value_returns_value_when_present():
    ex = _fake_exchange({"totalMarginBalance": "4732.5"})
    assert ex.get_total_portfolio_value() == pytest.approx(4732.5)


def test_abort_if_no_capital_helper():
    from tradingagents.execution.live import runner as R
    assert R._abort_if_no_capital(portfolio_before=0.0, floor=100.0) is True
    assert R._abort_if_no_capital(portfolio_before=50.0, floor=100.0) is True
    assert R._abort_if_no_capital(portfolio_before=5000.0, floor=100.0) is False
    assert R._abort_if_no_capital(portfolio_before=100.0, floor=100.0) is True  # at floor → abort
