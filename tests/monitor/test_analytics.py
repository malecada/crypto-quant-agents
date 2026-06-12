"""Pure aggregation of Binance income records + journal slippage."""
from __future__ import annotations

from tradingagents.monitor import analytics


INCOME = [
    {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "10.0"},
    {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "-4.0"},
    {"symbol": "ETHUSDT", "incomeType": "REALIZED_PNL", "income": "6.0"},
    {"symbol": "BTCUSDT", "incomeType": "COMMISSION", "income": "-0.5"},
    {"symbol": "ETHUSDT", "incomeType": "FUNDING_FEE", "income": "-0.2"},
    {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "0"},
]


def test_income_summary():
    s = analytics.income_summary(INCOME)
    assert s["realized_pnl_per_coin"] == {"BTCUSDT": 6.0, "ETHUSDT": 6.0}
    assert s["realized_pnl_total"] == 12.0
    assert s["fees_total"] == -0.5
    assert s["funding_total"] == -0.2
    # win rate over NONZERO realized-pnl records: wins 10,6 of [10,-4,6]
    assert abs(s["win_rate"] - 2 / 3) < 1e-9
    assert s["n_closing_fills"] == 3


def test_income_summary_empty():
    s = analytics.income_summary([])
    assert s["realized_pnl_per_coin"] == {}
    assert s["win_rate"] is None
    assert s["n_closing_fills"] == 0


def test_slippage_stats():
    trades = [{"slippage": 1.0}, {"slippage": 3.0}, {"slippage": None}]
    st = analytics.slippage_stats(trades)
    assert st == {"mean": 2.0, "max": 3.0, "n": 2}


def test_slippage_stats_empty():
    assert analytics.slippage_stats([]) == {"mean": None, "max": None, "n": 0}


def test_income_summary_malformed_income_skipped():
    s = analytics.income_summary([
        {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "N/A"},
        {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": None},
        {"symbol": "BTCUSDT", "incomeType": "REALIZED_PNL", "income": "5.0"},
    ])
    assert s["realized_pnl_per_coin"] == {"BTCUSDT": 5.0}
    assert s["n_closing_fills"] == 1
