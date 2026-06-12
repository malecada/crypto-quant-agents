import math

from tradingagents.monitor import metrics


def test_max_drawdown():
    # peak 120, trough 90 -> (90-120)/120 = -0.25
    series = [100.0, 120.0, 110.0, 90.0, 130.0]
    assert math.isclose(metrics.max_drawdown(series), -0.25)


def test_max_drawdown_monotonic_increasing():
    assert metrics.max_drawdown([100.0, 110.0, 120.0]) == 0.0


def test_max_drawdown_too_short():
    assert metrics.max_drawdown([100.0]) == 0.0


def test_sharpe_zero_variance():
    # constant equity -> no returns variance -> Sharpe 0.0
    assert metrics.sharpe([100.0, 100.0, 100.0]) == 0.0


def test_sharpe_positive_trend():
    series = [100.0, 101.0, 102.0, 103.5, 104.0, 106.0]
    assert metrics.sharpe(series) > 0.0


def test_sharpe_too_short():
    assert metrics.sharpe([100.0]) == 0.0


def test_cumulative_pnl():
    trades = [{"pnl": 100.0}, {"pnl": -30.0}, {"pnl": None}, {"pnl": 50.0}]
    assert metrics.cumulative_pnl(trades) == 120.0


def test_equity_series_from_snapshots():
    snaps = [
        {"ts": "2026-05-19T07:05:00+00:00", "total_value": 10150.0},
        {"ts": "2026-05-20T07:05:00+00:00", "total_value": 10280.0},
    ]
    series = metrics.equity_series(snaps, trades=[], start_capital=10000.0)
    assert series == [
        {"ts": "2026-05-19T07:05:00+00:00", "value": 10150.0},
        {"ts": "2026-05-20T07:05:00+00:00", "value": 10280.0},
    ]


def test_equity_series_fallback_to_trades():
    # no snapshots -> reconstruct from cumulative realized PnL
    trades = [
        {"cycle_id": "c1", "pnl": 100.0},
        {"cycle_id": "c1", "pnl": 50.0},
        {"cycle_id": "c2", "pnl": -30.0},
    ]
    series = metrics.equity_series([], trades=trades, start_capital=10000.0)
    assert series[-1]["value"] == 10120.0
    assert series[0]["value"] == 10000.0  # start point prepended


def test_equity_series_empty():
    assert metrics.equity_series([], trades=[], start_capital=10000.0) == []


def test_drawdown_series():
    eq = [{"ts": "t1", "value": 100.0}, {"ts": "t2", "value": 110.0},
          {"ts": "t3", "value": 99.0}]
    dd = metrics.drawdown_series(eq)
    assert dd == [{"ts": "t1", "value": 0.0}, {"ts": "t2", "value": 0.0},
                  {"ts": "t3", "value": -0.1}]


def test_drawdown_series_empty():
    assert metrics.drawdown_series([]) == []


def test_rolling_sharpe_short_series_is_empty():
    eq = [{"ts": f"t{i}", "value": 100.0 + i} for i in range(10)]
    assert metrics.rolling_sharpe(eq, window=30) == []


def test_rolling_sharpe_emits_from_window():
    # 40 points, constant 1% growth -> first point at index 30, huge sharpe
    vals, v = [], 100.0
    for i in range(40):
        vals.append({"ts": f"t{i}", "value": v})
        v *= 1.01
    rs = metrics.rolling_sharpe(vals, window=30)
    assert len(rs) == 10
    assert rs[0]["ts"] == "t30"
    assert all(p["value"] > 0 for p in rs)
