"""P4 — exclude the in-progress daily bar from sizing inputs.

The cycle fires at 00:05 UTC; the OHLCV cache may contain today's daily bar
with only ~5 minutes of data. Computing realized vol / SMA30 on that partial
bar corrupts the vol denominator and the trend multiplier. Sizing must use
bars only through `asof` (yesterday's complete close), matching the prediction.
"""
import numpy as np
import pandas as pd


def _history(dates, prices):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": prices})


def test_drops_bar_after_asof():
    from tradingagents.execution.live.sizer import bars_through

    h = _history(["2026-05-27", "2026-05-28", "2026-05-29"], [100.0, 101.0, 102.0])
    out = bars_through(h, "2026-05-28")
    assert list(pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")) == [
        "2026-05-27", "2026-05-28",
    ]


def test_keeps_all_when_none_after_asof():
    from tradingagents.execution.live.sizer import bars_through

    h = _history(["2026-05-27", "2026-05-28"], [100.0, 101.0])
    out = bars_through(h, "2026-05-28")
    assert len(out) == 2


def test_empty_history_passthrough():
    from tradingagents.execution.live.sizer import bars_through

    assert len(bars_through(pd.DataFrame(), "2026-05-28")) == 0


def test_partial_bar_does_not_corrupt_realized_vol():
    """A wild in-progress bar must not change the realized-vol the sizer sees
    once it's dropped."""
    from tradingagents.execution.live.sizer import bars_through
    from tradingagents.strategies.v2_sizing import compute_realized_vol

    dates = pd.date_range("2026-03-01", periods=40, freq="D")
    rng = np.random.default_rng(7)
    prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 40))))
    clean = _history(dates, prices)
    asof = dates[-1].strftime("%Y-%m-%d")

    # Append a partial "today" bar with a +40% spike.
    spiked = _history(list(dates) + [dates[-1] + pd.Timedelta(days=1)],
                      prices + [prices[-1] * 1.4])

    v_clean = compute_realized_vol(bars_through(clean, asof)["close"].values, lookback=20)[-1]
    v_fixed = compute_realized_vol(bars_through(spiked, asof)["close"].values, lookback=20)[-1]
    assert v_fixed == v_clean  # the spike bar was excluded
