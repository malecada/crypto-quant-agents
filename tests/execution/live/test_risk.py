import pytest


def test_leverage_cap_passes_within_limit():
    from tradingagents.execution.live.risk import check_leverage
    ok, _ = check_leverage(size=2.5, max_leverage=3.0)
    assert ok is True


def test_leverage_cap_rejects_over_limit():
    from tradingagents.execution.live.risk import check_leverage
    ok, reason = check_leverage(size=3.5, max_leverage=3.0)
    assert ok is False
    assert "3.5" in reason


def test_daily_loss_kill_switch_triggers_at_threshold():
    from tradingagents.execution.live.risk import check_daily_loss
    ok, reason = check_daily_loss(pnl_today_pct=-0.16, max_loss_pct=0.15)
    assert ok is False
    assert "kill" in reason.lower()


def test_daily_loss_under_limit_passes():
    from tradingagents.execution.live.risk import check_daily_loss
    ok, _ = check_daily_loss(pnl_today_pct=-0.05, max_loss_pct=0.15)
    assert ok is True


def test_max_open_positions_blocks_new_entry():
    from tradingagents.execution.live.risk import check_max_positions
    ok, _ = check_max_positions(current_open=3, max_open=3, opening_new=True)
    assert ok is False


def test_max_open_positions_allows_close():
    from tradingagents.execution.live.risk import check_max_positions
    ok, _ = check_max_positions(current_open=3, max_open=3, opening_new=False)
    assert ok is True


def test_frequency_guard_blocks_second_trade_today():
    from tradingagents.execution.live.risk import check_frequency_guard
    ok, _ = check_frequency_guard(coin="BTC", trades_today_count=1)
    assert ok is False


def test_frequency_guard_allows_first_trade():
    from tradingagents.execution.live.risk import check_frequency_guard
    ok, _ = check_frequency_guard(coin="BTC", trades_today_count=0)
    assert ok is True
