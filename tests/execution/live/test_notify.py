from unittest.mock import patch, MagicMock


def test_send_daily_summary_calls_telegram_api():
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram") as mock_post:
        mock_post.return_value = MagicMock(status_code=200)
        notify.send_daily_summary(
            bot_token="t", chat_id="123",
            cycle_id="2026-05-12",
            portfolio_before=10000.0, portfolio_after=10120.0,
            trades=[{"coin": "BTC", "side": "BUY", "qty": 0.1, "price": 60000}],
            agreement_rate=1.0,
        )
        mock_post.assert_called_once()
        text = mock_post.call_args.kwargs["text"]
        assert "2026-05-12" in text
        assert "BTC" in text


def test_send_alert_includes_severity():
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram") as mock_post:
        notify.send_alert(
            bot_token="t", chat_id="123",
            severity="UNPROTECTED",
            message="BTC stop-loss failed",
        )
        text = mock_post.call_args.kwargs["text"]
        assert "UNPROTECTED" in text
        assert "BTC stop-loss" in text


def test_telegram_failure_does_not_crash():
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram", side_effect=Exception("network")):
        notify.send_daily_summary(
            bot_token="t", chat_id="123",
            cycle_id="2026-05-12",
            portfolio_before=10000.0, portfolio_after=10100.0,
            trades=[], agreement_rate=1.0,
        )


def test_send_daily_summary_emits_drawdown_and_cumulative_lines():
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram") as mock_post:
        notify.send_daily_summary(
            bot_token="t", chat_id="123",
            cycle_id="2026-05-18",
            portfolio_before=4716.6, portfolio_after=4760.5,
            trades=[], agreement_rate=1.0,
            peak_value=10000.0, initial_capital=10000.0,
        )
        text = mock_post.call_args.kwargs["text"]
        assert "Peak: 10000.00" in text
        assert "DD-from-peak: -52.40%" in text
        assert "Cumulative vs initial (10000)" in text
        assert "-52.40%" in text


def test_send_daily_summary_first_cycle_handles_zero_peak():
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram") as mock_post:
        notify.send_daily_summary(
            bot_token="t", chat_id="123",
            cycle_id="2026-05-18",
            portfolio_before=10000.0, portfolio_after=10050.0,
            trades=[], agreement_rate=1.0,
            peak_value=0.0, initial_capital=10000.0,
        )
        text = mock_post.call_args.kwargs["text"]
        # First cycle: peak defaults to portfolio_after so drawdown shows 0%.
        assert "Peak: 10050.00" in text
        assert "DD-from-peak: +0.00%" in text


def test_send_daily_summary_back_compat_without_new_kwargs():
    """Old callsites that didn't pass peak/initial still work."""
    from tradingagents.execution.live import notify

    with patch.object(notify, "_post_telegram") as mock_post:
        notify.send_daily_summary(
            bot_token="t", chat_id="123",
            cycle_id="2026-05-18",
            portfolio_before=100.0, portfolio_after=110.0,
            trades=[], agreement_rate=1.0,
        )
        text = mock_post.call_args.kwargs["text"]
        # Without initial_capital, the cumulative line is suppressed.
        assert "Cumulative vs initial" not in text
        # Peak line still renders (peak defaults to portfolio_after).
        assert "Peak: 110.00" in text
