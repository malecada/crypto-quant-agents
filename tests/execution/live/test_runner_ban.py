"""run_cycle catches BinanceIPBan and emits BAN alert instead of CYCLE_ERROR."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch


def test_run_cycle_catches_BinanceIPBan_and_alerts_BAN(monkeypatch, tmp_path):
    monkeypatch.setenv("LIVE_MODE", "false")
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
    monkeypatch.setenv("COINGLASS_API_KEY", "test")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))

    from tradingagents.execution.exchange import BinanceIPBan
    from tradingagents.execution.live import runner

    until_ms = int(time.time() * 1000) + 600_000  # 10 min in future

    # Make the very first thing the runner does (load_config OK, exchange init)
    # crash with a BinanceIPBan as soon as the cycle touches the exchange.
    with patch.object(runner, "ExchangeClient") as mock_ex_cls, \
         patch.object(runner.notify, "send_alert") as mock_alert:
        mock_ex_cls.side_effect = BinanceIPBan(
            until_ms=until_ms,
            raw_message=f"APIError(code=-1003): banned until {until_ms}",
        )

        result = runner.run_cycle(cycle_id="2026-05-23-ban-test", dry_run=True)

    assert result.status == "banned", f"expected status='banned', got {result.status!r}"
    # Exactly one alert, severity=BAN.
    mock_alert.assert_called_once()
    kwargs = mock_alert.call_args.kwargs
    assert kwargs["severity"] == "BAN"
    assert "banned" in kwargs["message"].lower()
