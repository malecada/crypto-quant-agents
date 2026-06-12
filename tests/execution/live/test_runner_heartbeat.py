"""AL1: dead-man heartbeat + alert-failure visibility."""
import logging

import pytest


def test_heartbeat_file_written(tmp_path):
    from tradingagents.execution.live import runner as R
    R._write_heartbeat(tmp_path)
    hb = tmp_path / "last_cycle_heartbeat.txt"
    assert hb.exists() and hb.read_text().strip()


def test_send_alert_logs_on_exception(monkeypatch, caplog):
    from tradingagents.execution.live import notify

    def _boom(**kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(notify, "_post_telegram", _boom)
    with caplog.at_level(logging.ERROR):
        notify.send_alert(bot_token="t", chat_id="c", severity="X", message="m")
    assert any("alert failed" in r.message.lower() for r in caplog.records)


def test_post_telegram_raises_on_4xx(monkeypatch):
    """A 4xx must raise (so callers log it) rather than silently vanish."""
    import requests
    from tradingagents.execution.live import notify

    class _Resp:
        status_code = 400
        def raise_for_status(self):
            raise requests.HTTPError("400 Bad Request")

    monkeypatch.setattr(notify.requests, "post", lambda *a, **k: _Resp())
    with pytest.raises(requests.HTTPError):
        notify._post_telegram(token="t", chat_id="c", text="x")
