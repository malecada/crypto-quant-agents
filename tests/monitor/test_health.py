from tradingagents.monitor import health


def test_read_structured_log(log_dir):
    records = health.read_structured_log(log_dir)
    assert len(records) == 3
    assert records[0]["step"] == "data_refresh"
    assert records[-1]["status"] == "error"


def test_read_structured_log_missing_dir():
    assert health.read_structured_log("/nonexistent/logs") == []


def test_read_structured_log_empty_dir(tmp_path):
    assert health.read_structured_log(str(tmp_path)) == []


def test_recent_errors(log_dir):
    records = health.read_structured_log(log_dir)
    errors = health.recent_errors(records)
    assert len(errors) == 1
    assert errors[0]["step"] == "execute"
    assert errors[0]["payload"]["error"] == "binance timeout"
