import os
import pytest


@pytest.fixture
def env_vars(monkeypatch):
    monkeypatch.setenv("LIVE_MODE", "false")
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")
    monkeypatch.setenv("BINANCE_BASE_URL", "https://testnet.binancefuture.com")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123")
    monkeypatch.setenv("MAX_LEVERAGE", "3.0")
    monkeypatch.setenv("MAX_DAILY_LOSS_PCT", "0.15")
    monkeypatch.setenv("STOP_LOSS_PCT", "0.03")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "3")
    monkeypatch.setenv("TARGET_VOL", "0.10")
    monkeypatch.setenv("KELLY_FRACTION", "0.5")
    monkeypatch.setenv("VOL_LOOKBACK", "20")
    monkeypatch.setenv("VOL_CAP_PCT", "0.95")
    monkeypatch.setenv("CONFIDENCE_REF_RETURN", "0.02")
    monkeypatch.setenv("EARLY_EXIT_LOSS", "0.015")
    monkeypatch.setenv("MIN_HOLD", "7")
    monkeypatch.setenv("TREND_SMA", "30")
    monkeypatch.setenv("TREND_MULTIPLIER", "1.5")
    monkeypatch.setenv("HORIZONS", "7,14")
    monkeypatch.setenv("SYMMETRIC", "true")
    monkeypatch.setenv("ARIMA_FILTER", "false")
    monkeypatch.setenv("INITIAL_CAPITAL", "10000")
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin,ethereum,binancecoin")
    # Set COINGLASS_API_KEY deterministically so existing tests don't
    # depend on `.env` leaking the key from the worktree (CI runs in a
    # clean env and would otherwise fail `load_config()`'s required-env
    # check).
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")


def test_load_returns_typed_config(env_vars):
    from tradingagents.execution.live.config import load_config

    cfg = load_config()
    assert cfg.live_mode is False
    assert cfg.binance_api_key == "k"
    assert cfg.max_leverage == 3.0
    assert cfg.horizons == [7, 14]
    assert cfg.symmetric is True
    assert cfg.coin_universe == ["bitcoin", "ethereum", "binancecoin"]
    assert cfg.initial_capital == 10000.0


def test_to_binance_symbol_maps_known_coins():
    from tradingagents.execution.live.config import to_binance_symbol
    assert to_binance_symbol("bitcoin") == "BTCUSDT"
    assert to_binance_symbol("ethereum") == "ETHUSDT"
    assert to_binance_symbol("binancecoin") == "BNBUSDT"


def test_to_binance_symbol_falls_back_to_uppercase():
    from tradingagents.execution.live.config import to_binance_symbol
    # litecoin is not in the known map → upper-case fallback.
    assert to_binance_symbol("litecoin") == "LITECOINUSDT"


def test_missing_required_raises(monkeypatch):
    monkeypatch.delenv("BINANCE_API_KEY", raising=False)
    monkeypatch.delenv("BINANCE_API_SECRET", raising=False)
    from tradingagents.execution.live.config import load_config

    with pytest.raises(ValueError, match="BINANCE_API_KEY"):
        load_config()


def test_validate_rejects_negative_leverage(env_vars, monkeypatch):
    monkeypatch.setenv("MAX_LEVERAGE", "-1")
    from tradingagents.execution.live.config import load_config

    with pytest.raises(ValueError, match="MAX_LEVERAGE"):
        load_config()


def _set_v5_min_env(monkeypatch) -> None:
    """Set the minimum env vars required by `LiveConfig.from_env()` so V5
    tests can focus on the new V5 behaviour without re-asserting the existing
    required-env-var contract."""
    monkeypatch.setenv("BINANCE_API_KEY", "k")
    monkeypatch.setenv("BINANCE_API_SECRET", "s")


def test_v5_routing_defaults(monkeypatch) -> None:
    """V5 default ROUTING + 8-coin universe + kelly=0.25."""
    _set_v5_min_env(monkeypatch)
    monkeypatch.delenv("COIN_UNIVERSE", raising=False)
    monkeypatch.delenv("KELLY_FRACTION", raising=False)
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")

    from tradingagents.execution.live.config import LiveConfig
    cfg = LiveConfig.from_env()

    assert cfg.coin_universe == [
        "bitcoin", "ethereum", "binancecoin", "solana",
        "ripple", "dogecoin", "cardano", "tron",
    ]
    assert cfg.max_open_positions == 8
    assert cfg.kelly_fraction == 0.25
    assert "bitcoin" in cfg.routing
    assert cfg.routing["bitcoin"] == {"feature_set": "78f", "pool": ["bitcoin", "ethereum"]}
    assert cfg.routing["ethereum"] == {"feature_set": "193f", "pool": ["bitcoin", "ethereum"]}
    assert cfg.routing["binancecoin"] == {"feature_set": "78f",
                                           "pool": ["bitcoin", "ethereum", "binancecoin"]}
    assert cfg.routing["solana"] == {"feature_set": "193f",
                                      "pool": ["bitcoin", "ethereum", "solana"]}
    assert cfg.routing["cardano"] == {"feature_set": "193f",
                                      "pool": ["bitcoin", "ethereum", "cardano"]}
    assert cfg.routing["ripple"] == {"feature_set": "78f",
                                     "pool": ["bitcoin", "ethereum", "ripple"]}
    assert cfg.coinglass_api_key == "test-key"
    assert cfg.data_refresh_critical == {"ohlcv", "coinmetrics"}


def test_v5_missing_coinglass_key_raises(monkeypatch) -> None:
    _set_v5_min_env(monkeypatch)
    monkeypatch.delenv("COINGLASS_API_KEY", raising=False)
    from tradingagents.execution.live.config import LiveConfig

    with pytest.raises(ValueError, match="COINGLASS_API_KEY"):
        LiveConfig.from_env()


def test_v5_coin_universe_routing_drift_raises(monkeypatch) -> None:
    """COIN_UNIVERSE entry without routing entry raises clearly."""
    _set_v5_min_env(monkeypatch)
    # litecoin has no routing entry (cardano now does — it is an 8-coin satellite).
    monkeypatch.setenv("COIN_UNIVERSE", "bitcoin,ethereum,litecoin")
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")
    from tradingagents.execution.live.config import LiveConfig
    with pytest.raises(ValueError, match="litecoin.*no routing entry"):
        LiveConfig.from_env()


def test_v5_data_root_default(monkeypatch) -> None:
    _set_v5_min_env(monkeypatch)
    monkeypatch.delenv("TRADINGAGENTS_DATA_ROOT", raising=False)
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")
    from tradingagents.execution.live.config import LiveConfig
    cfg = LiveConfig.from_env()
    assert cfg.data_root == "data"


def test_v5_data_root_env_override(monkeypatch) -> None:
    _set_v5_min_env(monkeypatch)
    monkeypatch.setenv("TRADINGAGENTS_DATA_ROOT", "/sandbox/data")
    monkeypatch.setenv("COINGLASS_API_KEY", "test-key")
    from tradingagents.execution.live.config import LiveConfig
    cfg = LiveConfig.from_env()
    assert cfg.data_root == "/sandbox/data"


def test_eight_coin_routing_and_bases_present():
    """8-coin expansion: satellites have correct feature set + 2+1 pool + base."""
    from tradingagents.execution.live.config import (
        _V5_DEFAULT_ROUTING, to_binance_symbol,
    )
    expected = {"ripple": "78f", "dogecoin": "78f", "cardano": "193f", "tron": "78f"}
    for sat, fs in expected.items():
        assert _V5_DEFAULT_ROUTING[sat]["feature_set"] == fs
        assert _V5_DEFAULT_ROUTING[sat]["pool"] == ["bitcoin", "ethereum", sat]
    assert to_binance_symbol("ripple") == "XRPUSDT"
    assert to_binance_symbol("dogecoin") == "DOGEUSDT"
    assert to_binance_symbol("cardano") == "ADAUSDT"
    assert to_binance_symbol("tron") == "TRXUSDT"
