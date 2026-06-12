"""Online smoke test against Binance Futures testnet.

Run only with credentials in environment + RUN_ONLINE_TESTS=1:
    RUN_ONLINE_TESTS=1 pytest tests/execution/test_exchange_smoke.py -v
"""
import os
import pytest

_ONLINE = os.environ.get("RUN_ONLINE_TESTS") == "1"
_HAS_CREDS = bool(os.environ.get("BINANCE_API_KEY"))


@pytest.mark.skipif(not (_ONLINE and _HAS_CREDS),
                    reason="requires RUN_ONLINE_TESTS=1 and BINANCE_API_KEY set")
def test_testnet_ticker_query():
    from tradingagents.execution.exchange import ExchangeClient

    client = ExchangeClient(
        api_key=os.environ["BINANCE_API_KEY"],
        api_secret=os.environ["BINANCE_API_SECRET"],
        testnet=True,
    )
    price = client.get_ticker_price("BTCUSDT")
    assert price > 1000


@pytest.mark.skipif(not (_ONLINE and _HAS_CREDS),
                    reason="requires RUN_ONLINE_TESTS=1 and BINANCE_API_KEY set")
def test_testnet_balance_query():
    from tradingagents.execution.exchange import ExchangeClient

    client = ExchangeClient(
        api_key=os.environ["BINANCE_API_KEY"],
        api_secret=os.environ["BINANCE_API_SECRET"],
        testnet=True,
    )
    balance = client.get_usdt_balance()
    assert balance >= 0
