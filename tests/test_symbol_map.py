"""Reverse Binance-symbol mapping for the live monitor holdings view."""
from __future__ import annotations

from tradingagents.execution.live.config import (
    from_binance_symbol,
    to_binance_symbol,
)


def test_from_binance_symbol_round_trips():
    coins = [
        "bitcoin", "ethereum", "solana", "ripple",
        "dogecoin", "cardano", "tron", "binancecoin",
    ]
    for coin in coins:
        assert from_binance_symbol(to_binance_symbol(coin)) == coin


def test_from_binance_symbol_unknown_base_lowercased():
    assert from_binance_symbol("XYZUSDT") == "xyz"
