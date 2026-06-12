"""ML prediction models for cryptocurrency price forecasting.

Ported from Krypto-v0/src/models/, adapted to use TradingAgents' config
system and CoinGecko/Binance data vendor.

Each model exposes a ``forecast_next(symbol, lookback_days)`` function that:
  - Accepts a CoinGecko ID (e.g. "bitcoin") and optional lookback_days
  - Fetches OHLCV data, trains the model, and returns a formatted string
  - Handles errors gracefully (returns error message, does not raise)

Usage::

    from tradingagents.models.rf_model import forecast_next as rf_forecast
    from tradingagents.models.arima_model import forecast_next as arima_forecast
    from tradingagents.models.onchain_model import forecast_next as onchain_forecast

    result = rf_forecast("bitcoin", lookback_days=300)
    print(result)  # Formatted string for LLM consumption
"""

from tradingagents.models.prediction import Prediction

__all__ = [
    "Prediction",
]
