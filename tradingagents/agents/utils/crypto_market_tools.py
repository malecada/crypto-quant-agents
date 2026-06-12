from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_crypto_data(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'solana')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Retrieve cryptocurrency OHLCV (Open, High, Low, Close, Volume) price data.

    Uses Binance as primary source with CoinGecko fallback.
    Returns daily candle data for technical analysis.
    """
    return route_to_vendor("get_crypto_data", symbol, start_date, end_date)


@tool
def get_crypto_indicators(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    indicator: Annotated[str, "Technical indicator to calculate (e.g., 'rsi', 'macd', 'boll', 'atr', 'close_50_sma')"],
    curr_date: Annotated[str, "Current trading date in YYYY-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for indicator values"],
) -> str:
    """Compute a SINGLE technical indicator for a cryptocurrency.

    PREFER `get_crypto_indicators_batch` over this tool — it returns the
    full standard indicator set in one call, eliminating the sequential
    tool-call chain that dominates LLM cost. Only use this single-indicator
    tool when you need a specific indicator NOT covered by the batch tool.

    Available indicators: close_50_sma, close_200_sma, close_10_ema,
    macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma, mfi.

    Uses the same stockstats library as stock analysis, applied to crypto OHLCV data.
    """
    return route_to_vendor("get_crypto_indicators", symbol, indicator, curr_date, look_back_days)


# Default indicator set covers trend, momentum, volatility, volume —
# the four signal classes the market analyst typically inspects.
_DEFAULT_BATCH_INDICATORS = (
    "close_50_sma",   # trend (medium)
    "close_200_sma",  # trend (long)
    "close_10_ema",   # trend (short)
    "rsi",            # momentum
    "macd", "macds",  # momentum (line + signal)
    "boll", "boll_ub", "boll_lb",  # volatility (Bollinger)
    "atr",            # volatility (ATR)
    "vwma",           # volume-weighted trend
    "mfi",            # volume-weighted momentum
)


@tool
def get_crypto_indicators_batch(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    curr_date: Annotated[str, "Current trading date in YYYY-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back for indicator values"] = 30,
) -> str:
    """Fetch the FULL standard technical-indicator set in a SINGLE tool call.

    Replaces the typical 5-10 sequential `get_crypto_indicators` calls (one
    per indicator) with one batched response. Removes the quadratic input-
    token blow-up that comes from accumulating tool-call history in the
    market analyst's conversation context.

    The batched indicator set covers:
      * Trend:      close_10_ema, close_50_sma, close_200_sma, vwma
      * Momentum:   rsi, macd, macds, mfi
      * Volatility: boll, boll_ub, boll_lb, atr

    Returns a markdown block with one section per indicator.
    """
    blocks: list[str] = [
        f"# Indicators for {symbol} as of {curr_date} ({look_back_days}d lookback)",
        "",
    ]
    for ind in _DEFAULT_BATCH_INDICATORS:
        try:
            chunk = route_to_vendor(
                "get_crypto_indicators", symbol, ind, curr_date, look_back_days,
            )
        except Exception as e:
            chunk = f"[{ind} unavailable: {e}]"
        blocks.append(f"## {ind}")
        blocks.append(chunk if chunk else "(no data)")
        blocks.append("")
    return "\n".join(blocks)
