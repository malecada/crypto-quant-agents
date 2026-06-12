"""PIT stubs for news_data vendor category.

When config['data_vendors']['news_data'] = 'news_data_pit', these stubs
return explicit 'not available in PIT mode' messages instead of letting
the router silently fall back to today-relative yfinance/alpha_vantage.

In P1, the crypto_sentiment_analyst binds both `get_news` and
`get_global_news` alongside the PIT Alpaca tool. This module ensures the
non-Alpaca legs don't leak post-cutoff data.

Signatures match the live LangChain tools in
``tradingagents/agents/utils/news_data_tools.py`` so ``route_to_vendor``
can dispatch positionally:

    get_news(ticker, start_date, end_date)
    get_global_news(curr_date, look_back_days=7, limit=5)
"""
from __future__ import annotations

from typing import Annotated


def get_news_pit_stub(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date yyyy-mm-dd"],
    end_date: Annotated[str, "End date yyyy-mm-dd"],
) -> str:
    """P1 stub: ticker-specific news via yfinance/alpha_vantage is today-relative
    and would leak post-cutoff data in a historical backtest.

    In PIT mode, the crypto_sentiment_analyst should rely on the Alpaca
    PIT news source (routed via get_crypto_google_news -> crypto_sentiment_pit).
    """
    return (
        f"Ticker news is not available in PIT mode for {ticker} "
        f"({start_date} to {end_date}). "
        "Use the Alpaca PIT news source instead (via get_crypto_google_news)."
    )


def get_global_news_pit_stub(
    curr_date: Annotated[str, "Current date yyyy-mm-dd"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """P1 stub: global macro news via yfinance/alpha_vantage is today-relative
    and would leak post-cutoff data in a historical backtest.

    Signature matches the live ``get_global_news`` tool
    (``curr_date, look_back_days=7, limit=5``) so positional dispatch works.
    """
    return (
        f"Global news is not available in PIT mode (as of {curr_date}, "
        f"look_back_days={look_back_days}, limit={limit}). "
        "Use the Alpaca PIT news source for ticker-relevant news."
    )
