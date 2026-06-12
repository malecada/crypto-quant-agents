from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_reddit_posts(
    coin_name: Annotated[str, "Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum', 'Solana')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Reddit posts about a cryptocurrency from crypto subreddits.

    Returns raw post data (titles, content, engagement scores) from
    r/CryptoCurrency, r/CryptoCurrencyTrading, and other crypto subreddits.
    Analyze the raw text to gauge retail sentiment, hype cycles, and fear levels.
    """
    return route_to_vendor("get_reddit_posts", coin_name, start_date, end_date)


@tool
def get_crypto_google_news(
    coin_name: Annotated[str, "Name of the cryptocurrency (e.g., 'Bitcoin', 'Ethereum')"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """Fetch Google News articles about a cryptocurrency.

    Returns article titles, descriptions, and sources.
    Useful for identifying mainstream narratives, regulatory developments,
    and major events affecting the cryptocurrency.
    """
    return route_to_vendor("get_crypto_google_news", coin_name, start_date, end_date)
