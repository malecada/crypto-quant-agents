from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)

# Crypto-specific tools
from tradingagents.agents.utils.crypto_market_tools import (
    get_crypto_data,
    get_crypto_indicators,
    get_crypto_indicators_batch,
)
from tradingagents.agents.utils.onchain_tools import (
    get_funding_rates,
    get_tvl_metrics,
    get_stablecoin_metrics,
    get_gas_metrics,
    get_stablecoin_supply,
    get_onchain_pit,
)
from tradingagents.agents.utils.crypto_sentiment_tools import (
    get_reddit_posts,
    get_crypto_google_news,
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Only applied to user-facing agents (analysts, portfolio manager).
    Internal debate agents stay in English for reasoning quality.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def build_instrument_context(ticker: str) -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers.

    When asset-name anonymization is active (Phase 3 / A4), this returns
    a context referring to the masked alias (``Asset_X``) instead of the
    raw ticker. The Portfolio Manager re-attaches the real identity at
    the Layer 3 boundary via ``anonymizer.unmask``. See
    ``tradingagents/agents/utils/anonymizer.py`` for the rationale
    (Glasserman & Lin 2309.17322; Choi et al. 2510.07517).
    """
    from tradingagents.agents.utils.anonymizer import is_enabled, mask

    if is_enabled():
        alias = mask(ticker)
        return (
            f"The instrument to analyze is referred to as `{alias}` "
            "throughout this analysis (its real identity is intentionally "
            "withheld to reduce LLM training-corpus bias). "
            f"Use `{alias}` in every tool call, report, and recommendation."
        )
    return (
        f"The instrument to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`)."
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
