from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_news,
    get_global_news,
    get_reddit_posts,
    get_crypto_google_news,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_crypto_sentiment_analyst(llm):

    def crypto_sentiment_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            get_reddit_posts,
            get_crypto_google_news,
        ]

        system_message = (
            """You are a cryptocurrency sentiment analyst tasked with analyzing market sentiment from multiple sources. Your role is to synthesize information from news outlets, social media, and community discussions to gauge the overall sentiment around a specific cryptocurrency.

**Available Data Sources (use all of them):**

1. **Alpha Vantage / Financial News** (get_news): Professional financial news with sentiment scores. Supports crypto tickers. Use this for institutional-grade news coverage.

2. **Global/Macro News** (get_global_news): Broader macroeconomic and regulatory news that impacts all crypto markets (Fed policy, government regulation, major institutional adoption).

3. **Reddit Crypto Communities** (get_reddit_posts): Raw posts from r/CryptoCurrency, r/CryptoCurrencyTrading, and other crypto subreddits. This captures retail investor sentiment, community hype/fear, and grassroots narratives. Analyze the raw text — look for:
   - Dominant emotions (excitement, fear, uncertainty)
   - Recurring themes and narratives
   - Engagement levels (high-score posts indicate strong community interest)
   - Contrarian signals (extreme bullishness/bearishness)

4. **Google News** (get_crypto_google_news): Mainstream news coverage. Good for catching regulatory developments, major partnerships, exchange listings, and events that mainstream media picks up.

**Analysis Framework:**
1. Gather data from ALL four sources
2. For each source, identify the dominant sentiment (bullish/bearish/neutral)
3. Weight crypto-specific sources (Reddit, crypto news) higher than general sources
4. Look for narrative convergence or divergence across sources
5. Identify key sentiment drivers and catalysts
6. Assess sentiment momentum (is sentiment shifting?)

**Key Signals to Watch:**
- Regulatory news (SEC, CFTC, global regulators)
- Exchange events (listings, delistings, hacks, proof-of-reserves)
- Whale activity mentions
- Social media hype cycles (meme-driven vs fundamental)
- Fear/greed indicators from community discussions
- Narrative shifts (risk-on → risk-off transitions)

Write a comprehensive sentiment report that includes:
- Overall sentiment assessment (Strongly Bullish / Bullish / Neutral / Bearish / Strongly Bearish)
- Confidence level in the assessment
- Key drivers behind the sentiment
- Notable risks or red flags
- How current sentiment compares to recent trends"""
            + """ Append a Markdown summary table: Source | Dominant Sentiment | Key Theme | Confidence"""
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "\n\n{instrument_context}"
                    "\n\nFor your reference, the current date is {current_date}.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "sentiment_report": report,
        }

    return crypto_sentiment_analyst_node
