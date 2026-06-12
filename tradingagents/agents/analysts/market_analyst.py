from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_indicators,
    get_language_instruction,
    get_stock_data,
    get_crypto_data,
    get_crypto_indicators,
    get_crypto_indicators_batch,
)
from tradingagents.dataflows.config import get_config


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        # Select tools based on asset class
        config = get_config()
        asset_class = config.get("asset_class", "stock")

        if asset_class == "crypto":
            tools = [
                get_crypto_data,
                get_crypto_indicators_batch,  # PREFERRED — single call, full set
                get_crypto_indicators,        # fallback for unusual indicators
            ]
        else:
            tools = [
                get_stock_data,
                get_indicators,
            ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_30_sma: 30 SMA (REQUIRED — TREND FILTER): The baseline strategy uses 30-day SMA as the primary trend filter. Whether price is above or below this line drives position sizing (1.5x when aligned, 0.5x when against). Always compute this and report the trend regime explicitly.
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

**TOOL-CALL EFFICIENCY (IMPORTANT)**: For crypto, you MUST use `get_crypto_indicators_batch` (NOT `get_crypto_indicators`) as your first indicator-fetch call. The batch tool returns the complete standard indicator set — close_10_ema, close_50_sma, close_200_sma, rsi, macd, macds, mfi, boll, boll_ub, boll_lb, atr, vwma — in ONE call, eliminating the sequential per-indicator chain that dominates LLM cost. Only fall back to the single-indicator `get_crypto_indicators` tool if you need something the batch does not cover (e.g. close_30_sma, macdh).

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call the data retrieval tool first (get_stock_data for stocks, or get_crypto_data for cryptocurrencies) to retrieve the OHLCV data. Then use `get_crypto_indicators_batch` (preferred) or `get_indicators`/`get_crypto_indicators` (fallback) for any indicators not in the batch. Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions.

**REQUIRED — Trend Filter Analysis:**
After computing your selected indicators, ALWAYS include a dedicated "Trend Filter" section at the top of your report:
1. Compute `close_30_sma` via get_crypto_indicators (ALWAYS include this, regardless of your 8-indicator selection — it is required context for downstream sizing).
2. Report the current close price AND the current close_30_sma value.
3. State the trend regime explicitly:
   - If price > close_30_sma: "Bullish trend regime — downstream sizing should be 1.5x for longs, 0.5x for shorts."
   - If price < close_30_sma: "Bearish trend regime — downstream sizing should be 1.5x for shorts, 0.5x for longs."
4. Quantify the distance from the SMA (percent above/below) — a price very close to the SMA indicates a weak or transitioning trend.

This trend filter was the single highest-impact factor in our baseline strategy (Sharpe 1.88 → 2.69). The trader agent depends on this information to size positions correctly. Include it even if it seems redundant with other moving averages you select."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

        # Prompt structure ordered for OpenAI auto-prompt-caching: stable
        # global preamble → stable tool list → stable system_message →
        # per-coin instrument_context → per-day current_date. The first
        # ~3 sections are identical across every (coin, date) pair, so
        # OpenAI caches the ≥1024-token prefix automatically.
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
            "market_report": report,
        }

    return market_analyst_node
