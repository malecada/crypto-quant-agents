from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_funding_rates,
    get_tvl_metrics,
    get_stablecoin_metrics,
    get_gas_metrics,
    get_stablecoin_supply,
    get_onchain_pit,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


PIT_SYSTEM_MESSAGE = """You are a blockchain and on-chain data analyst.
Produce a point-in-time (PIT) safe on-chain assessment for the asset of
interest.

**Data access rule:** Call `get_onchain_pit` exactly once with the
coin's CoinGecko id and the current `trade_date` (passed to you below).
Do NOT call any other tool. The returned Markdown block contains values
with strict `as_of_ts <= trade_date` — no look-ahead.

**Metrics you will receive:**
- **Valuation regime**: MVRV, MVRV-Z (1y), Puell Multiple — cycle timing.
  Historical extremes: MVRV-Z ≤ -1.5 = accumulation zone, ≥ 2 = distribution.
- **Exchange flows**: 24h net USD flow + 30d z-score — supply on/off
  exchanges. Persistent outflows = bullish supply shock.
- **Network activity**: active addresses (24h + 30d z), tx count, hash
  rate (PoW).
- **DeFi / liquidity**: TVL per chain (ETH, BSC) and 7d pct change;
  global stablecoin market cap as a "dry powder" proxy.

**Report structure (Markdown):**
1. **Regime read**: MVRV-Z regime + Puell regime → one-line thesis.
2. **Flow read**: net flow + z-score interpretation → accumulation vs
   distribution stance.
3. **Activity read**: adoption trend (address + tx count vs 30d norm).
4. **Liquidity read**: TVL + stablecoin momentum → capital context.
5. **Divergences**: flag any conflict between price, flows, and valuation.
6. **Summary table**: Metric | Current Value | Regime/Trend |
   Signal (Bullish/Bearish/Neutral).

**BNB caveat:** CoinMetrics community tier does not cover BNB. If
coverage is thin (only TVL + stablecoin mcap), say so explicitly and
down-weight on-chain conviction for the BNB position.
"""

REALTIME_SYSTEM_MESSAGE_TEMPLATE = """You are a blockchain and on-chain data analyst specializing in cryptocurrency markets. Your role is to analyze on-chain metrics that reveal the underlying health and activity of blockchain networks and derivatives markets.

Use the available tools to gather and analyze the following categories of on-chain data:

**Derivatives Metrics (Binance Futures):**
- Funding rates: Indicate market leverage and directional bias. Positive = longs pay shorts (bullish bias). Extreme rates (>0.1% or <-0.1%) signal overleveraged positions and potential liquidation cascades.

**DeFi Metrics (DeFiLlama):**
- Total Value Locked (TVL): Measures capital deposited in DeFi protocols. Rising TVL = growing confidence and capital inflow. Falling TVL = capital flight.
- Stablecoin market cap: Indicates available liquidity. Growing supply = more dry powder for crypto purchases.
{web3_section}
**Analysis Framework:**
1. Start by fetching funding rates for the asset of interest
2. Get TVL metrics (both total and chain-specific if relevant)
3. Check stablecoin market cap for liquidity assessment
{web3_steps}. Synthesize findings into a coherent on-chain health assessment

You MUST call ALL available tools. Do not skip any tool.

Write a comprehensive report covering:
- Current market leverage conditions (funding rate analysis)
- DeFi capital flow trends (TVL momentum)
- Liquidity environment (stablecoin metrics)
{web3_bullet}
- Key risk signals from on-chain data
- How on-chain metrics support or contradict price action"""


def _is_pit_mode(config: dict) -> bool:
    """True when the operator has routed on-chain data to the PIT store."""
    vendors = config.get("data_vendors") or {}
    return vendors.get("onchain_data") == "onchain_pit"


def create_onchain_analyst(llm):

    def onchain_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])
        config = get_config()

        if _is_pit_mode(config):
            tools = [get_onchain_pit]
            system_message = PIT_SYSTEM_MESSAGE + get_language_instruction()
        else:
            tools = [get_funding_rates, get_tvl_metrics, get_stablecoin_metrics]
            has_web3 = bool(
                config.get("web3_provider_eth") or config.get("web3_provider_bsc")
            )
            if has_web3:
                tools.extend([get_gas_metrics, get_stablecoin_supply])
                web3_section = (
                    "\n**Network Metrics (Web3):**\n"
                    "- Gas prices: Network demand and congestion. High gas = heavy usage.\n"
                    "- Transaction counts: Network activity and adoption trends.\n"
                    "- Stablecoin supply on-chain: Capital availability on specific chains.\n"
                    "You MUST call get_gas_metrics and get_stablecoin_supply for both "
                    "'ethereum' and 'bsc' chains.\n"
                )
                web3_steps = "4. Get gas metrics and on-chain stablecoin supply for ethereum and bsc\n5"
                web3_bullet = "- Network activity indicators (gas prices, tx counts)"
            else:
                web3_section = (
                    "\n**Network Metrics:** Web3 RPC endpoints are not configured, so "
                    "gas and on-chain stablecoin supply data are unavailable for this run.\n"
                )
                web3_steps = "4"
                web3_bullet = ""
            system_message = REALTIME_SYSTEM_MESSAGE_TEMPLATE.format(
                web3_section=web3_section,
                web3_steps=web3_steps,
                web3_bullet=web3_bullet,
            )
            system_message += (
                " Append a Markdown summary table at the end with columns: "
                "Metric | Current Value | Trend | Signal (Bullish/Bearish/Neutral)."
            )
            system_message += get_language_instruction()

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
            "onchain_report": report,
        }

    return onchain_analyst_node
