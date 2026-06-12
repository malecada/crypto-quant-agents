"""Example usage of TradingAgents for cryptocurrency analysis.

This script demonstrates how to use the framework to analyze a cryptocurrency
using the multi-agent debate architecture with crypto-specific analysts.
"""

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create a custom config for crypto trading
config = DEFAULT_CONFIG.copy()

# LLM configuration
config["llm_provider"] = "openai"
config["deep_think_llm"] = "gpt-4o"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 1

# Asset class: "crypto" for cryptocurrencies, "stock" for equities
config["asset_class"] = "crypto"

# Initialize with crypto analysts
ta = TradingAgentsGraph(
    selected_analysts=["market", "onchain", "crypto_sentiment", "prediction"],
    debug=True,
    config=config,
)

# Analyze Bitcoin
_, decision = ta.propagate("bitcoin", "2026-04-09")
print(f"\nFinal decision: {decision}")

# Memorize mistakes and reflect (call after trade outcome is known)
# ta.reflect_and_remember(returns_losses=1000)


# --- Stock analysis example (for reference) ---
# config["asset_class"] = "stock"
# ta_stocks = TradingAgentsGraph(
#     selected_analysts=["market", "social", "news", "fundamentals"],
#     debug=True,
#     config=config,
# )
# _, decision = ta_stocks.propagate("NVDA", "2025-01-15")
