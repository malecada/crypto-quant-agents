# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
    # Crypto-specific tools
    get_crypto_data,
    get_crypto_indicators,
    get_crypto_indicators_batch,
    get_funding_rates,
    get_tvl_metrics,
    get_stablecoin_metrics,
    get_gas_metrics,
    get_stablecoin_supply,
    get_reddit_posts,
    get_crypto_google_news,
)

# Prediction tools are defined in the prediction analyst module
from tradingagents.agents.analysts.prediction_analyst import (
    get_lgb_forecast,
    get_rf_forecast,
    get_onchain_model_forecast,
    set_prediction_trade_date,
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        deep_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["deep_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )
        quick_client = create_llm_client(
            provider=self.config["llm_provider"],
            model=self.config["quick_think_llm"],
            base_url=self.config.get("backend_url"),
            **llm_kwargs,
        )

        self.deep_thinking_llm = deep_client.get_llm()
        self.quick_thinking_llm = quick_client.get_llm()

        # Optionally wrap LLMs with replay cache for deterministic backtesting
        if self.config.get("replay_cache"):
            from tradingagents.llm_clients.replay_cache import CachedChatModel
            cache_db = self.config.get("replay_cache_db", "./data/llm_replay_cache.db")
            self.deep_thinking_llm = CachedChatModel(self.deep_thinking_llm, db_path=cache_db)
            self.quick_thinking_llm = CachedChatModel(self.quick_thinking_llm, db_path=cache_db)
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.portfolio_manager_memory = FinancialSituationMemory("portfolio_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
        )
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.portfolio_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator(max_recur_limit=self.config.get("max_recur_limit", 100))
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        asset_class = self.config.get("asset_class", "stock")

        nodes = {}

        # Market analyst tools depend on asset class
        if asset_class == "crypto":
            nodes["market"] = ToolNode([
                get_crypto_data,
                get_crypto_indicators_batch,  # preferred
                get_crypto_indicators,        # fallback
            ])
        else:
            nodes["market"] = ToolNode([get_stock_data, get_indicators])

        # Stock-specific analyst tools
        nodes["social"] = ToolNode([get_news])
        nodes["news"] = ToolNode([get_news, get_global_news, get_insider_transactions])
        nodes["fundamentals"] = ToolNode([
            get_fundamentals, get_balance_sheet, get_cashflow, get_income_statement,
        ])

        # Crypto-specific analyst tools — only include Web3 tools when configured
        onchain_tools = [get_funding_rates, get_tvl_metrics, get_stablecoin_metrics]
        has_web3 = bool(
            self.config.get("web3_provider_eth") or self.config.get("web3_provider_bsc")
        )
        if has_web3:
            onchain_tools.extend([get_gas_metrics, get_stablecoin_supply])
        nodes["onchain"] = ToolNode(onchain_tools)
        nodes["prediction"] = ToolNode([
            get_lgb_forecast, get_rf_forecast, get_onchain_model_forecast,
        ])
        nodes["crypto_sentiment"] = ToolNode([
            get_news, get_global_news, get_reddit_posts, get_crypto_google_news,
        ])

        return nodes

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""
        # Clear session cache from previous run to avoid stale data
        from tradingagents.dataflows.coingecko_binance import clear_session_cache
        clear_session_cache()

        # Reset asset-name anonymizer so multi-coin runs get stable but
        # propagate-scoped aliases (Tier A4 / Phase 3).
        from tradingagents.agents.utils.anonymizer import configure as _anon_configure
        _anon_configure(bool(self.config.get("anonymize_assets", False)))

        # Bind prediction models to this trade_date so they only see
        # data up to this point (prevents look-ahead bias in backtests).
        set_prediction_trade_date(str(trade_date))

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing
            trace = []
            last_printed_msg_id = None
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    last_msg = chunk["messages"][-1]
                    msg_id = getattr(last_msg, "id", None)
                    # Only print when messages actually change
                    if msg_id != last_printed_msg_id:
                        last_msg.pretty_print()
                        last_printed_msg_id = msg_id

                    # Print debate state changes
                    debate = chunk.get("investment_debate_state", {})
                    if debate.get("current_response") and debate.get("count", 0) > 0:
                        latest = debate["current_response"]
                        if latest.startswith(("Bull", "Bear")):
                            speaker = latest.split(":")[0] if ":" in latest else "Researcher"
                            print(f"\n{'='*30} {speaker} {'='*30}")
                            print(latest[:500] + ("..." if len(latest) > 500 else ""))
                    if debate.get("judge_decision") and not debate["judge_decision"].startswith(
                        trace[-1].get("investment_debate_state", {}).get("judge_decision", "NONE") if trace else "NONE"
                    ):
                        print(f"\n{'='*30} Research Manager Decision {'='*30}")
                        print(debate["judge_decision"][:500] + ("..." if len(debate["judge_decision"]) > 500 else ""))

                    risk = chunk.get("risk_debate_state", {})
                    if risk.get("latest_speaker") and risk.get("count", 0) > 0:
                        speaker_map = {
                            "Aggressive": risk.get("current_aggressive_response", ""),
                            "Conservative": risk.get("current_conservative_response", ""),
                            "Neutral": risk.get("current_neutral_response", ""),
                            "Judge": risk.get("judge_decision", ""),
                        }
                        speaker = risk["latest_speaker"]
                        content = speaker_map.get(speaker, "")
                        prev_risk = trace[-1].get("risk_debate_state", {}) if trace else {}
                        if content and content != prev_risk.get(f"current_{speaker.lower()}_response", prev_risk.get("judge_decision", "")):
                            print(f"\n{'='*30} {speaker} Risk Analyst {'='*30}")
                            print(content[:500] + ("..." if len(content) > 500 else ""))

                    trace.append(chunk)

            final_state = trace[-1]
        else:
            # Standard mode without tracing
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def propagate_with_confidence(self, company_name, trade_date):
        """Like `propagate()` but also returns a confidence label.

        Returns:
            (final_state, signal, confidence, trader_text) tuple where
            signal ∈ {BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL}
            confidence ∈ {HIGH, MEDIUM, LOW, UNKNOWN}
            trader_text is the raw final_trade_decision string for audit.
        """
        final_state, signal = self.propagate(company_name, trade_date)
        trader_text = final_state.get("final_trade_decision", "") or ""
        try:
            confidence = self.signal_processor.extract_confidence(trader_text)
        except Exception:
            confidence = "UNKNOWN"
        return final_state, signal, confidence, trader_text

    def propagate_with_modulator(self, company_name, trade_date):
        """Run the hybrid graph and return Phase 4 modulator output.

        Returns:
            (final_state, modulated_position, quant_signal, narrative)
            where modulated_position and quant_signal are dicts (or None
            if Layer 1 / Layer 2 failed for this row).
        """
        final_state, _signal = self.propagate(company_name, trade_date)
        return (
            final_state,
            final_state.get("modulated_position"),
            final_state.get("quant_signal"),
            final_state.get("modulator_narrative", "") or "",
        )

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state.get("market_report", ""),
            "sentiment_report": final_state.get("sentiment_report", ""),
            "news_report": final_state.get("news_report", ""),
            "fundamentals_report": final_state.get("fundamentals_report", ""),
            "onchain_report": final_state.get("onchain_report", ""),
            "prediction_report": final_state.get("prediction_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(self.config["results_dir"]) / self.ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_portfolio_manager(
            self.curr_state, returns_losses, self.portfolio_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
