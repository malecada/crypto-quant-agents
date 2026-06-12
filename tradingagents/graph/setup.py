# TradingAgents/graph/setup.py

from typing import Any, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        bull_memory,
        bear_memory,
        trader_memory,
        invest_judge_memory,
        portfolio_manager_memory,
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.bull_memory = bull_memory
        self.bear_memory = bear_memory
        self.trader_memory = trader_memory
        self.invest_judge_memory = invest_judge_memory
        self.portfolio_manager_memory = portfolio_manager_memory
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "onchain" in selected_analysts:
            analyst_nodes["onchain"] = create_onchain_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["onchain"] = create_msg_delete()
            tool_nodes["onchain"] = self.tool_nodes["onchain"]

        if "prediction" in selected_analysts:
            analyst_nodes["prediction"] = create_prediction_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["prediction"] = create_msg_delete()
            tool_nodes["prediction"] = self.tool_nodes["prediction"]

        if "crypto_sentiment" in selected_analysts:
            analyst_nodes["crypto_sentiment"] = create_crypto_sentiment_analyst(
                self.quick_thinking_llm
            )
            delete_nodes["crypto_sentiment"] = create_msg_delete()
            tool_nodes["crypto_sentiment"] = self.tool_nodes["crypto_sentiment"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(
            self.quick_thinking_llm, self.bull_memory
        )
        bear_researcher_node = create_bear_researcher(
            self.quick_thinking_llm, self.bear_memory
        )
        research_manager_node = create_research_manager(
            self.deep_thinking_llm, self.invest_judge_memory
        )
        trader_node = create_trader(self.quick_thinking_llm, self.trader_memory)

        # Phase 4 hybrid quant+LLM modulator stack (asset-agnostic single path)
        quant_ingest_node = create_quant_signal_ingest()
        factual_node = create_factual_agent(self.quick_thinking_llm)
        subjective_node = create_subjective_agent(self.quick_thinking_llm)
        regime_reflector_node = create_regime_reflector()
        modulator_node = create_modulator(
            self.quick_thinking_llm, n_samples=5, temperature=0.5
        )

        # Phase 5 / Tier B7: Skeptic-Quant third debate agent
        from tradingagents.agents.researchers.skeptic_quant import create_skeptic_quant
        skeptic_quant_node = create_skeptic_quant(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(
            self.deep_thinking_llm, self.portfolio_manager_memory
        )

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            workflow.add_node(f"{analyst_type.capitalize()} Analyst", node)
            workflow.add_node(
                f"Msg Clear {analyst_type.capitalize()}", delete_nodes[analyst_type]
            )
            workflow.add_node(f"tools_{analyst_type}", tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Quant Signal Ingest", quant_ingest_node)
        workflow.add_node("Factual Agent", factual_node)
        workflow.add_node("Subjective Agent", subjective_node)
        workflow.add_node("Regime Reflector", regime_reflector_node)
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("SkepticQuant", skeptic_quant_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Modulator", modulator_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges — Layer 1 ingestion runs first, then analysts run as before.
        workflow.add_edge(START, "Quant Signal Ingest")
        first_analyst = selected_analysts[0]
        workflow.add_edge("Quant Signal Ingest", f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to the Factual/Subjective split when this
            # is the last analyst. Factual + Subjective run sequentially (LangGraph
            # parallel branches require fan-out edges; sequential keeps the state
            # contract simple). Then Regime Reflector deterministically scores them.
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Factual Agent")

        workflow.add_edge("Factual Agent", "Subjective Agent")
        workflow.add_edge("Subjective Agent", "Regime Reflector")
        workflow.add_edge("Regime Reflector", "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "SkepticQuant": "SkepticQuant",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "SkepticQuant": "SkepticQuant",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "SkepticQuant",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        # Phase 4 wiring: Trader → Modulator → Risk debate → PM
        workflow.add_edge("Trader", "Modulator")
        workflow.add_edge("Modulator", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        # Compile and return
        return workflow.compile()
