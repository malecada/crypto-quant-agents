# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(self, max_debate_rounds=1, max_risk_discuss_rounds=1):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_onchain(self, state: AgentState):
        """Determine if on-chain analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_onchain"
        return "Msg Clear Onchain"

    def should_continue_prediction(self, state: AgentState):
        """Determine if prediction model analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_prediction"
        return "Msg Clear Prediction"

    def should_continue_crypto_sentiment(self, state: AgentState):
        """Determine if crypto sentiment analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_crypto_sentiment"
        return "Msg Clear Crypto_sentiment"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue.

        Phase 5 / Tier B7: 3-way rotation Bull → Bear → Skeptic-Quant
        with the Skeptic-Quant only speaking once per debate round.
        Rotation order is detected from the most recent persona prefix
        in ``current_response`` so this remains independent of which
        agent kicked the debate off.
        """
        debate = state["investment_debate_state"]
        if debate["count"] >= 3 * self.max_debate_rounds:
            return "Research Manager"
        last = (debate.get("current_response") or "").strip()
        # Bull just spoke → Bear next
        if last.startswith("Bull"):
            return "Bear Researcher"
        # Bear just spoke → Skeptic-Quant next (once per round)
        if last.startswith("Bear"):
            return "SkepticQuant"
        # Skeptic-Quant just spoke or starting fresh → Bull
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
