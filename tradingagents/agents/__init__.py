from .utils.agent_utils import create_msg_delete
from .utils.agent_states import AgentState, InvestDebateState, RiskDebateState
from .utils.memory import FinancialSituationMemory

from .analysts.fundamentals_analyst import create_fundamentals_analyst
from .analysts.market_analyst import create_market_analyst
from .analysts.news_analyst import create_news_analyst
from .analysts.social_media_analyst import create_social_media_analyst
from .analysts.onchain_analyst import create_onchain_analyst
from .analysts.prediction_analyst import create_prediction_analyst
from .analysts.crypto_sentiment_analyst import create_crypto_sentiment_analyst

from .researchers.bear_researcher import create_bear_researcher
from .researchers.bull_researcher import create_bull_researcher

from .risk_mgmt.aggressive_debator import create_aggressive_debator
from .risk_mgmt.conservative_debator import create_conservative_debator
from .risk_mgmt.neutral_debator import create_neutral_debator

from .managers.research_manager import create_research_manager
from .managers.portfolio_manager import create_portfolio_manager

from .trader.trader import create_trader

# Phase 4 hybrid quant+LLM nodes
from .quant_signal_ingest import create_quant_signal_ingest
from .factual_agent import create_factual_agent
from .subjective_agent import create_subjective_agent
from .regime_reflector import create_regime_reflector
from .modulator import create_modulator

__all__ = [
    "FinancialSituationMemory",
    "AgentState",
    "create_msg_delete",
    "InvestDebateState",
    "RiskDebateState",
    "create_bear_researcher",
    "create_bull_researcher",
    "create_research_manager",
    "create_fundamentals_analyst",
    "create_market_analyst",
    "create_neutral_debator",
    "create_news_analyst",
    "create_aggressive_debator",
    "create_portfolio_manager",
    "create_conservative_debator",
    "create_social_media_analyst",
    "create_onchain_analyst",
    "create_prediction_analyst",
    "create_crypto_sentiment_analyst",
    "create_trader",
    "create_quant_signal_ingest",
    "create_factual_agent",
    "create_subjective_agent",
    "create_regime_reflector",
    "create_modulator",
]
