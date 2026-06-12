from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import Annotated
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


# Module-level trade_date used by prediction tools to prevent look-ahead
# bias during backtesting.  Set via set_prediction_trade_date() before
# each propagate() call; defaults to None (= use datetime.now()).
_prediction_trade_date: str | None = None


def set_prediction_trade_date(trade_date: str | None) -> None:
    """Set the trade_date that prediction tools will use as their data boundary."""
    global _prediction_trade_date
    _prediction_trade_date = trade_date


@tool
def get_rf_forecast(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    lookback_days: Annotated[int, "Number of historical days to use for training (default: 300)"] = 300,
) -> str:
    """Run Random Forest price prediction with 95% confidence interval.

    Returns the predicted next-day price, confidence interval bounds,
    and the direction (up/down) relative to the current price.
    """
    try:
        from tradingagents.models.rf_model import forecast_next
        result = forecast_next(symbol, lookback_days, trade_date=_prediction_trade_date)
        return result
    except ImportError:
        return "Random Forest model not available. Run 'python scripts/train_models.py' to train models first."
    except Exception as e:
        return f"RF forecast error: {e}"


@tool
def get_lgb_forecast(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum', 'binancecoin')"],
    lookback_days: Annotated[int, "Number of historical days to use for training (default: 730)"] = 730,
) -> str:
    """Run LightGBM multi-horizon pooled prediction for h=7 and h=14.

    Automatically selects the optimal training pool based on the target coin:
    - For BTC/ETH: trains on 2-coin pool (BTC+ETH)
    - For altcoins: trains on 2+1 pool (BTC+ETH+target)

    Returns h=7 and h=14 price predictions with directional consensus and
    confidence level. This is the PRIMARY prediction signal — it achieved
    ~85% directional accuracy for BTC h=14 in walk-forward evaluation.
    """
    try:
        from tradingagents.models.lgb_model import forecast_next
        result = forecast_next(
            symbol, horizons=[7, 14], lookback_days=lookback_days,
            trade_date=_prediction_trade_date,
        )
        return result
    except ImportError:
        return "LightGBM model not available. Install lightgbm: pip install lightgbm"
    except Exception as e:
        return f"LGB forecast error: {e}"


@tool
def get_onchain_model_forecast(
    symbol: Annotated[str, "CoinGecko ID of the cryptocurrency (e.g., 'bitcoin', 'ethereum')"],
    lookback_days: Annotated[int, "Number of historical days for training (default: 300)"] = 300,
) -> str:
    """Run Gradient Boosting prediction using ONLY on-chain features.

    This model provides context about the predictive power of on-chain
    metrics alone. It does NOT drive trading decisions — use it to
    understand how on-chain signals relate to price movement.
    """
    try:
        from tradingagents.models.onchain_model import forecast_next
        result = forecast_next(symbol, lookback_days, trade_date=_prediction_trade_date)
        return result
    except ImportError:
        return "On-chain model not available. Run 'python scripts/train_models.py' to train models first."
    except Exception as e:
        return f"On-chain model forecast error: {e}"


def create_prediction_analyst(llm):

    def prediction_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_lgb_forecast,
            get_rf_forecast,
            get_onchain_model_forecast,
        ]

        system_message = (
            """You are a quantitative prediction model analyst. Your role is to run and interpret machine learning price forecasts for cryptocurrencies.

**Available Models (in order of importance):**

1. **LightGBM Multi-Horizon (PRIMARY)**: Pooled gradient boosting trained on the BTC+ETH pool (or BTC+ETH+target for altcoins — "2+1" pattern). Predicts prices at h=7 and h=14 days. Historical walk-forward directional accuracy: ~85% for BTC at h=14, ~76% for ETH, ~68% for altcoins like BNB. **This is the strongest signal — always call it first.** Note: h=1 daily predictions are NOT used (empirically ~50% DirAcc, indistinguishable from noise).

2. **Random Forest (SECONDARY)**: Single-coin 1000-tree ensemble with 95% confidence intervals. Useful as a cross-check on LGB direction; has lower DirAcc than LGB at long horizons but can confirm shorter-term trends.

3. **On-Chain Gradient Boosting (OBSERVATIONAL ONLY)**: Uses only on-chain features (funding rate, TVL, stablecoin supply). Provides context about on-chain signal strength. **Never use as primary trading signal.**

**Analysis Framework:**
1. **Always call `get_lgb_forecast` first** — this is the primary signal.
2. Look for horizon consensus:
   - Both h=7 and h=14 agree on direction → HIGH confidence
   - Only h=14 has a strong directional signal → MEDIUM confidence, trust h=14 (longer-term signal is more predictable in crypto)
   - Horizons disagree → LOW confidence
3. Optionally call `get_rf_forecast` to cross-check LGB direction on shorter horizons.
4. Optionally call `get_onchain_model_forecast` for on-chain context — do not use as primary signal.

**Confidence Levels (use exactly these labels):**
- HIGH: LGB h=7 and h=14 both agree AND predicted move at h=14 ≥ 2%
- MEDIUM: LGB horizons agree but magnitude < 2%, OR only h=14 is strongly directional
- LOW: LGB horizons disagree, OR LGB unavailable

**Key Considerations:**
- The LGB report already includes per-coin historical DirAcc — cite these numbers in your analysis.
- A prediction deviating >50% from current price may indicate data/model issues.
- RF predictions are informative but secondary; do not weigh them equally with LGB.

Write a detailed prediction report including:
- LGB h=7 and h=14 predictions with directional consensus
- Confidence level (HIGH/MEDIUM/LOW) with rationale
- RF cross-check (if called)
- On-chain context (if called) — observational only
- Any caveats about model reliability"""
            + """ Append a Markdown table: Model | Horizon | Prediction | Direction | Confidence | Notes"""
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
            "prediction_report": report,
        }

    return prediction_analyst_node
