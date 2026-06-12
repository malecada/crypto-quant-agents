import functools

from tradingagents.agents.utils.agent_utils import build_instrument_context


def _build_momentum_context(company_name: str, trade_date: str) -> str:
    """Compute deterministic short-term momentum context from OHLCV at trade_date.

    Returns a short markdown block with SMA30 direction, 3-day and 7-day
    returns, and an RSI14 read. Injected into the trader prompt so the
    LLM has an explicit, non-negotiable short-term signal alongside the
    analyst reports.

    Runs on cached OHLCV (PIT-safe via `trade_date`). Graceful fallback
    message if data is unavailable — never raises.
    """
    try:
        import numpy as np
        from tradingagents.models.model_utils import fetch_ohlcv_for_model
        df = fetch_ohlcv_for_model(company_name, 60, trade_date=trade_date)
        if df.empty or "prices" not in df.columns:
            return "**Short-term momentum:** unavailable (no OHLCV)."
        p = df["prices"].astype(float).values
        if len(p) < 32:
            return "**Short-term momentum:** unavailable (insufficient history)."
        price_now = float(p[-1])
        sma30 = float(np.mean(p[-30:]))
        trend = "above SMA30 (bullish regime)" if price_now > sma30 else "below SMA30 (bearish regime)"
        ret_3d = (p[-1] / p[-4] - 1) * 100 if len(p) >= 4 else float("nan")
        ret_7d = (p[-1] / p[-8] - 1) * 100 if len(p) >= 8 else float("nan")
        # Simple RSI14
        deltas = np.diff(p[-15:])
        gains = np.where(deltas > 0, deltas, 0).sum()
        losses = np.where(deltas < 0, -deltas, 0).sum()
        rs = gains / losses if losses > 0 else float("inf")
        rsi = 100.0 if rs == float("inf") else 100 - 100 / (1 + rs)
        rsi_label = "overbought (>70)" if rsi > 70 else ("oversold (<30)" if rsi < 30 else "neutral")
        return (
            "**Short-term momentum (deterministic, PIT):**\n"
            f"- Price: {price_now:,.2f} vs SMA30 {sma30:,.2f} → {trend} ({(price_now/sma30 - 1)*100:+.2f}%)\n"
            f"- 3-day return: {ret_3d:+.2f}%\n"
            f"- 7-day return: {ret_7d:+.2f}%\n"
            f"- RSI14: {rsi:.1f} ({rsi_label})\n"
        )
    except Exception as e:
        return f"**Short-term momentum:** unavailable ({e})."


def create_trader(llm, memory):
    def trader_node(state, name):
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date", "")
        instrument_context = build_instrument_context(company_name)
        investment_plan = state["investment_plan"]
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        onchain_report = state.get("onchain_report", "")
        prediction_report = state.get("prediction_report", "")
        momentum_context = _build_momentum_context(company_name, trade_date)

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}\n\n{onchain_report}\n\n{prediction_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        if past_memories:
            for i, rec in enumerate(past_memories, 1):
                past_memory_str += rec["recommendation"] + "\n\n"
        else:
            past_memory_str = "No past memories found."

        context = {
            "role": "user",
            "content": f"Based on a comprehensive analysis by a team of analysts, here is an investment plan tailored for {company_name}. {instrument_context} This plan incorporates insights from current technical market trends, macroeconomic indicators, social media sentiment, on-chain analytics, and prediction model forecasts. Use this plan as a foundation for evaluating your next trading decision.\n\nProposed Investment Plan: {investment_plan}\n\nOn-chain analysis report: {onchain_report}\n\nPrediction model report: {prediction_report}\n\n{momentum_context}\nTreat the short-term momentum block as deterministic ground truth from the current price series — it is the highest-priority short-horizon signal after the LGB h=7/h=14 consensus.\n\nLeverage these insights, including prediction model outputs and confidence intervals, to make an informed and strategic decision.",
        }

        messages = [
            {
                "role": "system",
                "content": f"""You are a cryptocurrency trading agent analyzing market data to make investment decisions. Based on your analysis, provide a specific recommendation to buy, sell, or hold.

When evaluating the prediction and market reports, prioritize these signals (in order):

1. **LightGBM horizon consensus (PRIMARY SIGNAL)**: The prediction report includes h=7 and h=14 LGB forecasts.
   - If h=7 AND h=14 agree on direction AND the h=14 predicted move is ≥ 2% → HIGH confidence.
   - If only h=14 has a clear directional signal → MEDIUM confidence. Trust h=14 (85% historical DirAcc for BTC) over short-term signals.
   - If h=7 and h=14 disagree → LOW confidence, prefer HOLD.

2. **SMA30 trend alignment (POSITION SIZING CONTEXT)**: The market report's "Trend Filter" section tells you whether price is above or below the 30-day SMA.
   - Longs aligned with the bullish regime (price > SMA30) carry higher expected return.
   - Shorts aligned with the bearish regime (price < SMA30) carry higher expected return.
   - A trade AGAINST the SMA30 trend needs a stronger justification (e.g., extreme LGB consensus, clear reversal pattern).

3. **Cross-signal confirmation**: When LGB and Random Forest agree on direction, confidence rises. On-chain Gradient Boosting is observational only — do NOT use it as primary evidence.

**Confidence reporting (REQUIRED)**: Output your confidence as an integer score from 0 to 100 on its own line, formatted EXACTLY as:

    Confidence: NN/100

Where NN is your honest estimate of conviction strength:
- **85-100**: High conviction — strong multi-signal consensus (LGB h=7+h=14 agree, SMA30 trend aligned, on-chain/sentiment supportive), predicted move ≥ 2%, few caveats
- **65-84**: Solid conviction — clear lean in one direction, some supporting evidence, but one or more weaker cross-signals or mild caveats
- **40-64**: Mixed — clear signal on at least one axis but meaningful counter-evidence; would only size lightly
- **20-39**: Low conviction — predominantly HOLD reasoning, conflicting signals, "monitor closely" framing
- **0-19**: No conviction — insufficient data, contradictions dominate, clear wait-and-see regime

Calibrate the score to actual conviction — do NOT default to 50 when uncertain; pick 20-30 instead. Do NOT inflate past 85 without genuine multi-signal agreement.

End with a firm decision and always conclude your response with 'FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**' to confirm your recommendation. Apply lessons from past decisions to strengthen your analysis. Here are reflections from similar situations you traded in and the lessons learned: {past_memory_str}""",
            },
            context,
        ]

        result = llm.invoke(messages)

        return {
            "messages": [result],
            "trader_investment_plan": result.content,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
