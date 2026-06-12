"""Subjective analyst (FS-ReasoningAgent split, Tier A5).

Consumes only narrative / qualitative inputs: news, sentiment, social,
macro tone. No on-chain or technical-indicator data. Emits a concise
narrative report stored in ``state['subjective_report']``.

Dominant in sideways regimes per FS-ReasoningAgent (Li et al.
2410.12464): when there is no strong trend, narrative drives short-term
returns. The Regime Reflector picks the dominant lens for the current
regime and the Modulator weights the two views accordingly.
"""

from __future__ import annotations

import functools
import logging

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.anonymizer import is_enabled, mask
from tradingagents.agents.utils.labeled_examples import render_few_shots
from tradingagents.dataflows.cryptobert_local import sentiment as cryptobert_sentiment

logger = logging.getLogger(__name__)


def create_subjective_agent(llm):
    def subjective_node(state, name):
        coin = state["company_of_interest"]
        coin_label = mask(coin) if is_enabled() else coin
        instrument_context = build_instrument_context(coin)

        sentiment = state.get("sentiment_report", "")
        news = state.get("news_report", "")

        # Local CryptoBERT polarity baseline (Tier B6) — degrades to None
        # when transformers is not installed.
        bert_block = ""
        bert_in = (sentiment or "") + " " + (news or "")
        bert = cryptobert_sentiment(bert_in.strip()) if bert_in.strip() else None
        if bert is not None:
            bert_block = (
                f"\nCryptoBERT polarity (deterministic baseline): "
                f"{bert['polarity']:+.3f} ({bert['label']})\n"
            )

        sys = (
            "You are the Subjective analyst in a hybrid quant+LLM trading "
            "system. You read ONLY narrative data: news, sentiment, social, "
            "and macro tone. You do NOT read price, on-chain, or model "
            "predictions. Produce a 4-6 sentence subjective analysis ending "
            "with one of: 'Subjective stance: BULLISH | NEUTRAL | BEARISH' "
            "on its own line.\n\n"
            "Calibration anchor — when a market-derived example matches the "
            "current narrative pattern, lean toward the example's label.\n\n"
            f"{render_few_shots()}"
        )
        user = (
            f"{instrument_context}\n\n"
            f"Asset alias: {coin_label}\n"
            f"{bert_block}\n"
            f"Sentiment report:\n{sentiment or '(none)'}\n\n"
            f"News report:\n{news or '(none)'}\n\n"
            "Write the subjective analysis."
        )

        result = llm.invoke([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ])
        content = result.content if hasattr(result, "content") else str(result)
        return {
            "messages": [result],
            "subjective_report": content,
            "sender": name,
        }

    return functools.partial(subjective_node, name="Subjective")
