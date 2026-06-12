"""Factual analyst (FS-ReasoningAgent split, Tier A5).

Consumes only deterministic / on-chain / derivatives / price data. No
news, no sentiment, no social. Emits a concise narrative report stored
in ``state['factual_report']`` plus a structured ``AnalystReport`` for
downstream Pydantic-aware nodes.

Dominant in bull regimes (per FS-ReasoningAgent Li et al. 2410.12464:
factual reasoning wins in bull markets, subjective wins in sideways /
narrative-driven phases). The Regime Reflector consumes both and emits
a re-weighting note that the modulator factors in.
"""

from __future__ import annotations

import functools
import logging

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.anonymizer import is_enabled, mask

logger = logging.getLogger(__name__)


def create_factual_agent(llm):
    def factual_node(state, name):
        coin = state["company_of_interest"]
        coin_label = mask(coin) if is_enabled() else coin
        instrument_context = build_instrument_context(coin)

        market = state.get("market_report", "")
        onchain = state.get("onchain_report", "")
        prediction = state.get("prediction_report", "")
        # Layer 1 QuantSignal is dict-serialized in state by the modulator
        # ingestion or upstream nodes
        qs = state.get("quant_signal") or {}

        det_block = ""
        if isinstance(qs, dict) and qs.get("deterministic_signals"):
            pack = qs["deterministic_signals"]
            det_block = "\n".join(
                f"- {k}: {v!r}" for k, v in pack.items()
            )

        sys = (
            "You are the Factual analyst in a hybrid quant+LLM trading "
            "system. You read ONLY deterministic data: technical indicators, "
            "on-chain metrics, derivatives positioning, and ML price "
            "predictions. You do NOT read news, social media, or sentiment. "
            "Produce a 4-6 sentence factual analysis ending with one of: "
            "'Factual stance: BULLISH | NEUTRAL | BEARISH' on its own line."
        )
        user = (
            f"{instrument_context}\n\n"
            f"Asset alias: {coin_label}\n\n"
            f"Market / technical report:\n{market}\n\n"
            f"On-chain report:\n{onchain}\n\n"
            f"Prediction model report:\n{prediction}\n\n"
            f"Layer 1 deterministic signals:\n{det_block or '(none)'}\n\n"
            "Write the factual analysis."
        )

        result = llm.invoke([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ])
        content = result.content if hasattr(result, "content") else str(result)
        return {
            "messages": [result],
            "factual_report": content,
            "sender": name,
        }

    return functools.partial(factual_node, name="Factual")
