"""Skeptic-Quant: third debate agent (Tier B7).

Reads the Layer 1 ``QuantSignal`` plus the Bull and Bear arguments and
emits a regime-aware judgement on which side to trust. Fights two
known LLM trading pathologies:

1. FINSABER 2505.07078 — LLMs are "too timid in uptrends and too reckless
   in downturns." Skeptic-Quant explicitly cites the regime label so
   the debate aggregator sees a coherent third voice anchored to the
   quant signal.
2. SYCON-Bench 2505.23840 — third-person aggregator framing reduces
   sycophancy by ~64%. The Skeptic-Quant prompt is third-person and
   names quant evidence by feature, not by which persona endorsed it.

The agent does NOT propose a final trade; it scores who to trust on a
per-side basis. The Research Manager weighs that vote.
"""

from __future__ import annotations

import functools
import logging

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.anonymizer import is_enabled, mask

logger = logging.getLogger(__name__)


def create_skeptic_quant(llm):
    def skeptic_node(state, name):
        coin = state["company_of_interest"]
        coin_label = mask(coin) if is_enabled() else coin
        instrument_context = build_instrument_context(coin)

        qs = state.get("quant_signal") or {}
        regime = qs.get("regime", "unknown") if isinstance(qs, dict) else "unknown"
        direction = qs.get("direction", "flat") if isinstance(qs, dict) else "flat"
        magnitude = qs.get("magnitude", 0.0) if isinstance(qs, dict) else 0.0
        det = qs.get("deterministic_signals", {}) if isinstance(qs, dict) else {}

        debate = state.get("investment_debate_state", {}) or {}
        bull = debate.get("bull_history", "") or debate.get("current_response", "")
        bear = debate.get("bear_history", "")
        # Pull the most recent Bear turn explicitly if available
        bear_recent = bear.splitlines()[-1] if bear else ""

        sys = (
            "You are the Skeptic-Quant: an independent third voice in a "
            "Bull/Bear investment debate. You read the Layer 1 quant signal, "
            "the Bull arguments, and the Bear arguments, then issue a "
            "regime-conditional verdict on which side's claims are best "
            "supported by deterministic evidence.\n\n"
            "Output structure:\n"
            "1. Regime: <bull|sideways|bear>\n"
            "2. Quant-aligned side: <bull|bear|neither>\n"
            "3. 3-4 sentence justification citing specific deterministic "
            "signals (LGB consensus, funding Z, on-chain flow, regime).\n"
            "4. End with one line: 'Skeptic-Quant lean: <BULLISH|NEUTRAL|"
            "BEARISH>'.\n\n"
            "Do NOT echo bull or bear personas; refer to them in the third "
            "person ('the bull case', 'the bear case')."
        )
        user = (
            f"{instrument_context}\n\n"
            f"Asset alias: {coin_label}\n"
            f"Detected regime: {regime}\n"
            f"Layer 1 direction: {direction}, magnitude: {magnitude}\n"
            f"Layer 1 deterministic signals: {det}\n\n"
            f"Bull case (most recent turns):\n{bull[-3000:]}\n\n"
            f"Bear case (most recent turns):\n{bear[-3000:]}\n\n"
            "Issue your verdict now."
        )

        result = llm.invoke([
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ])
        content = result.content if hasattr(result, "content") else str(result)

        new_debate = dict(debate)
        new_debate["history"] = (
            new_debate.get("history", "") + "\n\n[Skeptic-Quant]\n" + content
        )
        new_debate["current_response"] = content
        new_debate["count"] = int(new_debate.get("count", 0)) + 1

        return {
            "messages": [result],
            "investment_debate_state": new_debate,
            "sender": name,
        }

    return functools.partial(skeptic_node, name="SkepticQuant")
