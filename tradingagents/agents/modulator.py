"""LangGraph node: Layer 2 modulator (Tier A2).

Inserted between the Trader and Risk Debate. Takes the Trader's
investment plan + the Layer 1 ``QuantSignal`` and produces a
``ModulatedPosition`` via ``apply_modulator``. Self-MoA (N=5 samples
at T=0.5) gives the multiplier mean and uncertainty.

Multiplier parsing: prompt asks the LLM to emit a single line
``Multiplier: X.YZ`` where ``X.YZ ∈ [0.0, 1.5]``. The agent regex-extracts
each sample's number. Samples that fail to parse are dropped with a
warning so the wrapper fails open, not closed.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Optional

import numpy as np

from tradingagents.agents.utils.agent_utils import build_instrument_context
from tradingagents.agents.utils.anonymizer import unmask
from tradingagents.dataflows.beliefs_store import latest_belief
from tradingagents.llm_clients.multi_sample import MultiSampleCachedChatModel
from tradingagents.strategies.contracts import (
    ModulatedPosition,
    ModulatorOutput,
    QuantSignal,
)
from tradingagents.strategies.calibration import load_or_identity
from tradingagents.strategies.modulator import apply_modulator
from tradingagents.strategies.quant_signal_provider import get_active_quant_signal

logger = logging.getLogger(__name__)

_MULTIPLIER_RE = re.compile(r"multiplier\s*[:=]\s*([0-9]+\.?[0-9]*)", re.IGNORECASE)


def _extract_multiplier(text: str) -> Optional[float]:
    m = _MULTIPLIER_RE.search(text)
    if m is None:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if not (0.0 <= v <= 1.5):
        return None
    return v


def _build_prompt(
    coin_alias: str,
    quant_signal: QuantSignal,
    trader_plan: str,
    factual_report: str,
    subjective_report: str,
    regime_note: str,
    belief: str = "",
) -> list[dict]:
    pack = quant_signal.deterministic_signals
    det_block = "\n".join(
        f"- {k}: {pack.get(k)!r}"
        for k in (
            "lgb_h7", "lgb_h14", "ref_price", "lgb_confidence",
            "funding_z", "usdt_netflow", "ndf", "unlock_flag", "kimchi",
        )
        if k in pack
    )
    sys = (
        "You are the Layer 2 LLM modulator in a hybrid quant+LLM trading "
        "stack. Your job is NOT to pick a direction — the Layer 1 quant "
        "engine has already done that. Your job is to scale the Layer 1 "
        "position by a multiplier ∈ [0.0, 1.5] reflecting how much you "
        "trust the LLM-side qualitative signal versus the pure quant.\n\n"
        "Multiplier semantics:\n"
        "  0.0  — fully damp the LLM signal; trust quant only.\n"
        "  1.0  — neutral; LLM does not adjust the quant magnitude.\n"
        "  1.5  — amplify; LLM strongly corroborates the quant direction.\n\n"
        "Rules:\n"
        "1. Output exactly one line of the form 'Multiplier: X.YZ'.\n"
        "2. Then 2-3 sentences of narrative explaining the multiplier.\n"
        "3. Do NOT propose flipping the quant direction. If you believe "
        "the quant signal is wrong, return Multiplier: 0.0 and explain.\n"
        "4. The asset is intentionally referred to by an alias to reduce "
        "training-corpus bias. Treat it as one cryptocurrency among many."
    )
    belief_block = (
        f"\nLast week's investment belief (FinCon CVRF):\n{belief}\n"
        if belief else ""
    )
    user = (
        f"Asset: {coin_alias}\n"
        f"Layer 1 quant direction: {quant_signal.direction}\n"
        f"Layer 1 quant magnitude: {quant_signal.magnitude:+.3f}\n"
        f"Detected regime: {quant_signal.regime} "
        f"(confidence {quant_signal.regime_confidence:.2f}, "
        f"Hurst {quant_signal.hurst:.2f})\n"
        f"Deterministic signals:\n{det_block}\n\n"
        f"Regime reflector note: {regime_note}\n"
        f"{belief_block}\n"
        f"Factual analyst summary:\n{factual_report}\n\n"
        f"Subjective analyst summary:\n{subjective_report}\n\n"
        f"Trader's proposal:\n{trader_plan}\n\n"
        "Now output the multiplier line followed by 2-3 sentences."
    )
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def create_modulator(llm, n_samples: int = 5, temperature: float = 0.5):
    """Factory returning a LangGraph node that emits ``state['modulated_position']``."""

    sampler = MultiSampleCachedChatModel(
        delegate=llm, n=n_samples, temperature=temperature
    )

    def modulator_node(state, name):
        coin = state["company_of_interest"]
        trade_date = state.get("trade_date", "")
        coin_alias = build_instrument_context(coin)  # masked alias context block

        try:
            qs_in = state.get("quant_signal")
            if isinstance(qs_in, dict):
                quant_signal = QuantSignal(**qs_in)
            elif isinstance(qs_in, QuantSignal):
                quant_signal = qs_in
            else:
                quant_signal = get_active_quant_signal(coin, trade_date)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"modulator: get_active_quant_signal failed: {exc}")
            return {
                "modulated_position": None,
                "modulator_narrative": "Layer 1 unavailable — modulator skipped.",
            }

        trader_plan = state.get("trader_investment_plan", "")
        factual_report = state.get("factual_report", state.get("onchain_report", ""))
        subjective_report = state.get("subjective_report", state.get("sentiment_report", ""))
        regime_note = state.get("regime_reflector_note", "")

        # Use only the alias inside the prompt body so the LLM never sees raw name
        from tradingagents.agents.utils.anonymizer import is_enabled, mask
        coin_label = mask(coin) if is_enabled() else coin
        belief = latest_belief(coin) or ""
        messages = _build_prompt(
            coin_label, quant_signal, trader_plan,
            factual_report, subjective_report, regime_note, belief,
        )

        samples = sampler.sample_n(messages)
        multipliers = [
            v for v in (_extract_multiplier(s.content or "") for s in samples)
            if v is not None
        ]
        if not multipliers:
            logger.warning("modulator: no parseable multipliers; defaulting to 1.0")
            multipliers = [1.0]
        m_mean = float(np.mean(multipliers))
        m_std = float(np.std(multipliers, ddof=1)) if len(multipliers) > 1 else 0.0
        narrative = samples[0].content if samples else "no LLM response"

        # Tier B5: isotonic calibration on the verbalized confidence.
        raw_conf = max(0.0, min(1.0, 1.0 - m_std))
        cal = load_or_identity(coin)
        calibrated_conf = cal.transform(raw_conf)

        llm_output = ModulatorOutput(
            multiplier=m_mean,
            confidence=calibrated_conf,
            uncertainty=m_std,
            narrative=narrative[:500],
        )

        cfg = state.get("config", {})
        position: ModulatedPosition = apply_modulator(
            quant_signal, llm_output, config=cfg
        )

        # Re-attach real coin name in the audit trail so downstream PM/risk
        # decisions reference the actual asset.
        narr = unmask(position.narrative, coin)
        position = position.model_copy(update={"narrative": narr})

        return {
            "modulated_position": position.model_dump(mode="json"),
            "modulator_narrative": narr,
            "quant_signal": quant_signal.model_dump(mode="json"),
            "sender": name,
        }

    return functools.partial(modulator_node, name="Modulator")
