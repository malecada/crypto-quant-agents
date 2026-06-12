"""Layer 1 ingestion node — populates ``state['quant_signal']``.

Pure plumbing: calls ``quant_engine.get_quant_signal(coin, date)`` and
writes the resulting ``QuantSignal`` to state as a dict so downstream
LangGraph nodes can JSON-serialise the state for replay.

Inserted at graph start, before any LLM analyst runs, so the Layer 1
quant baseline is available to every downstream agent (factual,
subjective, modulator).
"""

from __future__ import annotations

import functools
import logging

from tradingagents.strategies.contracts import QuantSignal
from tradingagents.strategies.quant_signal_provider import get_active_quant_signal

logger = logging.getLogger(__name__)


def create_quant_signal_ingest():
    def ingest_node(state, name):
        coin = state["company_of_interest"]
        trade_date = state.get("trade_date", "")
        try:
            sig: QuantSignal = get_active_quant_signal(coin, trade_date)
            return {
                "quant_signal": sig.model_dump(mode="json"),
                "sender": name,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Layer 1 ingest failed for {coin} @ {trade_date}: {exc}")
            return {"quant_signal": None, "sender": name}

    return functools.partial(ingest_node, name="QuantSignalIngest")
