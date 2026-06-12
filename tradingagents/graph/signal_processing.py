# TradingAgents/graph/signal_processing.py

from typing import Any


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted rating (BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, or SELL)
        """
        messages = [
            (
                "system",
                "You are an efficient assistant that extracts the trading decision from analyst reports. "
                "Extract the rating as exactly one of: BUY, OVERWEIGHT, HOLD, UNDERWEIGHT, SELL. "
                "Output only the single rating word, nothing else.",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content

    def extract_confidence(self, full_signal: str) -> str:
        """Extract a 0-100 confidence score or HIGH/MEDIUM/LOW label from trader output.

        The trader prompt (see tradingagents/agents/trader/trader.py) now asks
        for an explicit `Confidence: NN/100` line. This method first looks for
        that literal numeric score via regex. If found, it is returned as a
        zero-padded 3-digit string (e.g. "075"). Downstream consumers
        (backtest_system_v2.py) detect the numeric format and use it as a
        continuous [0, 1] confidence multiplier.

        If the numeric line is absent, the method falls back to the
        HIGH/MEDIUM/LOW rubric by delegating to the quick LLM as before —
        maintaining backward compatibility with P2 signals that were generated
        before the prompt change.

        Args:
            full_signal: Trader/portfolio-manager text.

        Returns:
            Either a 3-digit string "000"-"100" when a numeric confidence is
            emitted, or one of {"HIGH", "MEDIUM", "LOW", "UNKNOWN"} when
            falling back to rubric classification.
        """
        # Fast path: look for an explicit numeric line like "Confidence: 75/100"
        import re
        num_match = re.search(
            r"confidence\s*[:=]?\s*(\d{1,3})\s*(?:/\s*100)?",
            full_signal, re.IGNORECASE,
        )
        if num_match:
            score = int(num_match.group(1))
            if 0 <= score <= 100:
                return f"{score:03d}"

        messages = [
            (
                "system",
                "You are a trading-decision confidence rater. Read the trader's "
                "output and rate how confident the decision is, on the following "
                "rubric:\n"
                "  HIGH  — strong directional commitment, clear thesis, decisive "
                "language ('execute immediate', 'strong conviction', 'clear buy/sell signal'), "
                "few or no caveats. If the trader explicitly states 'Confidence: HIGH', return HIGH.\n"
                "  MEDIUM — clear lean in one direction but acknowledges meaningful "
                "counter-evidence or risks; recommends moderate sizing / staged entry. "
                "If the trader explicitly states 'Confidence: MEDIUM', return MEDIUM.\n"
                "  LOW   — hedged, HOLD with 'conflicting signals', 'monitor closely', "
                "'wait for confirmation', 'conservative position sizing', or a decision driven "
                "by uncertainty rather than evidence. If the trader explicitly states "
                "'Confidence: LOW', return LOW.\n"
                "Prefer an explicit label (HIGH/MEDIUM/LOW) when the trader provides one; "
                "otherwise infer from conviction strength per the rubric.\n"
                "Output exactly one word: HIGH, MEDIUM, or LOW. No other text.",
            ),
            ("human", full_signal),
        ]

        raw = self.quick_thinking_llm.invoke(messages).content
        cleaned = (raw or "").strip().upper()
        # Strip punctuation / markdown the LLM may have added
        for sep in (".", ",", "*", "`", ":", ";"):
            cleaned = cleaned.replace(sep, "")
        cleaned = cleaned.strip()
        # Handle cases like "HIGH CONFIDENCE" or "HIGH." robustly
        first = cleaned.split()[0] if cleaned else ""
        if first in {"HIGH", "MEDIUM", "LOW"}:
            return first
        return "UNKNOWN"
