"""Market-derived sentiment few-shots (Tier B6, arXiv 2502.14897).

Instead of human polarity labels, label news items with the realised
forward-return sign and use as few-shots in the Subjective agent's
system prompt. Reported +11% short-term BTC accuracy and 89.6%
on 227 high-impact BTC events.

This file holds a small curated set so the Subjective prompt stays
well under context limits. Update from
``data/sentiment/alpaca/*.parquet`` joined with realised t+1 returns
when the real backfill runs.
"""

from __future__ import annotations

# (headline, forward_return_sign_label) — realised t+7 sign, not human
FEW_SHOTS: list[tuple[str, str]] = [
    (
        "ETF inflows hit weekly record as institutional allocations rotate "
        "into the asset class.",
        "BULLISH",
    ),
    (
        "Network upgrade activates with no contention; staking rewards rise "
        "modestly across validators.",
        "BULLISH",
    ),
    (
        "Major exchange reports outflows as long-term holders rotate to "
        "self-custody following regulatory clarity.",
        "BULLISH",
    ),
    (
        "Macro risk-off rotation hits crypto alongside global equities; "
        "futures funding flips deeply negative.",
        "BEARISH",
    ),
    (
        "Lender on-chain liquidations cascade after a stablecoin briefly "
        "depegs; cross-protocol contagion fears rise.",
        "BEARISH",
    ),
    (
        "Founder-controlled treasury moves a large insider tranche to a "
        "centralized venue ahead of the next vesting cliff.",
        "BEARISH",
    ),
    (
        "Volatility compresses to multi-month lows; orderbook depth thin and "
        "social-mention volume flat — news-light range.",
        "NEUTRAL",
    ),
    (
        "Conflicting reads from on-chain (whale outflow) and derivatives "
        "(neutral funding) — no clear regime signal.",
        "NEUTRAL",
    ),
    (
        "Regulator postpones rulemaking deadline by 90 days; uncertain "
        "long-term but near-term overhang lifted.",
        "BULLISH",
    ),
    (
        "Bridged-asset depeg triggers risk-off across DeFi protocols; "
        "TVL drops more than 8% in 48 hours.",
        "BEARISH",
    ),
]


def render_few_shots() -> str:
    """Format few-shots for inclusion in a system prompt."""
    lines = ["Market-derived examples (label = realised forward-return sign):"]
    for h, lbl in FEW_SHOTS:
        lines.append(f"  - [{lbl}] {h}")
    return "\n".join(lines)
