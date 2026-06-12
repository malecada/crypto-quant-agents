#!/usr/bin/env python
"""Analyze the LLM replay cache to find dedup opportunities.

The cache stores one row per unique (input, model) pair. By comparing the
total cache hit count (entries written) against the number of LLM
invocations during a known signal-gen run we can estimate the hit rate
and identify which agent/analyst prompts dominate.

Usage:
    python scripts/analyze_replay_cache.py
    python scripts/analyze_replay_cache.py --db ./data/llm_replay_cache.db --top 20
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path


PROMPT_FINGERPRINTS = [
    ("trader",            ["trader_node", "trader's transaction proposal", "FINAL TRANSACTION PROPOSAL"]),
    ("portfolio_mgr",     ["Portfolio Manager", "Rating: ", "Investment Thesis"]),
    ("market_analyst",    ["Market Analyst", "technical indicators", "stockstats", "RSI", "MACD"]),
    ("onchain_analyst",   ["onchain", "TVL", "funding rate", "DeFiLlama"]),
    ("sentiment_analyst", ["sentiment", "Alpaca News", "Benzinga", "Fear & Greed", "GDELT"]),
    ("prediction_analyst",["LightGBM", "h=7", "h=14", "ARIMA", "Random Forest"]),
    ("bull_researcher",   ["Bull Researcher", "investment thesis"]),
    ("bear_researcher",   ["Bear Researcher", "downside"]),
    ("aggressive",        ["Aggressive Analyst", "high-risk"]),
    ("conservative",      ["Conservative Analyst", "risk-averse"]),
    ("neutral",           ["Neutral Analyst", "balanced perspective"]),
    ("research_mgr",      ["Research Manager", "Investment Plan"]),
    ("confidence_parser", ["Confidence", "confidence score", "HIGH/MEDIUM/LOW"]),
]


def fingerprint(text: str) -> str:
    """Heuristic classify a cached response by the agent that produced it."""
    lo = text.lower()
    for label, hints in PROMPT_FINGERPRINTS:
        for h in hints:
            if h.lower() in lo:
                return label
    return "other"


def main():
    p = argparse.ArgumentParser(description="Replay-cache analyzer")
    p.add_argument("--db", default="./data/llm_replay_cache.db")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    if not Path(args.db).exists():
        raise SystemExit(f"cache db not found: {args.db}")

    con = sqlite3.connect(args.db)
    cur = con.cursor()

    total = cur.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
    by_model = dict(cur.execute("SELECT model, COUNT(*) FROM llm_cache GROUP BY model").fetchall())
    print(f"\n{'=' * 60}")
    print(f"  Replay-cache analysis: {args.db}")
    print(f"{'=' * 60}")
    print(f"  Total unique cached responses: {total}")
    print(f"  By model: {by_model}")
    print()

    # Classify by agent fingerprint
    bucket = Counter()
    sizes = {}
    sample_per_bucket: dict[str, str] = {}
    rows = cur.execute("SELECT cache_key, response_json FROM llm_cache").fetchall()
    for key, resp in rows:
        try:
            resp_obj = json.loads(resp)
            content = json.dumps(resp_obj)[:4000]
        except Exception:
            content = resp[:4000]
        label = fingerprint(content)
        bucket[label] += 1
        sizes.setdefault(label, []).append(len(resp))
        if label not in sample_per_bucket:
            sample_per_bucket[label] = content[:200]

    print(f"  By agent fingerprint:")
    for label, n in bucket.most_common(args.top):
        avg_size = sum(sizes[label]) / len(sizes[label])
        pct = n / total * 100
        print(f"    {label:<22} {n:>5}  ({pct:5.1f}%)   avg_size={avg_size/1024:.1f} KB")

    # Hit rate estimate: 90 days * 2 coins = 180 rows. Each row triggers
    # ~13-15 LLM calls (4 analysts + bull + bear + research_mgr + trader +
    # 3 risk + portfolio_mgr + confidence_parser). So expected unique
    # invocations ≈ 180 * 14 = 2,520.
    expected_invocations = 180 * 14
    hit_rate = 1 - (total / expected_invocations) if expected_invocations > total else 0
    print()
    print(f"  Hit-rate estimate (P3 BTC+ETH 90d × 14 agents = {expected_invocations} expected invocations):")
    print(f"    cached unique responses: {total}")
    print(f"    estimated hit rate     : {hit_rate*100:.1f}%")
    if total >= expected_invocations:
        print(f"    note: cached entries ≥ expected → multi-run accumulation, hit rate not meaningful from totals alone")

    # Identify likely dedup candidates: agents with low row count relative
    # to dates suggest reused prompts that don't depend on per-day state.
    print()
    print(f"  Per-coin-day uniqueness (lower = more dedup-able):")
    for label, n in bucket.most_common(args.top):
        if label == "other":
            continue
        per_coin_day = n / 180  # 180 unique (coin, date) pairs
        verdict = "DEDUPED" if per_coin_day < 0.5 else ("ok" if per_coin_day < 1.5 else "1:1")
        print(f"    {label:<22} {per_coin_day:5.2f} entries / coin-day  [{verdict}]")

    con.close()


if __name__ == "__main__":
    main()
