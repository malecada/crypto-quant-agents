#!/usr/bin/env python
"""Generate and cache agent signals for a coin list over a date range.

This is the expensive step (LLM calls). Runs separately from the backtest
so strategy parameters can be tuned without re-paying for signal generation.

Usage:
    # BTC+ETH, 90 days
    python scripts/generate_agent_signals.py \\
        --coins bitcoin ethereum \\
        --start 2024-05-01 --end 2024-08-01

    # Force regenerate (ignore cache)
    python scripts/generate_agent_signals.py \\
        --coins bitcoin --start 2024-05-01 --end 2024-05-10 --force
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate agent signals over a date range (caches to CSV per coin).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coins", nargs="+", required=True,
                    help="CoinGecko IDs (e.g. bitcoin ethereum binancecoin).")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    p.add_argument("--analysts", nargs="+",
                    default=["market", "onchain", "prediction"],
                    help="Analyst types to include.")
    p.add_argument("--sentiment-mode", choices=["live", "pit"], default="live",
                    help="Select sentiment vendor: 'live' (today-relative) or 'pit' (Alpaca PIT).")
    p.add_argument("--onchain-mode", choices=["live", "pit"], default="live",
                    help="Select on-chain vendor: 'live' (Binance/DefiLlama realtime) "
                         "or 'pit' (CoinMetrics bitemporal store).")
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--deep-think", default="gpt-5.4-mini")
    p.add_argument("--quick-think", default="gpt-5.4-nano")
    p.add_argument("--output-dir", default="data/agent_signals")
    p.add_argument("--force", action="store_true",
                    help="Ignore cached CSVs and regenerate.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    t0 = time.time()

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.backtesting.runner import generate_system_signals_v2

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["deep_think_llm"] = args.deep_think
    config["quick_think_llm"] = args.quick_think
    config["asset_class"] = "crypto"
    config["replay_cache"] = True
    if args.sentiment_mode == "pit":
        config["data_vendors"] = dict(config.get("data_vendors", {}))
        config["data_vendors"]["crypto_sentiment"] = "crypto_sentiment_pit"
        config["data_vendors"]["news_data"] = "news_data_pit"
    if args.onchain_mode == "pit":
        config["data_vendors"] = dict(config.get("data_vendors", {}))
        config["data_vendors"]["onchain_data"] = "onchain_pit"

    print(f"\n{'=' * 60}")
    print(f"  Agent Signal Generation")
    print(f"{'=' * 60}")
    print(f"  Coins     : {', '.join(args.coins)}")
    print(f"  Period    : {args.start} -> {args.end}")
    print(f"  Analysts  : {', '.join(args.analysts)}")
    print(f"  LLM       : {args.deep_think} / {args.quick_think}")
    print(f"  Sentiment : {args.sentiment_mode}")
    print(f"  On-chain  : {args.onchain_mode}")
    print(f"  Force run : {args.force}")
    print(f"  Output    : {args.output_dir}")
    print()

    results = generate_system_signals_v2(
        coins=args.coins,
        start_date=args.start,
        end_date=args.end,
        config=config,
        selected_analysts=args.analysts,
        output_dir=Path(args.output_dir),
        force_rerun=args.force,
    )

    print(f"\n{'=' * 60}")
    print(f"  Summary")
    print(f"{'=' * 60}")
    for coin, df in results.items():
        sig_counts = df["signal"].value_counts().to_dict()
        conf_counts = df["confidence"].value_counts().to_dict()
        print(f"  {coin}: {len(df)} signals  signals={sig_counts}  conf={conf_counts}")

    print(f"\n  Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
