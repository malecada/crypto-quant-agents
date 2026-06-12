#!/usr/bin/env python
"""Run full multi-agent system backtest with strategy comparison.

Usage:
    # First run (expensive — generates signals via LLM calls):
    python scripts/backtest_system.py --coin bitcoin --start 2024-05-01 --end 2025-03-01

    # Reuse cached signals (free):
    python scripts/backtest_system.py --coin bitcoin --start 2024-05-01 --end 2025-03-01 \
        --signals-csv data/system_signals_bitcoin.csv
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Run multi-agent system backtest with strategy comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coin", required=True, help="CoinGecko ID.")
    p.add_argument("--start", required=True, help="Start date (YYYY-MM-DD).")
    p.add_argument("--end", required=True, help="End date (YYYY-MM-DD).")
    p.add_argument("--signals-csv", default=None, help="Pre-computed signals CSV.")
    p.add_argument("--analysts", nargs="+",
                    default=["market", "onchain", "prediction"],
                    help="Analyst types to include.")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.0005)
    p.add_argument("--short-cost", type=float, default=0.0003)
    p.add_argument("--llm-provider", default="openai")
    p.add_argument("--deep-think", default="gpt-5.4-mini")
    p.add_argument("--quick-think", default="gpt-5.4-nano")
    p.add_argument("--output-dir", default="data")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    t0 = time.time()

    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.backtesting.runner import run_system_backtest
    from tradingagents.backtesting.reporting import (
        print_summary_table,
        plot_equity_curves,
        save_results_json,
    )
    from tradingagents.models.model_utils import fetch_ohlcv_for_model

    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = args.llm_provider
    config["deep_think_llm"] = args.deep_think
    config["quick_think_llm"] = args.quick_think
    config["asset_class"] = "crypto"
    config["replay_cache"] = True

    output_dir = Path(args.output_dir)
    signals_csv = Path(args.signals_csv) if args.signals_csv else None

    print(f"\n{'=' * 60}")
    print(f"  System Backtest: {args.coin}")
    print(f"  Period: {args.start} -> {args.end}")
    print(f"  Analysts: {', '.join(args.analysts)}")
    print(f"  LLM: {args.deep_think} / {args.quick_think}")
    print(f"{'=' * 60}\n")

    results = run_system_backtest(
        coin=args.coin,
        start_date=args.start,
        end_date=args.end,
        config=config,
        selected_analysts=args.analysts,
        signals_csv=signals_csv,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        short_cost=args.short_cost,
        output_dir=output_dir,
    )

    if not results:
        print("ERROR: No backtest results produced.")
        sys.exit(1)

    # Buy & Hold benchmark
    lookback = (pd.to_datetime(args.end) - pd.to_datetime(args.start)).days + 30
    df_prices = fetch_ohlcv_for_model(args.coin, lookback, trade_date=args.end)
    if not df_prices.empty:
        prices = df_prices["prices"]
        bh_return = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
    else:
        bh_return = None

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Strategy Comparison")
    print(f"{'=' * 60}")
    print_summary_table(results, buy_hold_return=bh_return)

    print(f"\nCost assumptions:")
    print(f"  Fee per side   : {args.fee_rate:.2%}")
    print(f"  Slippage       : {args.slippage:.2%}")
    print(f"  Short cost/day : {args.short_cost:.2%}")

    n_days = len(results[0].dates) if results else 0
    if n_days < 100:
        print(f"\n  NOTE: Only {n_days} trading days. Annualized metrics may be unreliable.")

    # Plot
    plot_path = output_dir / f"backtest_equity_{args.coin}.png"
    plot_equity_curves(results, plot_path, args.initial_capital)
    print(f"Equity curve -> {plot_path}")

    # Save JSON
    json_path = output_dir / f"backtest_results_{args.coin}.json"
    save_results_json(results, json_path, metadata={
        "coin": args.coin, "start": args.start, "end": args.end,
        "analysts": args.analysts,
    })
    print(f"Results JSON -> {json_path}")

    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
