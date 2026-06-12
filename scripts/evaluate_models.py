#!/usr/bin/env python
"""Evaluate RF / ARIMA / GBR models via walk-forward and save dated predictions.

Usage:
    python scripts/evaluate_models.py --coin bitcoin --days 730
    python scripts/evaluate_models.py --coin bitcoin --days 365 --models rf arima
    python scripts/evaluate_models.py --coin ethereum --min-train 360
"""

import argparse
import logging
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate prediction models via walk-forward backtesting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coin", required=True, help="CoinGecko ID, e.g. 'bitcoin'.")
    p.add_argument("--days", type=int, default=730, help="Lookback days for data.")
    p.add_argument("--trade-date", default=None, help="Upper date bound (YYYY-MM-DD).")
    p.add_argument("--models", nargs="+", default=["rf", "arima"],
                    choices=["rf", "arima", "gbr"], help="Models to evaluate.")
    p.add_argument("--min-train", type=int, default=None,
                    help="Min training window for walk-forward eval.")
    p.add_argument("--output-dir", default="data", help="Output directory.")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    t0 = time.time()

    from tradingagents.backtesting.runner import evaluate_models
    from tradingagents.backtesting.reporting import (
        print_model_metrics,
        plot_predictions_vs_actuals,
    )

    output_dir = Path(args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"  Model Evaluation: {args.coin} ({args.days} days)")
    print(f"{'=' * 60}\n")

    results = evaluate_models(
        coin=args.coin,
        lookback_days=args.days,
        trade_date=args.trade_date,
        min_train_window=args.min_train,
        models=args.models,
        output_dir=output_dir,
    )

    if not results:
        print("ERROR: No models evaluated successfully.")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward Evaluation Results")
    print(f"{'=' * 60}")
    for key, res in results.items():
        print_model_metrics(res.model_name, res.metrics)

    # Plot predictions vs actuals
    plot_path = output_dir / "eval_predictions_plot.png"
    plot_predictions_vs_actuals(results, plot_path)
    print(f"\nPlot saved -> {plot_path}")

    # Summary
    csv_path = output_dir / "eval_predictions.csv"
    print(f"Predictions CSV -> {csv_path}")
    print(f"\nTotal runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
