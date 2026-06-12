"""Reporting utilities for backtest results: tables, plots, JSON export."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from tradingagents.backtesting.engine import BacktestResult

logger = logging.getLogger(__name__)


def print_model_metrics(model_name: str, metrics: dict) -> None:
    """Print regression metrics for a single model evaluation."""
    print(f"  {model_name:15s}  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  "
          f"RMSE={metrics['rmse']:.2f}  MAPE={metrics['mape']:.4f}")


def print_summary_table(
    results: list[BacktestResult],
    buy_hold_return: Optional[float] = None,
) -> str:
    """Print a strategy comparison table. Returns the formatted string."""
    header = (
        f"{'Strategy':<20s} {'Tot.Ret.':>10s} {'Ann.Ret.':>10s} {'Sharpe':>8s} "
        f"{'MaxDD':>8s} {'WinRate':>8s} {'#Trades':>8s} {'PF':>8s}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for r in results:
        m = r.metrics
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
        lines.append(
            f"{r.strategy_name:<20s} {m['total_return']:>+10.2%} "
            f"{m['annualized_return']:>+10.2%} {m['sharpe_ratio']:>8.2f} "
            f"{m['max_drawdown']:>8.2%} {m['win_rate']:>8.1%} "
            f"{m['n_trades']:>8d} {pf:>8s}"
        )

    if buy_hold_return is not None:
        lines.append(
            f"{'Buy & Hold':<20s} {buy_hold_return:>+10.2%} "
            f"{'':>10s} {'':>8s} {'':>8s} {'':>8s} {'':>8s} {'':>8s}"
        )

    lines.append(sep)
    table_str = "\n".join(lines)
    print(table_str)
    return table_str


def plot_equity_curves(
    results: list[BacktestResult],
    output_path: Path | str,
    initial_capital: float = 10_000.0,
) -> None:
    """Plot equity curves for all strategies on one figure."""
    fig, ax = plt.subplots(figsize=(14, 6))

    for r in results:
        dates = pd.to_datetime(r.dates)
        equity = r.equity_curve[1:]  # align with trade dates
        ax.plot(dates, equity, linewidth=1.5, label=r.strategy_name)

    ax.axhline(y=initial_capital, color="gray", linestyle=":", linewidth=1,
               label="Initial Capital")

    ax.set_title("Backtest Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"Equity curve plot saved -> {output_path}")


def plot_predictions_vs_actuals(
    results: dict,
    output_path: Path | str,
) -> None:
    """Plot model predictions overlaid on actuals.

    Args:
        results: Dict mapping model key to ModelEvalResult.
        output_path: Path to save the plot PNG.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    actuals_plotted = False

    for key, res in results.items():
        df = res.result_df
        idx = df.index
        if hasattr(idx, "tz") and idx.tz is not None:
            idx = idx.tz_localize(None)

        if not actuals_plotted and "actual" in df.columns:
            ax.plot(idx, df["actual"], color="tab:orange", linewidth=1.5, label="Actual")
            actuals_plotted = True

        ax.plot(idx, df["prediction"], linewidth=1.5, linestyle="--",
                label=f"{res.model_name} prediction")

    ax.set_title("Model Predictions vs Actuals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    if hasattr(list(results.values())[0].result_df.index[0], "strftime"):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info(f"Predictions plot saved -> {output_path}")


def save_results_json(
    results: list[BacktestResult],
    output_path: Path | str,
    metadata: Optional[dict] = None,
) -> None:
    """Save backtest results to JSON for reproducibility."""
    data = {
        "metadata": metadata or {},
        "strategies": [],
    }
    for r in results:
        entry = {
            "strategy_name": r.strategy_name,
            "ticker": r.ticker,
            "metrics": r.metrics,
            "n_dates": len(r.dates),
            "date_range": {
                "start": str(r.dates[0]) if r.dates else None,
                "end": str(r.dates[-1]) if r.dates else None,
            },
        }
        data["strategies"].append(entry)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info(f"Results JSON saved -> {output_path}")
