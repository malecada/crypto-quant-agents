#!/usr/bin/env python
"""Side-by-side equity curve comparison for baseline vs +PIT.

Loads predictions from two directories, runs V2 strategy at default
config, plots:
  1. Per-coin equity curves (baseline + PIT on same axes)
  2. Portfolio equity curve (equal-weight)
  3. Drawdown comparison
  4. Annotated Sharpe / Return / MaxDD per regime (if BTC present)

Usage:
    python scripts/plot_pit_comparison.py \
        --baseline-dir data/multi_2c_5yr_baseline \
        --pit-dir       data/multi_2c_5yr_pit \
        --output        data/comparison_2c_5yr.png \
        --title "BTC+ETH 5.5yr OOS — V2 strategy"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regime_breakdown import replay_variant  # noqa: E402

INITIAL = 10_000.0


def equity_curve(daily_returns: np.ndarray) -> np.ndarray:
    eq = np.empty(len(daily_returns) + 1)
    eq[0] = INITIAL
    for i, r in enumerate(daily_returns, start=1):
        eq[i] = eq[i - 1] * (1 + r)
    return eq


def drawdown(eq: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(eq)
    return (peak - eq) / peak


def sharpe_of(daily_returns: np.ndarray, traded_mask: np.ndarray | None = None) -> float:
    if traded_mask is not None:
        sub = daily_returns[traded_mask]
    else:
        sub = daily_returns
    if len(sub) < 2:
        return 0.0
    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    excess = sub - daily_rf
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--pit-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--title", default="V2 Strategy — Baseline vs +PIT On-Chain")
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14])
    return p.parse_args()


def main():
    args = parse_args()
    base = replay_variant(Path(args.baseline_dir), args.horizons)
    pit = replay_variant(Path(args.pit_dir), args.horizons)
    coins = sorted(set(base) & set(pit))

    fig, axes = plt.subplots(
        nrows=len(coins) + 1, ncols=2,
        figsize=(13, 3.5 * (len(coins) + 1)),
        gridspec_kw={"width_ratios": [3, 1]},
    )
    if len(coins) + 1 == 1:
        axes = np.array([axes])

    fig.suptitle(args.title, fontsize=14, fontweight="bold")

    port_base_ret = None
    port_pit_ret = None

    for i, coin in enumerate(coins):
        bdf = base[coin].set_index("date")
        pdf = pit[coin].set_index("date")
        common = bdf.index.intersection(pdf.index)
        bdf, pdf = bdf.loc[common].sort_index(), pdf.loc[common].sort_index()

        b_ret = bdf["daily_return"].values
        p_ret = pdf["daily_return"].values
        b_eq = equity_curve(b_ret)
        p_eq = equity_curve(p_ret)

        b_pos = np.abs(bdf["position"].values) > 1e-9
        p_pos = np.abs(pdf["position"].values) > 1e-9
        b_sh = sharpe_of(b_ret, b_pos)
        p_sh = sharpe_of(p_ret, p_pos)

        b_ret_total = (b_eq[-1] / INITIAL - 1) * 100
        p_ret_total = (p_eq[-1] / INITIAL - 1) * 100
        b_dd = drawdown(b_eq).max() * 100
        p_dd = drawdown(p_eq).max() * 100

        ax = axes[i, 0]
        dates_plot = pd.to_datetime(common)
        # equity_curve length = len(returns) + 1; drop initial value to align with dates
        ax.semilogy(dates_plot, b_eq[1:], label=f"Baseline (Sharpe {b_sh:.2f})",
                    color="#888888", linewidth=1.5)
        ax.semilogy(dates_plot, p_eq[1:], label=f"+PIT (Sharpe {p_sh:.2f})",
                    color="#1f77b4", linewidth=1.8)
        ax.set_title(f"{coin.upper()} — Equity ($)")
        ax.set_ylabel("Equity (log scale)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        # Side panel: stats
        sax = axes[i, 1]
        sax.axis("off")
        text = (
            f"$\\bf{{Baseline}}$\n"
            f"Return: {b_ret_total:+.1f}%\n"
            f"Sharpe: {b_sh:.2f}\n"
            f"MaxDD:  {b_dd:.1f}%\n\n"
            f"$\\bf{{+PIT}}$\n"
            f"Return: {p_ret_total:+.1f}%\n"
            f"Sharpe: {p_sh:.2f}\n"
            f"MaxDD:  {p_dd:.1f}%\n\n"
            f"$\\bf{{Δ}}$\n"
            f"Return: {p_ret_total - b_ret_total:+.1f}pp\n"
            f"Sharpe: {p_sh - b_sh:+.2f}"
        )
        sax.text(0.0, 0.95, text, fontsize=10, family="monospace",
                 verticalalignment="top")

        if port_base_ret is None:
            port_base_ret = b_ret.copy()
            port_pit_ret = p_ret.copy()
            port_dates = common
        else:
            # Equal-weight portfolio: avg daily return
            port_base_ret = 0.5 * (port_base_ret + b_ret)
            port_pit_ret = 0.5 * (port_pit_ret + p_ret)

    # Portfolio panel
    if port_base_ret is not None and len(coins) > 1:
        b_eq = equity_curve(port_base_ret)
        p_eq = equity_curve(port_pit_ret)
        b_sh = sharpe_of(port_base_ret)
        p_sh = sharpe_of(port_pit_ret)
        b_ret_total = (b_eq[-1] / INITIAL - 1) * 100
        p_ret_total = (p_eq[-1] / INITIAL - 1) * 100
        b_dd = drawdown(b_eq).max() * 100
        p_dd = drawdown(p_eq).max() * 100

        ax = axes[-1, 0]
        dates_plot = pd.to_datetime(port_dates)
        ax.semilogy(dates_plot, b_eq[1:], label=f"Baseline (Sharpe {b_sh:.2f})",
                    color="#888888", linewidth=1.5)
        ax.semilogy(dates_plot, p_eq[1:], label=f"+PIT (Sharpe {p_sh:.2f})",
                    color="#d62728", linewidth=2.0)
        ax.set_title(f"Portfolio (equal-weight {len(coins)}-coin) — Equity ($)")
        ax.set_ylabel("Equity (log scale)")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

        sax = axes[-1, 1]
        sax.axis("off")
        text = (
            f"$\\bf{{Baseline}}$\n"
            f"Return: {b_ret_total:+.1f}%\n"
            f"Sharpe: {b_sh:.2f}\n"
            f"MaxDD:  {b_dd:.1f}%\n\n"
            f"$\\bf{{+PIT}}$\n"
            f"Return: {p_ret_total:+.1f}%\n"
            f"Sharpe: {p_sh:.2f}\n"
            f"MaxDD:  {p_dd:.1f}%\n\n"
            f"$\\bf{{Δ}}$\n"
            f"Return: {p_ret_total - b_ret_total:+.1f}pp\n"
            f"Sharpe: {p_sh - b_sh:+.2f}"
        )
        sax.text(0.0, 0.95, text, fontsize=10, family="monospace",
                 verticalalignment="top", color="#d62728")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
