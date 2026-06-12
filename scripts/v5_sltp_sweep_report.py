#!/usr/bin/env python
"""Generate top-20 markdown + 12 heatmap PNGs from V5 MIX TP/SL sweep results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE = dict(sl=0.03, ee=0.015, tp=0.0)


def _heatmap(pivot: pd.DataFrame, title: str, cbar_label: str,
             out: Path, baseline: tuple[float, float] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    arr = pivot.values
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{r:g}" for r in pivot.index])
    ax.set_xlabel(pivot.columns.name)
    ax.set_ylabel(pivot.index.name)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=cbar_label)

    # Annotate each cell with value
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.2f}",
                    ha="center", va="center", fontsize=7, color="black")

    if baseline is not None:
        sl_b, tp_b = baseline
        if sl_b in pivot.index and tp_b in pivot.columns:
            yi = list(pivot.index).index(sl_b)
            xi = list(pivot.columns).index(tp_b)
            ax.plot(xi, yi, marker="x", markersize=18, mew=3, color="blue",
                    label="V5 baseline")
            ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", default="data/v5_sltp_sweep")
    args = p.parse_args()

    in_dir = PROJECT_ROOT / args.input_dir
    df = pd.read_csv(in_dir / "results.csv")
    port = df[df["scope"] == "portfolio"].copy()

    # ── Top-20 markdown ────────────────────────────────────────────────
    top = port.sort_values("sharpe", ascending=False).head(20).copy()
    baseline_rows = port[(port["sl"] == 0.03) & (port["ee"] == 0.015) & (port["tp"] == 0.0)]
    baseline_sr = float(baseline_rows["sharpe"].iloc[0]) if len(baseline_rows) else float("nan")
    lines = [
        "# V5 MIX TP/SL Sweep — Top 20 Cells (by portfolio Sharpe)",
        "",
        f"Source: `{in_dir / 'results.csv'}` ({len(port)} portfolio cells)",
        "",
        f"Baseline V5 cell: SL=0.03, EE=0.015, TP=off → SR = {baseline_sr:+.3f}",
        "",
        "| Rank | SL | EE | TP | Sharpe | Total Ret | Max DD | Calmar | Win % | PF |",
        "|------|-----|-----|-----|--------|-----------|--------|--------|-------|-----|",
    ]
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        is_baseline = (r["sl"] == 0.03 and r["ee"] == 0.015 and r["tp"] == 0.0)
        marker = " ← **baseline**" if is_baseline else ""
        lines.append(
            f"| {rank} | {r['sl']:g} | {r['ee']:g} | {r['tp']:g} | "
            f"{r['sharpe']:+.3f}{marker} | {r['total_return']:+.1%} | "
            f"{r['max_drawdown']:.1%} | {r['calmar']:+.2f} | "
            f"{r['win_rate']:.1%} | {r['profit_factor']:.2f} |"
        )
    (in_dir / "top20.md").write_text("\n".join(lines) + "\n")
    print(f"  Wrote: {in_dir / 'top20.md'}")

    # ── Heatmaps: per EE, SR(SL × TP) + DD(SL × TP) ────────────────────
    heat_dir = in_dir / "heatmaps"
    heat_dir.mkdir(exist_ok=True)
    n = 0
    for ee in sorted(port["ee"].unique()):
        sub = port[port["ee"] == ee]
        for metric, label in [
            ("sharpe", "Portfolio Sharpe"),
            ("max_drawdown", "Max Drawdown"),
        ]:
            pivot = sub.pivot(index="sl", columns="tp", values=metric)
            pivot.index.name = "stop_loss"
            pivot.columns.name = "take_profit"
            title = f"V5 MIX {label}  (early_exit_loss = {ee:g})"
            out = heat_dir / f"{metric}_sl_x_tp__ee_{ee:g}.png"
            baseline = (BASELINE["sl"], BASELINE["tp"]) if ee == BASELINE["ee"] else None
            _heatmap(pivot, title, label, out, baseline=baseline)
            n += 1
    print(f"  Wrote: {n} heatmaps to {heat_dir}")


if __name__ == "__main__":
    main()
