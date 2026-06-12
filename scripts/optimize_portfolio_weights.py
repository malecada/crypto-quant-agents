#!/usr/bin/env python
"""V5 MIX portfolio weight optimization.

§17 V5 MIX uses naive 50/50 equal-weight (BTC=V2-78f, ETH=V4-B-193f) → SR +2.50.
This script tests whether non-EW weights improve risk-adjusted return:

  1. Grid sweep — w_btc ∈ {0, 0.05, ..., 1.0}, full-sample SR per weight.
  2. Static in-sample optima — max-Sharpe (tangency) + min-variance closed forms.
  3. Walk-forward optimal — per-quarter expanding-window: estimate μ, Σ from
     all prior quarters, derive max-Sharpe weight, apply OOS to the next
     quarter. This is the only honest (non-look-ahead) optimized result.

All compared against the 50/50 EW baseline.

Usage:
    python scripts/optimize_portfolio_weights.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ANN = np.sqrt(252)
QUARTER_BARS = 63

BTC_SRC = "data/walkforward_v4_v2repro/daily_returns.csv"        # V2-78f
ETH_SRC = "data/walkforward_v4b_pit_noregime/daily_returns.csv"  # V4-B-193f


def _sharpe(r: np.ndarray) -> float:
    s = r.std()
    return float(r.mean() / s * ANN) if s > 0 else 0.0


def _port_metrics(r: np.ndarray) -> dict:
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    dd = float(((eq - peak) / peak).min())
    return {
        "sharpe": _sharpe(r),
        "total_return": float(eq[-1] - 1.0),
        "max_drawdown": dd,
        "ann_vol": float(r.std() * ANN),
    }


def _max_sharpe_weight(mu: np.ndarray, cov: np.ndarray) -> float:
    """2-asset tangency weight on BTC (rf=0). Clipped to [0, 1] (long-only)."""
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.5
    raw = inv @ mu
    s = raw.sum()
    if abs(s) < 1e-12:
        return 0.5
    w = raw / s
    return float(np.clip(w[0], 0.0, 1.0))


def _min_var_weight(cov: np.ndarray) -> float:
    """2-asset min-variance weight on BTC. Clipped to [0, 1]."""
    try:
        inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        return 0.5
    ones = np.ones(2)
    raw = inv @ ones
    s = raw.sum()
    if abs(s) < 1e-12:
        return 0.5
    w = raw / s
    return float(np.clip(w[0], 0.0, 1.0))


def main() -> None:
    btc = pd.read_csv(PROJECT_ROOT / BTC_SRC, parse_dates=["date"])
    eth = pd.read_csv(PROJECT_ROOT / ETH_SRC, parse_dates=["date"])
    btc = btc[btc["coin"] == "bitcoin"].set_index("date")["ret"].sort_index()
    eth = eth[eth["coin"] == "ethereum"].set_index("date")["ret"].sort_index()
    df = pd.DataFrame({"btc": btc, "eth": eth}).dropna().sort_index()
    logger.info("Aligned %d daily bars %s → %s", len(df), df.index.min().date(), df.index.max().date())

    R = df.values  # (n, 2): col 0 = btc, col 1 = eth
    n = len(R)

    print("\n" + "=" * 80)
    print("  V5 MIX PORTFOLIO WEIGHT OPTIMIZATION")
    print("=" * 80)

    # ── Baseline: 50/50 EW ────────────────────────────────────────────
    ew = 0.5 * R[:, 0] + 0.5 * R[:, 1]
    ew_m = _port_metrics(ew)
    print(f"\n  Baseline 50/50 EW:  SR={ew_m['sharpe']:+.3f}  ret={ew_m['total_return']:+.1%}  "
          f"maxDD={ew_m['max_drawdown']:.1%}  annVol={ew_m['ann_vol']:.1%}")

    # ── 1. Grid sweep (full-sample, in-sample) ────────────────────────
    print("\n  [1] Grid sweep — full-sample SR per BTC weight:")
    grid = []
    for w in np.arange(0.0, 1.0001, 0.05):
        r = w * R[:, 0] + (1 - w) * R[:, 1]
        m = _port_metrics(r)
        grid.append({"w_btc": float(w), **m})
    grid_df = pd.DataFrame(grid)
    best_grid = grid_df.loc[grid_df["sharpe"].idxmax()]
    for _, row in grid_df.iterrows():
        marker = "  ← best" if abs(row["w_btc"] - best_grid["w_btc"]) < 1e-9 else ""
        bar = "#" * int(row["sharpe"] / best_grid["sharpe"] * 40)
        print(f"    w_btc={row['w_btc']:.2f}  SR={row['sharpe']:+.3f}  ret={row['total_return']:+7.1%}  "
              f"DD={row['max_drawdown']:6.1%}  {bar}{marker}")

    # ── 2. Static in-sample optima (closed form) ──────────────────────
    mu = R.mean(axis=0)
    cov = np.cov(R, rowvar=False)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    w_ms = _max_sharpe_weight(mu, cov)
    w_mv = _min_var_weight(cov)
    ms_r = w_ms * R[:, 0] + (1 - w_ms) * R[:, 1]
    mv_r = w_mv * R[:, 0] + (1 - w_mv) * R[:, 1]
    ms_m = _port_metrics(ms_r)
    mv_m = _port_metrics(mv_r)
    print(f"\n  [2] Static in-sample optima  (BTC/ETH daily corr = {corr:+.3f})")
    print(f"    Max-Sharpe (tangency):  w_btc={w_ms:.3f}  SR={ms_m['sharpe']:+.3f}  "
          f"ret={ms_m['total_return']:+.1%}  maxDD={ms_m['max_drawdown']:.1%}")
    print(f"    Min-variance:           w_btc={w_mv:.3f}  SR={mv_m['sharpe']:+.3f}  "
          f"ret={mv_m['total_return']:+.1%}  maxDD={mv_m['max_drawdown']:.1%}")

    # ── 3. Walk-forward optimal (expanding-window, OOS) ───────────────
    # Per quarter q: use returns from bars [0, q_start) to estimate μ, Σ;
    # apply the resulting max-Sharpe weight to quarter q. First quarter uses
    # 50/50 (no prior data). This is the only look-ahead-free optimized run.
    print("\n  [3] Walk-forward optimal — expanding-window max-Sharpe, applied OOS:")
    wf_returns = np.empty(n)
    wf_returns[:] = np.nan
    q_starts = list(range(0, n, QUARTER_BARS))
    applied_weights = []
    for qi, q_start in enumerate(q_starts):
        q_end = min(q_start + QUARTER_BARS, n)
        if q_start == 0:
            w = 0.5  # cold start
        else:
            hist = R[:q_start]
            mu_h = hist.mean(axis=0)
            cov_h = np.cov(hist, rowvar=False)
            w = _max_sharpe_weight(mu_h, cov_h)
        applied_weights.append({"quarter_idx": qi, "q_start": q_start, "w_btc": w})
        wf_returns[q_start:q_end] = w * R[q_start:q_end, 0] + (1 - w) * R[q_start:q_end, 1]
    wf_m = _port_metrics(wf_returns)
    aw = pd.DataFrame(applied_weights)
    print(f"    Quarters: {len(q_starts)}   w_btc range=[{aw['w_btc'].min():.2f}, {aw['w_btc'].max():.2f}]  "
          f"mean={aw['w_btc'].mean():.2f}")
    print(f"    OOS result:  SR={wf_m['sharpe']:+.3f}  ret={wf_m['total_return']:+.1%}  "
          f"maxDD={wf_m['max_drawdown']:.1%}  annVol={wf_m['ann_vol']:.1%}")
    print(f"    Per-quarter applied w_btc: " +
          " ".join(f"{w:.2f}" for w in aw["w_btc"].values))

    # ── Summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    rows = [
        ("50/50 EW (baseline)",        0.50,  ew_m),
        (f"Grid best (w={best_grid['w_btc']:.2f}, in-sample)", best_grid["w_btc"],
         {"sharpe": best_grid["sharpe"], "total_return": best_grid["total_return"],
          "max_drawdown": best_grid["max_drawdown"], "ann_vol": best_grid.get("ann_vol", float("nan"))}),
        (f"Max-Sharpe (in-sample)",    w_ms,  ms_m),
        (f"Min-variance (in-sample)",  w_mv,  mv_m),
        ("Walk-forward optimal (OOS)", aw["w_btc"].mean(), wf_m),
    ]
    print(f"  {'strategy':<34} {'w_btc':>7} {'SR':>8} {'return':>10} {'maxDD':>8} {'annVol':>8}")
    for name, w, m in rows:
        print(f"  {name:<34} {w:>7.2f} {m['sharpe']:>+8.3f} {m['total_return']:>+9.1%} "
              f"{m['max_drawdown']:>7.1%} {m.get('ann_vol', float('nan')):>7.1%}")

    out = PROJECT_ROOT / "data" / "v5_weight_opt"
    out.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(out / "grid_sweep.csv", index=False)
    aw.to_csv(out / "walkforward_weights.csv", index=False)
    summary = {
        "ew_baseline": ew_m,
        "grid_best": {"w_btc": float(best_grid["w_btc"]), "sharpe": float(best_grid["sharpe"])},
        "max_sharpe_insample": {"w_btc": w_ms, **ms_m},
        "min_var_insample": {"w_btc": w_mv, **mv_m},
        "walkforward_optimal": {"mean_w_btc": float(aw["w_btc"].mean()), **wf_m},
        "btc_eth_corr": float(corr),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote: {out / 'grid_sweep.csv'}")
    print(f"  Wrote: {out / 'walkforward_weights.csv'}")
    print(f"  Wrote: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
