#!/usr/bin/env python
"""Block-bootstrap 95% confidence interval for Sharpe ratio.

Replays V2 strategy on baseline and PIT prediction directories, gets
per-coin daily PnL, computes equal-weight portfolio returns, then runs
a stationary block bootstrap (Politis-Romano) to compute 95% CI for
Sharpe per variant and the difference distribution.

Usage:
    python scripts/bootstrap_sharpe.py \
        --baseline-dir data/multi_2c_5yr_baseline \
        --pit-dir       data/multi_2c_5yr_pit \
        --n-iter 5000 --block-size 21
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.regime_breakdown import replay_variant  # noqa: E402

DAILY_RF = (1 + 0.045) ** (1 / 252) - 1


def sharpe(returns: np.ndarray) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - DAILY_RF
    std = np.std(excess, ddof=1)
    return float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0


def stationary_bootstrap_sample(
    returns: np.ndarray, block_size: int, rng: np.random.Generator,
) -> np.ndarray:
    """Politis-Romano stationary bootstrap. Geometric block lengths with
    expected length = block_size. Wraps around."""
    n = len(returns)
    p = 1.0 / block_size
    out = np.empty(n)
    i = rng.integers(0, n)
    for t in range(n):
        out[t] = returns[i % n]
        if rng.random() < p:
            i = rng.integers(0, n)
        else:
            i += 1
    return out


def bootstrap_sharpe_ci(
    returns: np.ndarray, n_iter: int, block_size: int, seed: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    rng = np.random.default_rng(seed)
    point = sharpe(returns)
    samples = np.empty(n_iter)
    for k in range(n_iter):
        boot = stationary_bootstrap_sample(returns, block_size, rng)
        samples[k] = sharpe(boot)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return point, lo, hi, samples


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--pit-dir", required=True)
    p.add_argument("--n-iter", type=int, default=5000)
    p.add_argument("--block-size", type=int, default=21)
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading & replaying strategy on baseline + PIT predictions...")
    base = replay_variant(Path(args.baseline_dir), args.horizons)
    pit = replay_variant(Path(args.pit_dir), args.horizons)
    coins = sorted(set(base) & set(pit))

    # Build aligned per-coin daily returns. Mask out non-trade days
    # (position == 0) to match V2 strategy's reported Sharpe basis.
    aligned = None
    coin_returns = {}
    for coin in coins:
        bdf = base[coin].set_index("date")
        pdf = pit[coin].set_index("date")
        common = bdf.index.intersection(pdf.index)
        if aligned is None:
            aligned = common
        else:
            aligned = aligned.intersection(common)
        b_ret = bdf.loc[common]["daily_return"]
        p_ret = pdf.loc[common]["daily_return"]
        b_pos_mask = bdf.loc[common]["position"].abs() > 1e-9
        p_pos_mask = pdf.loc[common]["position"].abs() > 1e-9
        coin_returns[coin] = (b_ret, p_ret, b_pos_mask, p_pos_mask)

    print(f"\nN OOS days (aligned): {len(aligned)}")
    print(f"Bootstrap: {args.n_iter} iterations, block size {args.block_size}")
    print(f"\n{'=' * 78}\n  Per-coin Sharpe with 95% block-bootstrap CI\n{'=' * 78}")
    print(f"  {'coin':<10} {'point base':>11} {'CI base':>20} {'point PIT':>11} "
          f"{'CI PIT':>20} {'Δ':>7} {'p(PIT > base)':>14}")

    portfolio_base = np.zeros(len(aligned))
    portfolio_pit = np.zeros(len(aligned))
    for coin in coins:
        b_full, p_full, b_mask, p_mask = coin_returns[coin]
        b = b_full.loc[aligned].values
        p = p_full.loc[aligned].values
        bm = b_mask.loc[aligned].values
        pm = p_mask.loc[aligned].values
        portfolio_base += b / len(coins)
        portfolio_pit += p / len(coins)

        # Per-coin bootstrap on traded-days only (matches V2 reporting basis)
        b_pt, b_lo, b_hi, b_samp = bootstrap_sharpe_ci(b[bm], args.n_iter, args.block_size, seed=args.seed)
        p_pt, p_lo, p_hi, p_samp = bootstrap_sharpe_ci(p[pm], args.n_iter, args.block_size, seed=args.seed + 1)
        # Probability PIT Sharpe exceeds baseline (paired bootstrap on diff)
        diff_samp = p_samp - b_samp
        prob = float(np.mean(diff_samp > 0))
        print(
            f"  {coin:<10} {b_pt:>11.3f} [{b_lo:>+5.2f}, {b_hi:>+5.2f}]   "
            f"{p_pt:>11.3f} [{p_lo:>+5.2f}, {p_hi:>+5.2f}] {p_pt - b_pt:>+7.3f} {prob:>14.3f}"
        )

    print(f"\n{'=' * 78}\n  Portfolio (equal-weight) Sharpe with 95% block-bootstrap CI\n{'=' * 78}")
    b_pt, b_lo, b_hi, b_samp = bootstrap_sharpe_ci(portfolio_base, args.n_iter, args.block_size, seed=args.seed)
    p_pt, p_lo, p_hi, p_samp = bootstrap_sharpe_ci(portfolio_pit, args.n_iter, args.block_size, seed=args.seed + 1)
    diff_samp = p_samp - b_samp
    diff_lo = float(np.quantile(diff_samp, 0.025))
    diff_hi = float(np.quantile(diff_samp, 0.975))
    prob = float(np.mean(diff_samp > 0))
    print(f"  Baseline : {b_pt:.3f}  CI [{b_lo:+.2f}, {b_hi:+.2f}]")
    print(f"  +PIT     : {p_pt:.3f}  CI [{p_lo:+.2f}, {p_hi:+.2f}]")
    print(f"  Δ Sharpe : {p_pt - b_pt:+.3f}  paired CI [{diff_lo:+.3f}, {diff_hi:+.3f}]")
    print(f"  P(PIT > Baseline | bootstrap) = {prob:.3f}")
    if diff_lo > 0:
        print(f"  ✓ Δ Sharpe 95% CI excludes zero — PIT lift is statistically significant at 5%")
    elif diff_hi < 0:
        print(f"  ✗ PIT statistically WORSE")
    else:
        print(f"  ~ Δ Sharpe 95% CI includes zero — lift not statistically significant at 5%")


if __name__ == "__main__":
    main()
