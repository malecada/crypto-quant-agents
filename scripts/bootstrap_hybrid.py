#!/usr/bin/env python
"""Statistical hardening battery for hybrid backtest results.

Implements BT1-BT5 from researche_materials/BACKTESTING_METHODOLOGY.md:

  BT1 — Block-bootstrap 95% CI for per-coin Sharpe (Ledoit-Wolf 2008)
  BT2 — Block-bootstrap CI for SR difference (hybrid vs baseline) on
        paired daily-return diff series
  BT3 — Deflated Sharpe Ratio (DSR; Bailey & Lopez de Prado 2014) +
        Probabilistic Sharpe Ratio (PSR) + Min Backtest Length (MinBTL)
  BT4 — Shuffled-signal placebo (permute LLM multiplier order on the
        hybrid position; rerun pnl; build null distribution)
  BT5 — Transaction cost sensitivity sweep (0/3/5/10/20 bps per side)

Reads ``daily_returns.csv`` produced by ``scripts/backtest_hybrid.py``.

Usage:
    python scripts/bootstrap_hybrid.py \
        --backtest-dir data/hybrid_signals_p1/backtest \
        --signals-dir  data/hybrid_signals_p1 \
        --n-iter 5000 --block-size 5 --n-trials 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the existing Politis-Romano stationary block bootstrap
from scripts.bootstrap_sharpe import (  # type: ignore  # noqa: E402
    DAILY_RF, sharpe, stationary_bootstrap_sample,
)
from scripts.baseline_strategy_v2 import run_coin_backtest  # type: ignore  # noqa: E402
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402


# ── BT1: Block bootstrap CI per series ──────────────────────────────


def block_bootstrap_ci(
    returns: np.ndarray, n_iter: int, block_size: int, seed: int = 42,
) -> tuple[float, float, float, np.ndarray]:
    """Politis-Romano stationary bootstrap. Returns (point, lo95, hi95, samples)."""
    rng = np.random.default_rng(seed)
    point = sharpe(returns)
    samples = np.empty(n_iter)
    for k in range(n_iter):
        samples[k] = sharpe(stationary_bootstrap_sample(returns, block_size, rng))
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    return point, lo, hi, samples


# ── BT2: SR-difference block bootstrap on paired diff series ────────


def diff_sharpe_ci(
    hybrid_ret: np.ndarray, baseline_ret: np.ndarray,
    n_iter: int, block_size: int, seed: int = 7,
) -> dict:
    """Bootstrap SR-of-difference and ΔSR with paired resampling.

    Aligned indices, so blocks are drawn from the same date pairs in
    both series (preserves contemporaneous correlation).
    """
    rng = np.random.default_rng(seed)
    n = min(len(hybrid_ret), len(baseline_ret))
    h = np.asarray(hybrid_ret[:n], dtype=float)
    b = np.asarray(baseline_ret[:n], dtype=float)
    diff = h - b

    sr_h0 = sharpe(h)
    sr_b0 = sharpe(b)
    sr_d0 = sharpe(diff)
    delta0 = sr_h0 - sr_b0

    diff_samples = np.empty(n_iter)
    delta_samples = np.empty(n_iter)
    p = 1.0 / block_size
    for k in range(n_iter):
        # Paired stationary block bootstrap on indices 0..n-1
        idx = np.empty(n, dtype=int)
        i = rng.integers(0, n)
        for t in range(n):
            idx[t] = i % n
            if rng.random() < p:
                i = rng.integers(0, n)
            else:
                i += 1
        h_b = h[idx]
        b_b = b[idx]
        diff_samples[k] = sharpe(h_b - b_b)
        delta_samples[k] = sharpe(h_b) - sharpe(b_b)

    return {
        "sr_hybrid": sr_h0,
        "sr_baseline": sr_b0,
        "sr_diff_series": sr_d0,
        "sr_diff_ci": [float(np.quantile(diff_samples, 0.025)),
                       float(np.quantile(diff_samples, 0.975))],
        "delta_sr_point": delta0,
        "delta_sr_ci": [float(np.quantile(delta_samples, 0.025)),
                        float(np.quantile(delta_samples, 0.975))],
        "p_delta_le_0": float((delta_samples <= 0).mean()),
    }


# ── BT3: PSR + DSR + MinBTL ─────────────────────────────────────────


def psr(sr_observed: float, sr_benchmark: float, T: int,
        skew_v: float, kurt_v: float) -> float:
    """Probabilistic Sharpe Ratio: P(SR_true > sr_benchmark)."""
    se = np.sqrt(
        (1 - skew_v * sr_observed + (kurt_v - 1) / 4 * sr_observed ** 2)
        / max(T - 1, 1)
    )
    if se <= 0 or np.isnan(se):
        return float("nan")
    return float(norm.cdf((sr_observed - sr_benchmark) / se))


def deflated_sharpe(
    sr_observed: float, sr_variance: float, n_trials: int,
    T: int, skew_v: float, kurt_v: float,
) -> tuple[float, float]:
    """Returns (DSR, expected-max-SR-under-null) per Bailey & López de Prado 2014."""
    if n_trials <= 1 or sr_variance <= 0:
        return float("nan"), 0.0
    gamma_em = 0.5772156649
    sr_0 = float(np.sqrt(sr_variance) * (
        (1 - gamma_em) * norm.ppf(1 - 1 / n_trials)
        + gamma_em * norm.ppf(1 - 1 / (n_trials * np.e))
    ))
    se = np.sqrt(
        (1 - skew_v * sr_observed + (kurt_v - 1) / 4 * sr_observed ** 2)
        / max(T - 1, 1)
    )
    if se <= 0 or np.isnan(se):
        return float("nan"), sr_0
    dsr = float(norm.cdf((sr_observed - sr_0) / se))
    return dsr, sr_0


def min_backtest_length(sr_observed: float, alpha: float = 0.05,
                        skew_v: float = 0.0, kurt_v: float = 3.0) -> float:
    """Minimum bars for SR > 0 to be significant at (1-alpha) confidence."""
    z = norm.ppf(1 - alpha)
    if sr_observed <= 0:
        return float("inf")
    daily = sr_observed / np.sqrt(252)
    se_unit = np.sqrt(1 - skew_v * daily + (kurt_v - 1) / 4 * daily ** 2)
    return float(1 + (z * se_unit / daily) ** 2)


# ── BT4: Shuffled-signal placebo ───────────────────────────────────


COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)


def _load_prices(coin: str, end_date: str) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coin, end_date)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def shuffled_signal_placebo(
    coin: str, signals_csv: Path, start: str, end: str, n_perms: int = 1000,
    seed: int = 42,
) -> dict:
    """Permute the hybrid `position` series across time and re-run pnl."""
    rng = np.random.default_rng(seed)
    sig = pd.read_csv(signals_csv, parse_dates=["date"])
    sig["date"] = sig["date"].dt.tz_localize(None).dt.normalize()
    sig = sig[(sig["date"] >= start) & (sig["date"] <= end)].copy()
    sig = sig.dropna(subset=["position"])
    sign = sig["quant_direction"].map({"long": 1, "short": -1, "flat": 0}).fillna(0)
    pos_mag = sign * sig["position"].abs()

    prices = _load_prices(coin, end)
    merged = sig.merge(prices[["Date", "Close"]], left_on="date", right_on="Date")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    base_pos = pos_mag.iloc[: len(merged)].values
    px = merged["Close"].astype(float).values
    dates = merged["date"].values

    # Observed
    _, observed_metrics = run_coin_backtest(
        dates=dates, prices=px, positions=base_pos,
        initial_capital=10_000.0, **COSTS,
    )
    sr_obs = float(observed_metrics.get("sharpe_ratio", float("nan")))

    # Null distribution
    perm_sr = np.empty(n_perms)
    for k in range(n_perms):
        perm = rng.permutation(base_pos)
        _, m = run_coin_backtest(
            dates=dates, prices=px, positions=perm,
            initial_capital=10_000.0, **COSTS,
        )
        perm_sr[k] = float(m.get("sharpe_ratio", 0.0))

    p_value = float((perm_sr >= sr_obs).mean())
    return {
        "sr_observed": sr_obs,
        "perm_mean": float(perm_sr.mean()),
        "perm_std": float(perm_sr.std(ddof=1)),
        "perm_p95": float(np.quantile(perm_sr, 0.95)),
        "perm_p99": float(np.quantile(perm_sr, 0.99)),
        "p_value": p_value,
        "n_perms": n_perms,
    }


# ── BT5: Cost sensitivity sweep ─────────────────────────────────────


def cost_sensitivity(
    coin: str, signals_csv: Path, start: str, end: str,
    bps_list: list[float] = (0.0, 3.0, 5.0, 10.0, 20.0),
) -> list[dict]:
    """Run hybrid backtest at multiple cost levels.

    Cost level (bps per side) → fee_rate + spread + slippage. We map
    bps to fee_rate while keeping price_impact + funding fixed.
    """
    sig = pd.read_csv(signals_csv, parse_dates=["date"])
    sig["date"] = sig["date"].dt.tz_localize(None).dt.normalize()
    sig = sig[(sig["date"] >= start) & (sig["date"] <= end)].copy()
    sig = sig.dropna(subset=["position"])
    sign = sig["quant_direction"].map({"long": 1, "short": -1, "flat": 0}).fillna(0)
    pos_mag = sign * sig["position"].abs()

    prices = _load_prices(coin, end)
    merged = sig.merge(prices[["Date", "Close"]], left_on="date", right_on="Date")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    base_pos = pos_mag.iloc[: len(merged)].values
    px = merged["Close"].astype(float).values
    dates = merged["date"].values

    out = []
    for bps in bps_list:
        bps_decimal = bps * 1e-4
        costs = dict(
            fee_rate=bps_decimal,
            slippage=bps_decimal,
            spread=bps_decimal / 2,
            price_impact=COSTS["price_impact"],
            funding_rate=COSTS["funding_rate"],
            stop_loss=COSTS["stop_loss"],
            max_portfolio_dd=COSTS["max_portfolio_dd"],
        )
        equity, m = run_coin_backtest(
            dates=dates, prices=px, positions=base_pos,
            initial_capital=10_000.0, **costs,
        )
        out.append({
            "bps_per_side": bps,
            "sharpe_ratio": float(m.get("sharpe_ratio", float("nan"))),
            "total_return": float(m.get("total_return", float("nan"))),
            "max_drawdown": float(m.get("max_drawdown", float("nan"))),
        })
    return out


# ── Main ────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backtest-dir", required=True,
                   help="Dir with daily_returns.csv from backtest_hybrid.py")
    p.add_argument("--signals-dir", required=True,
                   help="Dir with hybrid signal CSVs for placebo + cost sweep")
    p.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    p.add_argument("--start", default="2026-01-16")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--n-iter", type=int, default=5000,
                   help="Bootstrap iterations for BT1+BT2")
    p.add_argument("--block-size", type=int, default=5,
                   help="Politis-Romano expected block length")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of strategy variants tested for DSR deflation")
    p.add_argument("--n-perms", type=int, default=500,
                   help="Permutations for shuffled-signal placebo (BT4)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    bt_dir = Path(args.backtest_dir)
    sig_dir = Path(args.signals_dir)

    daily = pd.read_csv(bt_dir / "daily_returns.csv", parse_dates=["date"])
    summary = json.loads((bt_dir / "summary.json").read_text())

    print(f"\n{'=' * 78}")
    print(f"  Hybrid backtest hardening battery (BT1-BT5)")
    print(f"  Window: {args.start} -> {args.end}")
    print(f"{'=' * 78}\n")

    out: dict = {"_config": vars(args), "coins": {}}

    # Per-coin SR variance for DSR — collect across coins so DSR can use
    # cross-strategy variance proxy (small-sample workaround for σ_SR).
    sr_pool = []

    for coin in args.coins:
        sub = daily[daily["coin"] == coin].sort_values("date")
        h_ret = sub["hybrid_ret"].values
        b_ret = sub["baseline_ret"].values
        T = int(len(h_ret))
        if T < 2:
            continue

        # BT1: per-strategy Sharpe CIs
        h_pt, h_lo, h_hi, _ = block_bootstrap_ci(h_ret, args.n_iter, args.block_size,
                                                 args.seed)
        b_pt, b_lo, b_hi, _ = block_bootstrap_ci(b_ret, args.n_iter, args.block_size,
                                                 args.seed + 1)
        sr_pool.extend([h_pt, b_pt])

        # BT2: SR difference CI
        diff = diff_sharpe_ci(h_ret, b_ret, args.n_iter, args.block_size,
                              args.seed + 2)

        # BT3: PSR + DSR + MinBTL on hybrid
        sk_h = float(skew(h_ret, bias=False)) if T > 3 else 0.0
        kt_h = float(kurtosis(h_ret, bias=False, fisher=False)) if T > 3 else 3.0
        psr_h_gt0 = psr(h_pt, 0.0, T, sk_h, kt_h)
        psr_h_gt1 = psr(h_pt, 1.0, T, sk_h, kt_h)
        psr_h_gt2 = psr(h_pt, 2.0, T, sk_h, kt_h)
        minbtl_h = min_backtest_length(h_pt, 0.05, sk_h, kt_h)

        sk_b = float(skew(b_ret, bias=False)) if T > 3 else 0.0
        kt_b = float(kurtosis(b_ret, bias=False, fisher=False)) if T > 3 else 3.0
        psr_b_gt0 = psr(b_pt, 0.0, T, sk_b, kt_b)
        minbtl_b = min_backtest_length(b_pt, 0.05, sk_b, kt_b)

        # BT4: Shuffled-signal placebo
        signals_csv = sig_dir / f"{coin}_{args.start}_{args.end}.csv"
        placebo = (
            shuffled_signal_placebo(coin, signals_csv, args.start, args.end,
                                    args.n_perms, args.seed + 3)
            if signals_csv.exists() else None
        )

        # BT5: cost sensitivity
        cost_sweep = (
            cost_sensitivity(coin, signals_csv, args.start, args.end)
            if signals_csv.exists() else None
        )

        out["coins"][coin] = {
            "T": T,
            "hybrid": {"sharpe": h_pt, "ci_95": [h_lo, h_hi],
                       "skew": sk_h, "excess_kurt": kt_h - 3,
                       "psr_gt0": psr_h_gt0, "psr_gt1": psr_h_gt1,
                       "psr_gt2": psr_h_gt2, "min_btl_alpha05": minbtl_h},
            "baseline": {"sharpe": b_pt, "ci_95": [b_lo, b_hi],
                         "skew": sk_b, "excess_kurt": kt_b - 3,
                         "psr_gt0": psr_b_gt0, "min_btl_alpha05": minbtl_b},
            "comparison": diff,
            "placebo_shuffled_signal": placebo,
            "cost_sensitivity": cost_sweep,
        }

        print(f"  {coin}:  T={T}")
        print(f"    Hybrid    SR={h_pt:>6.3f}  CI95=[{h_lo:>6.3f},{h_hi:>6.3f}]  "
              f"skew={sk_h:+.2f} ex.kurt={kt_h-3:+.2f}")
        print(f"              PSR>0={psr_h_gt0:.3f}  PSR>1={psr_h_gt1:.3f}  "
              f"PSR>2={psr_h_gt2:.3f}  MinBTL={minbtl_h:.0f}")
        print(f"    Baseline  SR={b_pt:>6.3f}  CI95=[{b_lo:>6.3f},{b_hi:>6.3f}]  "
              f"PSR>0={psr_b_gt0:.3f}")
        print(f"    ΔSR       point={diff['delta_sr_point']:+.3f}  "
              f"CI95=[{diff['delta_sr_ci'][0]:+.3f},{diff['delta_sr_ci'][1]:+.3f}]"
              f"  P(ΔSR≤0)={diff['p_delta_le_0']:.3f}")
        if placebo:
            print(f"    Placebo   obs={placebo['sr_observed']:>6.3f}  "
                  f"null_mean={placebo['perm_mean']:+.2f}  "
                  f"null_p95={placebo['perm_p95']:+.2f}  "
                  f"p_value={placebo['p_value']:.3f}  "
                  f"({placebo['n_perms']} perms)")
        if cost_sweep:
            print(f"    Costs     "
                  + "  ".join(f"{c['bps_per_side']:>3.0f}bps→SR={c['sharpe_ratio']:+.2f}"
                              for c in cost_sweep))
        print()

    # BT3 deflation across the trial pool — variance across coins/strategies
    if len(sr_pool) >= 2:
        sr_var = float(np.var(sr_pool, ddof=1))
        for coin, payload in out["coins"].items():
            T = payload["T"]
            sk_h = payload["hybrid"]["skew"]
            kt_h = payload["hybrid"]["excess_kurt"] + 3
            sr_h = payload["hybrid"]["sharpe"]
            dsr, sr_0 = deflated_sharpe(sr_h, sr_var, args.n_trials, T, sk_h, kt_h)
            payload["hybrid"]["dsr"] = dsr
            payload["hybrid"]["expected_max_sr_under_null"] = sr_0
            print(f"  DSR ({coin}):  hybrid={sr_h:.3f}  "
                  f"E[max SR | null]={sr_0:.3f}  DSR={dsr:.3f}  "
                  f"(N_trials={args.n_trials})")
        out["sr_pool_variance"] = sr_var

    out_path = bt_dir / "hardening.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Wrote: {out_path}")


if __name__ == "__main__":
    main()
