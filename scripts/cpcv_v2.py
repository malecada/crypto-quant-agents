#!/usr/bin/env python
"""BT9 — Combinatorial Purged CV (CPCV) on V2 strategy layer.

Per BACKTESTING_METHODOLOGY.md §2.3 — applies CPCV to the
**downstream** combination (signals + V2 sizing) holding the LGB
walk-forward predictions fixed (already trained per-bar in the
existing preds_lgb_h{7,14}.csv).

Group structure: divide the walk-forward bars into N=10 sequential
groups; pick k=2 groups per test set; C(10,2)=45 combinations.

Per split:
  1. Concatenate the chosen test groups in chronological order
  2. Apply 14-bar embargo: drop bars at the start of each test group
     immediately after a training group → information leakage from
     overlapping h=14 labels
  3. Run V2 sizing pipeline on the test bar series (vol, SMA30, hold)
  4. Compute Sharpe / return / MaxDD on the resulting positions

Output: distribution of 45 SR estimates → quantiles, frac > 0/1/2,
Probability of Backtest Overfitting (PBO).

Usage:
    python scripts/cpcv_v2.py \\
        --pred-dir data/multi_2coins_walkforward \\
        --coins bitcoin ethereum \\
        --start 2021-11-07 --end 2026-04-15 \\
        --n-groups 10 --k-test 2 --embargo 14 \\
        --output-dir data/walkforward_v2_2coin
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_strategy_v2 import run_coin_backtest  # type: ignore  # noqa: E402
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v2_sizing import (  # noqa: E402
    apply_trend_filter, build_positions_with_hold, compute_realized_vol,
    generate_term_structure_signals, vol_regime_mask,
)


COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)


def _load_preds(pred_dir: Path, coin: str) -> pd.DataFrame:
    p7 = pd.read_csv(pred_dir / "preds_lgb_h7.csv", parse_dates=["date"])
    p14 = pd.read_csv(pred_dir / "preds_lgb_h14.csv", parse_dates=["date"])
    for df in (p7, p14):
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    p7 = p7[p7["coin_id"] == coin].rename(columns={"prediction": "pred_h7"})
    p14 = p14[p14["coin_id"] == coin].rename(columns={"prediction": "pred_h14"})[["date", "pred_h14"]]
    return p7.merge(p14, on="date").sort_values("date").reset_index(drop=True)


def _build_v2_path(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute V2 positions over the FULL series; returns (dates, prices, positions).

    Sized over full series so vol/SMA history is correct; we slice into
    test groups afterward.
    """
    sig, conf = generate_term_structure_signals(merged, [7, 14], 0.05, asymmetric=True)
    px = merged["Close"].astype(float).values
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=sig, vol_ok=mask, confidence=conf, realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    pos = apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)
    return merged["date"].values, px, pos


def _slice_with_embargo(
    pos_full: np.ndarray, dates: np.ndarray, px: np.ndarray,
    test_indices: np.ndarray, embargo: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop the first ``embargo`` bars of each contiguous test segment to
    avoid h=14 label leakage from training bars immediately preceding."""
    if len(test_indices) == 0:
        return np.array([]), np.array([]), np.array([])
    sorted_idx = np.sort(test_indices)
    keep = []
    prev = -10**9
    run_start = sorted_idx[0]
    for i in sorted_idx:
        if i != prev + 1:  # new contiguous segment starts
            run_start = i
        if i - run_start >= embargo:  # past embargo for this segment
            keep.append(i)
        prev = i
    keep_arr = np.array(keep, dtype=int)
    return dates[keep_arr], px[keep_arr], pos_full[keep_arr]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--start", default="2021-11-07")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--n-groups", type=int, default=10)
    p.add_argument("--k-test", type=int, default=2)
    p.add_argument("--embargo", type=int, default=14)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 84}")
    print(f"  BT9 CPCV  N={args.n_groups} k={args.k_test} embargo={args.embargo}  "
          f"({args.start} -> {args.end})")
    print(f"{'=' * 84}\n")

    summary: dict = {}
    for coin in args.coins:
        preds = _load_preds(Path(args.pred_dir), coin)
        preds = preds[(preds["date"] >= args.start) & (preds["date"] <= args.end)]
        if preds.empty:
            print(f"[skip] {coin}")
            continue

        ohlcv = _load_crypto_ohlcv(coin, args.end)
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.tz_localize(None).dt.normalize()
        merged = preds.merge(ohlcv[["Date", "Close"]], left_on="date", right_on="Date")
        merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
        merged["ref_price"] = merged["Close"]

        dates_full, px_full, pos_full = _build_v2_path(merged)
        n_bars = len(pos_full)
        # Sequential group boundaries
        group_edges = np.linspace(0, n_bars, args.n_groups + 1, dtype=int)
        group_idx = [
            np.arange(group_edges[i], group_edges[i + 1])
            for i in range(args.n_groups)
        ]

        combos = list(itertools.combinations(range(args.n_groups), args.k_test))
        print(f"  {coin}: {n_bars} bars; {args.n_groups} groups; "
              f"{len(combos)} CPCV splits")

        sr_arr, ret_arr, mdd_arr, win_arr, n_arr = [], [], [], [], []
        for combo in combos:
            test_idx = np.concatenate([group_idx[g] for g in combo])
            test_idx = np.sort(test_idx)
            d, x, p_test = _slice_with_embargo(
                pos_full, dates_full, px_full, test_idx, args.embargo,
            )
            if len(d) < 10:
                continue
            equity, m = run_coin_backtest(
                dates=d, prices=x, positions=p_test,
                initial_capital=10_000.0, **COSTS,
            )
            sr_arr.append(float(m.get("sharpe_ratio", float("nan"))))
            ret_arr.append(float(m.get("total_return", float("nan"))))
            mdd_arr.append(float(m.get("max_drawdown", float("nan"))))
            win_arr.append(float(m.get("win_rate", float("nan"))))
            n_arr.append(int(len(d)))

        sr = np.array(sr_arr)
        sr_clean = sr[~np.isnan(sr)]
        # PBO surrogate: frac of splits where SR <= 0 — proxy for backtest overfit
        # (López de Prado's full PBO needs an "in-sample best" notion; here we
        # report the simpler empirical positive-split fraction)
        agg = {
            "n_splits": int(len(sr_clean)),
            "sr_mean": float(np.mean(sr_clean)) if len(sr_clean) else float("nan"),
            "sr_median": float(np.median(sr_clean)) if len(sr_clean) else float("nan"),
            "sr_std": float(np.std(sr_clean, ddof=1)) if len(sr_clean) > 1 else float("nan"),
            "sr_p05": float(np.quantile(sr_clean, 0.05)) if len(sr_clean) else float("nan"),
            "sr_p25": float(np.quantile(sr_clean, 0.25)) if len(sr_clean) else float("nan"),
            "sr_p75": float(np.quantile(sr_clean, 0.75)) if len(sr_clean) else float("nan"),
            "sr_p95": float(np.quantile(sr_clean, 0.95)) if len(sr_clean) else float("nan"),
            "frac_sr_gt_0": float((sr_clean > 0).mean()) if len(sr_clean) else float("nan"),
            "frac_sr_gt_1": float((sr_clean > 1).mean()) if len(sr_clean) else float("nan"),
            "frac_sr_gt_2": float((sr_clean > 2).mean()) if len(sr_clean) else float("nan"),
            "pbo_proxy_frac_negative": float((sr_clean <= 0).mean()) if len(sr_clean) else float("nan"),
            "mean_test_bars": float(np.mean(n_arr)) if n_arr else float("nan"),
        }
        summary[coin] = agg
        print(f"  {coin}  SR splits ({agg['n_splits']} valid):")
        print(f"    mean={agg['sr_mean']:+.2f}  median={agg['sr_median']:+.2f}  "
              f"std={agg['sr_std']:.2f}")
        print(f"    p05={agg['sr_p05']:+.2f}  IQR=[{agg['sr_p25']:+.2f},{agg['sr_p75']:+.2f}]  "
              f"p95={agg['sr_p95']:+.2f}")
        print(f"    Frac SR>0: {agg['frac_sr_gt_0']:.0%}  >1: {agg['frac_sr_gt_1']:.0%}  "
              f">2: {agg['frac_sr_gt_2']:.0%}  PBO_proxy(SR≤0): "
              f"{agg['pbo_proxy_frac_negative']:.0%}")
        print(f"    Mean test bars per split: {agg['mean_test_bars']:.0f}\n")

    with open(out_dir / "cpcv.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote: {out_dir / 'cpcv.json'}")


if __name__ == "__main__":
    main()
