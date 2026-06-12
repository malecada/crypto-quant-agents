#!/usr/bin/env python
"""BT11 — Random-entry placebo for V2 quant baseline.

Per BACKTESTING_METHODOLOGY.md §8.3: replace LGB direction calls with
random ±1 (matching the empirical long/short frequency), keep the V2
sizing pipeline intact, run K simulations. Compares observed SR
against the random-entry null distribution.

Decomposes total Sharpe into signal-quality vs sizing-mechanics
contributions: if random-entry SR is surprisingly high (e.g. >1.0),
the V2 sizing layer is doing significant work independent of LGB
signal accuracy.

Usage:
    python scripts/random_entry_placebo.py \\
        --pred-dir data/multi_2coins_walkforward \\
        --coins bitcoin ethereum \\
        --start 2021-11-07 --end 2026-04-15 \\
        --n-perms 1000 --output-dir data/walkforward_v2_2coin
"""

from __future__ import annotations

import argparse
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


def _v2_pipeline(merged: pd.DataFrame, signals: np.ndarray, conf: np.ndarray) -> np.ndarray:
    px = merged["Close"].astype(float).values
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=signals, vol_ok=mask, confidence=conf, realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    return apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--start", default="2021-11-07")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'=' * 84}")
    print(f"  BT11 random-entry placebo  ({args.start} -> {args.end})  K={args.n_perms}")
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

        # Observed
        obs_sig, obs_conf = generate_term_structure_signals(
            merged, [7, 14], 0.05, asymmetric=True,
        )
        obs_pos = _v2_pipeline(merged, obs_sig, obs_conf)
        equity, m = run_coin_backtest(
            dates=merged["date"].values, prices=merged["Close"].values, positions=obs_pos,
            initial_capital=10_000.0, **COSTS,
        )
        sr_obs = float(m.get("sharpe_ratio", float("nan")))
        ret_obs = float(m.get("total_return", float("nan")))

        # Empirical long/short/flat frequencies from observed signals
        n = len(obs_sig)
        n_long = int((obs_sig > 0).sum())
        n_short = int((obs_sig < 0).sum())
        n_flat = int((obs_sig == 0).sum())
        p_long = n_long / n
        p_short = n_short / n
        p_flat = n_flat / n

        # Random direction draws matching the empirical mix
        choices = np.array([1.0, -1.0, 0.0])
        weights = np.array([p_long, p_short, p_flat])

        sr_perm = np.empty(args.n_perms)
        ret_perm = np.empty(args.n_perms)
        for k in range(args.n_perms):
            rand_sig = rng.choice(choices, size=n, p=weights)
            # Use the OBSERVED confidence series so the sizing signal still
            # has realistic magnitude — only the direction is randomised
            rand_pos = _v2_pipeline(merged, rand_sig, np.abs(obs_conf))
            _, mr = run_coin_backtest(
                dates=merged["date"].values, prices=merged["Close"].values,
                positions=rand_pos, initial_capital=10_000.0, **COSTS,
            )
            sr_perm[k] = float(mr.get("sharpe_ratio", 0.0))
            ret_perm[k] = float(mr.get("total_return", 0.0))

        p_value = float((sr_perm >= sr_obs).mean())
        summary[coin] = {
            "T": int(n),
            "p_long": p_long, "p_short": p_short, "p_flat": p_flat,
            "sr_observed": sr_obs,
            "ret_observed": ret_obs,
            "sr_perm_mean": float(sr_perm.mean()),
            "sr_perm_std": float(sr_perm.std(ddof=1)),
            "sr_perm_p05": float(np.quantile(sr_perm, 0.05)),
            "sr_perm_median": float(np.quantile(sr_perm, 0.50)),
            "sr_perm_p95": float(np.quantile(sr_perm, 0.95)),
            "sr_perm_p99": float(np.quantile(sr_perm, 0.99)),
            "p_value": p_value,
            "n_perms": args.n_perms,
            "alpha_attribution": {
                "signal_quality_sr": sr_obs - float(sr_perm.mean()),
                "sizing_floor_sr": float(sr_perm.mean()),
                "pct_from_signal": float(
                    (sr_obs - sr_perm.mean()) / sr_obs * 100
                ) if sr_obs > 0 else float("nan"),
            },
        }
        print(f"  {coin}  T={n}  signal mix(long/short/flat)={p_long:.0%}/{p_short:.0%}/{p_flat:.0%}")
        print(f"    Observed SR={sr_obs:+.2f}  ret={ret_obs:+.1%}")
        print(f"    Random-entry null:  mean={sr_perm.mean():+.2f}  std={sr_perm.std(ddof=1):.2f}")
        print(f"      p05={float(np.quantile(sr_perm, 0.05)):+.2f}  "
              f"med={float(np.quantile(sr_perm, 0.50)):+.2f}  "
              f"p95={float(np.quantile(sr_perm, 0.95)):+.2f}  "
              f"p99={float(np.quantile(sr_perm, 0.99)):+.2f}")
        print(f"    p-value (SR_perm >= SR_obs): {p_value:.4f}")
        sig_attr = summary[coin]["alpha_attribution"]["pct_from_signal"]
        print(f"    Signal contribution: {sig_attr:.0f}% of observed SR  "
              f"(sizing floor SR ~ {sr_perm.mean():+.2f})\n")

    with open(out_dir / "random_entry_placebo.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote: {out_dir / 'random_entry_placebo.json'}")


if __name__ == "__main__":
    main()
