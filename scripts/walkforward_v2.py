#!/usr/bin/env python
"""BT8 — Expanding-window walk-forward backtest of V2 quant baseline.

Per BACKTESTING_METHODOLOGY.md §5: slices walk-forward LGB predictions
into non-overlapping quarterly test blocks (63 bars each, 14-bar
embargo enforced upstream by evaluate_models_multi's purging) and runs
the production V2 sizing pipeline on each. Aggregates quarterly Sharpe,
returns, max-drawdown.

The LGB model is already retrained per training window inside
evaluate_models_multi (walk-forward). This script consumes the
resulting prediction CSVs and re-runs the strategy layer on each
quarter slice — exactly the protocol §5.3 prescribes for fixed
hyperparameters.

Usage:
    python scripts/walkforward_v2.py \\
        --pred-dir data/multi_2coins_walkforward \\
        --coins bitcoin ethereum \\
        --start 2022-01-01 --end 2026-04-15 \\
        --quarter-bars 63 --output-dir data/walkforward_v2_2coin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def _load_prices(coin: str, end: str) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coin, end)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def _v2_positions(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
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
    return pos, px


def _quarter_blocks(dates: pd.DatetimeIndex, q_bars: int) -> list[tuple[int, int]]:
    """Return list of (start_idx, end_idx_exclusive) blocks of length q_bars."""
    n = len(dates)
    out = []
    for s in range(0, n, q_bars):
        e = min(s + q_bars, n)
        if e - s >= max(20, q_bars // 2):  # need at least half a quarter
            out.append((s, e))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--quarter-bars", type=int, default=63)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    coin_curves: dict[str, list[float]] = {}
    coin_dates: dict[str, list] = {}
    daily_returns_rows: list[dict] = []

    for coin in args.coins:
        preds = _load_preds(Path(args.pred_dir), coin)
        preds = preds[(preds["date"] >= args.start) & (preds["date"] <= args.end)]
        if preds.empty:
            print(f"[skip] {coin}: no preds in window")
            continue

        prices = _load_prices(coin, args.end)
        merged = preds.merge(prices[["Date", "Close"]], left_on="date", right_on="Date")
        merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
        merged = merged.rename(columns={"date": "_date"})

        # Add ref_price for generate_term_structure_signals
        merged["ref_price"] = merged["Close"]

        # Compute V2 positions over the FULL series first (vol/SMA need history)
        pos_full, px_full = _v2_positions(merged)
        dates_full = merged["_date"].values

        # Slice into quarterly test blocks
        blocks = _quarter_blocks(merged["_date"], args.quarter_bars)
        equity_full = []
        coin_dates[coin] = list(dates_full)

        for s, e in blocks:
            block_dates = dates_full[s:e]
            block_px = px_full[s:e]
            block_pos = pos_full[s:e]
            equity, m = run_coin_backtest(
                dates=block_dates, prices=block_px, positions=block_pos,
                initial_capital=10_000.0, **COSTS,
            )
            eq_arr = np.asarray(equity, dtype=float)
            if len(eq_arr) > 1:
                rets = eq_arr[1:] / eq_arr[:-1] - 1.0
                for i, r in enumerate(rets):
                    daily_returns_rows.append({
                        "date": pd.Timestamp(block_dates[i + 1]).strftime("%Y-%m-%d"),
                        "coin": coin,
                        "ret": float(r),
                    })
            sr = float(m.get("sharpe_ratio", float("nan")))
            ret = float(m.get("total_return", float("nan")))
            mdd = float(m.get("max_drawdown", float("nan")))
            wr = float(m.get("win_rate", float("nan")))
            ntr = int(m.get("n_trades", 0))
            quarter_label = pd.Timestamp(block_dates[0]).strftime("%Y-Q%q").replace(
                "Q1", "Q" + str((pd.Timestamp(block_dates[0]).month - 1) // 3 + 1)
            )
            # Cleaner quarter label
            ts0 = pd.Timestamp(block_dates[0])
            ts1 = pd.Timestamp(block_dates[-1])
            q_lbl = f"{ts0.strftime('%Y-%m-%d')}_{ts1.strftime('%Y-%m-%d')}"
            all_rows.append({
                "coin": coin,
                "quarter": q_lbl,
                "n_bars": e - s,
                "sharpe": sr,
                "total_return": ret,
                "max_dd": mdd,
                "win_rate": wr,
                "n_trades": ntr,
            })
            equity_full.extend(equity[1:] if equity else [])
        coin_curves[coin] = equity_full

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "quarterly_metrics.csv", index=False)
    pd.DataFrame(daily_returns_rows).to_csv(out_dir / "daily_returns.csv", index=False)

    # Aggregate per coin
    print(f"\n{'=' * 86}")
    print(f"  Walk-forward V2 baseline ({args.start} -> {args.end})  q={args.quarter_bars} bars")
    print(f"{'=' * 86}\n")
    summary = {}

    # Aggregated daily returns → block bootstrap CI for OOS Sharpe
    daily_df = pd.DataFrame(daily_returns_rows)
    from scripts.bootstrap_sharpe import sharpe, stationary_bootstrap_sample  # type: ignore
    rng = np.random.default_rng(42)
    DAILY_RF = (1 + 0.045) ** (1 / 252) - 1
    bs_n_iter = 3000
    bs_block = 5
    for coin in args.coins:
        sub = df[df["coin"] == coin]
        if sub.empty:
            continue
        sr_arr = sub["sharpe"].dropna().values
        ret_arr = sub["total_return"].dropna().values
        # Aggregated OOS Sharpe + bootstrap CI on concatenated daily returns
        coin_daily = daily_df[daily_df["coin"] == coin]["ret"].values
        sr_oos = float(sharpe(coin_daily)) if len(coin_daily) > 1 else float("nan")
        bs_samples = np.empty(bs_n_iter)
        for k in range(bs_n_iter):
            bs_samples[k] = sharpe(stationary_bootstrap_sample(coin_daily, bs_block, rng))
        sr_ci_lo = float(np.quantile(bs_samples, 0.025))
        sr_ci_hi = float(np.quantile(bs_samples, 0.975))
        p_sr_gt_0 = float((bs_samples > 0).mean())
        p_sr_gt_1 = float((bs_samples > 1).mean())

        agg = {
            "n_quarters": int(len(sub)),
            "n_daily_bars": int(len(coin_daily)),
            "sr_oos_aggregated": sr_oos,
            "sr_oos_ci95": [sr_ci_lo, sr_ci_hi],
            "p_sr_gt_0_boot": p_sr_gt_0,
            "p_sr_gt_1_boot": p_sr_gt_1,
            "sr_quarter_mean": float(np.mean(sr_arr)) if len(sr_arr) else float("nan"),
            "sr_quarter_median": float(np.median(sr_arr)) if len(sr_arr) else float("nan"),
            "sr_quarter_std": float(np.std(sr_arr, ddof=1)) if len(sr_arr) > 1 else float("nan"),
            "sr_quarter_p25": float(np.quantile(sr_arr, 0.25)) if len(sr_arr) else float("nan"),
            "sr_quarter_p75": float(np.quantile(sr_arr, 0.75)) if len(sr_arr) else float("nan"),
            "frac_sr_gt_0": float((sr_arr > 0).mean()) if len(sr_arr) else float("nan"),
            "frac_sr_gt_1": float((sr_arr > 1).mean()) if len(sr_arr) else float("nan"),
            "frac_sr_gt_2": float((sr_arr > 2).mean()) if len(sr_arr) else float("nan"),
            "geo_total_return": float(np.prod(1 + ret_arr) - 1) if len(ret_arr) else float("nan"),
            "max_quarter_dd": float(sub["max_dd"].max()),
        }
        summary[coin] = agg
        print(f"  {coin}  ({agg['n_quarters']} quarters, {agg['n_daily_bars']} daily bars)")
        print(f"    OOS Sharpe (aggregated): {agg['sr_oos_aggregated']:+.2f}  "
              f"CI95=[{agg['sr_oos_ci95'][0]:+.2f},{agg['sr_oos_ci95'][1]:+.2f}]  "
              f"P(SR>0)={agg['p_sr_gt_0_boot']:.3f}  P(SR>1)={agg['p_sr_gt_1_boot']:.3f}")
        print(f"    Quarter SR  mean={agg['sr_quarter_mean']:+.2f}  median={agg['sr_quarter_median']:+.2f}  "
              f"std={agg['sr_quarter_std']:.2f}  IQR=[{agg['sr_quarter_p25']:+.2f},{agg['sr_quarter_p75']:+.2f}]")
        print(f"    Frac SR>0: {agg['frac_sr_gt_0']:.0%}  >1: {agg['frac_sr_gt_1']:.0%}  >2: {agg['frac_sr_gt_2']:.0%}")
        print(f"    Compounded return: {agg['geo_total_return']:+.1%}  Max quarter DD: {agg['max_quarter_dd']:.1%}")
        print()
        # Per-quarter row dump
        print(f"    {'quarter':<22} {'SR':>7}  {'ret':>8}  {'MaxDD':>7}  {'win':>5}  {'#tr':>4}")
        for _, r in sub.iterrows():
            print(f"    {r['quarter']:<22} {r['sharpe']:>+7.2f}  {r['total_return']:>+7.2%}  "
                  f"{r['max_dd']:>6.2%}  {r['win_rate']:>4.0%}   {r['n_trades']:>4d}")
        print()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Plot per-coin equity curves
    fig, ax = plt.subplots(figsize=(14, 6))
    for coin, eq in coin_curves.items():
        if not eq:
            continue
        # Compounded across quarters: each quarter starts at 10k; accumulate gross return
        sub = df[df["coin"] == coin]
        compounded = np.cumprod(1 + sub["total_return"].fillna(0).values) * 10000.0
        q_dates = [pd.Timestamp(q.split("_")[1]) for q in sub["quarter"]]
        ax.plot(q_dates, compounded, marker="o", label=coin, linewidth=1.6)
    ax.set_xlabel("Quarter end")
    ax.set_ylabel("Compounded equity (10k initial per coin)")
    ax.set_title(f"Walk-forward V2 baseline — quarterly compounded equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.savefig(out_dir / "walkforward_equity.png", dpi=130, bbox_inches="tight")

    print(f"  Wrote: {out_dir / 'quarterly_metrics.csv'}")
    print(f"  Wrote: {out_dir / 'summary.json'}")
    print(f"  Plot:  {out_dir / 'walkforward_equity.png'}")


if __name__ == "__main__":
    main()
