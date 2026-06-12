#!/usr/bin/env python
"""Evaluate V2 quant baseline on an exact date window.

The stock baseline_strategy_v2.py reports metrics over the full prediction
range, which makes apples-to-apples comparison with the LLM agent runs
(limited to a 90-day window) impossible.

This wrapper:
  1. Reads full predictions from --pred-dir (all history)
  2. Runs the V2 strategy on the full history so SMA30 / vol lookback /
     position-building use all available warmup data
  3. Slices the per-day equity curve to [--window-start, --window-end]
  4. Recomputes return/Sharpe/MaxDD/etc. on the sliced equity

Usage:
    python scripts/baseline_on_window.py \\
        --pred-dir data/multi_3coins_bnb \\
        --window-start 2026-01-16 --window-end 2026-04-15 \\
        --symmetric
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling scripts importable when run directly
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from baseline_strategy_v2 import (
    apply_trend_filter,
    build_positions_with_hold,
    compute_realized_vol,
    generate_term_structure_signals,
    load_horizon_predictions,
    run_coin_backtest,
    vol_regime_mask,
)


def slice_equity(dates: np.ndarray, equity: list[float], positions: np.ndarray,
                 start: str, end: str) -> dict:
    """Slice a per-day equity/positions trace to a date window and recompute metrics."""
    ts = pd.to_datetime(dates, utc=True).tz_localize(None)
    win_start = pd.to_datetime(start)
    win_end = pd.to_datetime(end)

    # Find indices covered by the window
    mask = (ts >= win_start) & (ts <= win_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return {"error": f"no bars in window {start} → {end}"}

    i0, i1 = idx[0], idx[-1]
    if i0 == 0:
        # Need at least one earlier bar for PnL on the first window day;
        # shift in by one
        i0 = 1

    eq = np.asarray(equity, dtype=float)
    window_eq = eq[i0 : i1 + 1]
    window_pos = positions[i0 : i1 + 1]

    # Daily returns from the sliced equity
    daily_returns = np.diff(window_eq) / window_eq[:-1]

    total_return = (window_eq[-1] - window_eq[0]) / window_eq[0]
    n_days = len(daily_returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    # Count only days where the strategy held a position
    traded_mask = np.abs(window_pos[1:]) > 1e-9
    traded_returns = daily_returns[traded_mask]

    if len(traded_returns) > 1:
        excess = traded_returns - daily_rf
        std_ex = np.std(excess, ddof=1)
        sharpe = float(np.mean(excess) / std_ex * np.sqrt(252)) if std_ex > 0 else 0
    else:
        sharpe = 0

    running_max = np.maximum.accumulate(window_eq)
    dd = np.where(running_max > 0, (running_max - window_eq) / running_max, 0)
    max_dd = float(np.max(dd))

    n_trades = int(np.sum(np.abs(np.diff(window_pos)) > 1e-9))
    wins = int(np.sum(traded_returns > 0))
    win_rate = wins / len(traded_returns) if len(traded_returns) > 0 else 0

    return {
        "window_start": start,
        "window_end": end,
        "n_days": int(n_days),
        "total_return": float(total_return),
        "annualized_return": float(ann_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": float(win_rate),
        "n_trades": n_trades,
        "start_equity": float(window_eq[0]),
        "end_equity": float(window_eq[-1]),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="V2 baseline evaluated on an exact date window.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True)
    p.add_argument("--window-start", required=True)
    p.add_argument("--window-end", required=True)
    p.add_argument("--coins", nargs="+", default=None)
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14])
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--min-hold", type=int, default=7)
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--vol-lookback", type=int, default=20)
    p.add_argument("--kelly-fraction", type=float, default=0.5)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--stop-loss", type=float, default=0.03)
    p.add_argument("--max-portfolio-dd", type=float, default=0.15)
    p.add_argument("--vol-cap-pct", type=float, default=0.95)
    p.add_argument("--confidence-ref-return", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--spread", type=float, default=0.0005)
    p.add_argument("--price-impact", type=float, default=0.001)
    p.add_argument("--funding-rate", type=float, default=0.0001)
    p.add_argument("--symmetric", action="store_true")
    p.add_argument("--early-exit-loss", type=float, default=0.015)
    p.add_argument("--trend-sma", type=int, default=30)
    p.add_argument("--trend-multiplier", type=float, default=1.5)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)

    merged = load_horizon_predictions(pred_dir, args.horizons)
    coins = args.coins or sorted(merged["coin_id"].unique())

    print(f"\n{'=' * 70}")
    print(f"  V2 Baseline evaluated on exact window")
    print(f"  Window   : {args.window_start} → {args.window_end}")
    print(f"  Coins    : {', '.join(coins)}")
    print(f"{'=' * 70}")

    per_coin_equity = {}
    per_coin_dates = {}
    per_coin_positions = {}
    per_coin_window_metrics = {}

    for coin in coins:
        df_coin = merged[merged["coin_id"] == coin].sort_values("date").reset_index(drop=True)
        if len(df_coin) < 30:
            print(f"\n  {coin}: skipped (only {len(df_coin)} rows)")
            continue

        signals, confidence = generate_term_structure_signals(
            df_coin, args.horizons, args.confidence_ref_return,
            asymmetric=not args.symmetric,
        )
        prices = df_coin["ref_price"].values
        dates = df_coin["date"].values
        realized_vol = compute_realized_vol(prices, args.vol_lookback)
        vol_ok = vol_regime_mask(realized_vol, args.vol_cap_pct)

        positions = build_positions_with_hold(
            signals, vol_ok, confidence, realized_vol, prices,
            args.target_vol, args.kelly_fraction, args.max_leverage,
            args.min_hold, args.early_exit_loss,
        )
        if args.trend_sma > 0:
            positions = apply_trend_filter(
                positions, prices, args.trend_sma, args.trend_multiplier,
            )

        equity, _metrics = run_coin_backtest(
            dates, prices, positions,
            initial_capital=args.initial_capital,
            fee_rate=args.fee_rate,
            slippage=args.slippage,
            spread=args.spread,
            price_impact=args.price_impact,
            funding_rate=args.funding_rate,
            stop_loss=args.stop_loss,
            max_portfolio_dd=args.max_portfolio_dd,
        )
        win_metrics = slice_equity(
            dates, equity, positions,
            args.window_start, args.window_end,
        )
        per_coin_equity[coin] = equity
        per_coin_dates[coin] = dates
        per_coin_positions[coin] = positions
        per_coin_window_metrics[coin] = win_metrics

        print(f"\n{coin:<12}  return={win_metrics['total_return']*100:+.2f}%  "
              f"sharpe={win_metrics['sharpe_ratio']:+.2f}  "
              f"maxDD={win_metrics['max_drawdown']*100:.2f}%  "
              f"winRate={win_metrics['win_rate']*100:.1f}%  "
              f"trades={win_metrics['n_trades']}  "
              f"({win_metrics['n_days']} days)")

    # Equal-weight portfolio on the sliced window
    # Align all coins on a shared date index inside the window
    all_window_returns = []
    for coin in coins:
        dates = pd.to_datetime(per_coin_dates[coin], utc=True).tz_localize(None)
        mask = (dates >= pd.to_datetime(args.window_start)) & \
               (dates <= pd.to_datetime(args.window_end))
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        i0, i1 = max(idx[0], 1), idx[-1]
        eq = np.asarray(per_coin_equity[coin])[i0 : i1 + 1]
        rets = np.diff(eq) / eq[:-1]
        all_window_returns.append(rets)

    if all_window_returns:
        min_len = min(len(r) for r in all_window_returns)
        mat = np.stack([r[:min_len] for r in all_window_returns])
        port_rets = mat.mean(axis=0)
        port_equity = np.cumprod(1 + port_rets) * args.initial_capital

        port_total = (port_equity[-1] - args.initial_capital) / args.initial_capital
        port_ann = (1 + port_total) ** (252 / len(port_rets)) - 1
        daily_rf = (1 + 0.045) ** (1 / 252) - 1
        excess = port_rets - daily_rf
        std_ex = np.std(excess, ddof=1)
        port_sharpe = float(np.mean(excess) / std_ex * np.sqrt(252)) if std_ex > 0 else 0
        rm = np.maximum.accumulate(port_equity)
        port_dd = float(np.max(np.where(rm > 0, (rm - port_equity) / rm, 0)))

        print(f"\n{'=' * 70}")
        print(f"  Equal-Weight Portfolio ({len(coins)} coins)")
        print(f"  Window: {args.window_start} → {args.window_end} ({len(port_rets)} bars)")
        print(f"{'=' * 70}")
        print(f"  Return   : {port_total*100:+.2f}%")
        print(f"  Ann.Ret  : {port_ann*100:+.2f}%")
        print(f"  Sharpe   : {port_sharpe:+.2f}")
        print(f"  Max DD   : {port_dd*100:.2f}%")

        per_coin_window_metrics["_portfolio"] = {
            "total_return": float(port_total),
            "annualized_return": float(port_ann),
            "sharpe_ratio": port_sharpe,
            "max_drawdown": port_dd,
            "n_bars": int(len(port_rets)),
        }

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(per_coin_window_metrics, f, indent=2)
        print(f"\n  Metrics JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
