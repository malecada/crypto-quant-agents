#!/usr/bin/env python
"""Multi-horizon term structure consensus strategy.

Reads predictions produced by evaluate_models_multi.py and trades only when
signals from horizons {1, 7, 14} agree on direction.
Reuses all risk-management primitives from scripts/baseline_strategy.py.

Usage:
    # Basic: BTC using LightGBM at h∈{1,7,14}
    python scripts/baseline_strategy_multi.py --coin bitcoin --model lgb

    # Looser consensus: only 2 of 3 horizons must agree
    python scripts/baseline_strategy_multi.py --coin bitcoin --model lgb \
        --horizons 1 7 14 --min-agreeing 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Reuse risk-management primitives from the single-horizon baseline
from scripts.baseline_strategy import (
    apply_leverage,
    build_positions,
    compute_realized_vol,
    run_baseline_backtest,
    vol_regime_mask,
    vol_targeted_size,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-horizon term structure consensus strategy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--preds-dir", default="data/multi",
                   help="Directory with preds_{model}_h{h}.csv files.")
    p.add_argument("--coin", required=True, help="CoinGecko ID.")
    p.add_argument("--model", default="lgb", choices=["rf", "lgb", "arima"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 7, 14],
                   help="Horizons to use for consensus.")
    p.add_argument("--min-agreeing", type=int, default=3,
                   help="Min horizons that must agree on direction (default: all).")

    # Risk / sizing — mirror baseline_strategy.py defaults
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--target-vol", type=float, default=0.10)
    p.add_argument("--vol-lookback", type=int, default=20)
    p.add_argument("--kelly-fraction", type=float, default=0.5)
    p.add_argument("--max-leverage", type=float, default=3.0)
    p.add_argument("--stop-loss", type=float, default=0.03)
    p.add_argument("--max-portfolio-dd", type=float, default=0.15)
    p.add_argument("--vol-cap-pct", type=float, default=0.95)
    p.add_argument("--confidence-ref-return", type=float, default=0.02)

    # Costs
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--spread", type=float, default=0.0005)
    p.add_argument("--price-impact", type=float, default=0.001)
    p.add_argument("--funding-rate", type=float, default=0.0001)

    p.add_argument("--output-plot", default=None,
                   help="Equity curve plot path. Default: preds_dir/term_struct_{coin}_{model}.png")
    return p.parse_args()


def load_horizon_predictions(
    preds_dir: Path, model: str, horizons: list, coin: str,
) -> pd.DataFrame:
    """Load and merge predictions for one coin across multiple horizons.

    Returns a date-indexed DataFrame with columns:
      - pred_h{h} for each horizon
      - actual_h{h} for each horizon
      - current_price (today's spot, derived from actual_h1 shifted forward by 1)
    """
    frames = []
    for h in horizons:
        path = preds_dir / f"preds_{model}_h{h}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing predictions: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df[df["coin_id"] == coin].copy()
        if df.empty:
            raise ValueError(f"No predictions for {coin} in {path}")
        df = df.sort_values("date").set_index("date")[["prediction", "actual"]]
        df.columns = [f"pred_h{h}", f"actual_h{h}"]
        frames.append(df)

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.join(f, how="inner")

    # Derive current spot price from actual_h1 shifted forward by 1 bar.
    # actual_h1 at date t is close[t+1]; shift(1) gives close[t] = today's spot.
    if "actual_h1" in merged.columns:
        merged["current_price"] = merged["actual_h1"].shift(1)
    else:
        # Fall back: use the lowest-horizon actual as an approximation
        lowest_h = min(horizons)
        merged["current_price"] = merged[f"actual_h{lowest_h}"].shift(1)

    merged = merged.dropna()
    return merged


def generate_term_structure_signals(
    df: pd.DataFrame,
    horizons: list,
    min_agreeing: int,
) -> tuple:
    """Generate term-structure consensus signals and confidence proxies.

    For each bar:
      - At each horizon h, direction = sign(pred_h{h} - current_price)
      - If >= min_agreeing horizons agree long -> signal = +1
      - If >= min_agreeing horizons agree short -> signal = -1
      - Else -> signal = 0 (no trade)

    Confidence = mean of |pred_h{h} - current| / current across horizons.

    Returns:
        (signals: np.ndarray, pred_returns: np.ndarray) aligned with df.index
    """
    signals = np.zeros(len(df))
    pred_returns = np.zeros(len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        current = row["current_price"]
        if current <= 0 or np.isnan(current):
            continue

        dirs = []
        rets = []
        for h in horizons:
            p = row[f"pred_h{h}"]
            if np.isnan(p):
                continue
            if p > current:
                dirs.append(1)
            elif p < current:
                dirs.append(-1)
            else:
                dirs.append(0)
            rets.append(abs(p - current) / current)

        if not dirs:
            continue

        n_long = sum(1 for d in dirs if d == 1)
        n_short = sum(1 for d in dirs if d == -1)

        if n_long >= min_agreeing:
            signals[i] = 1
        elif n_short >= min_agreeing:
            signals[i] = -1
        # else flat

        pred_returns[i] = float(np.mean(rets)) if rets else 0.0

    return signals, pred_returns


def main():
    args = parse_args()
    preds_dir = Path(args.preds_dir)

    df = load_horizon_predictions(preds_dir, args.model, args.horizons, args.coin)
    print(f"\nLoaded {len(df)} bars for {args.coin} using {args.model} predictions")
    print(f"Date range: {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}")

    signals, pred_returns = generate_term_structure_signals(
        df, args.horizons, args.min_agreeing,
    )
    print(f"\nSignals: long={int((signals==1).sum())}  "
          f"short={int((signals==-1).sum())}  "
          f"flat={int((signals==0).sum())}")

    prices = df["current_price"].values
    dates = df.index.values

    realized_vol = compute_realized_vol(prices, args.vol_lookback)
    vol_ok = vol_regime_mask(realized_vol, args.vol_cap_pct)

    # Confidence from mean predicted return magnitude (clipped to [0,1])
    confidence = np.minimum(1.0, pred_returns / args.confidence_ref_return)

    positions = build_positions(
        signals, vol_ok, confidence, realized_vol,
        args.target_vol, args.kelly_fraction, args.max_leverage,
    )

    equity, metrics = run_baseline_backtest(
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

    print(f"\n{'='*60}")
    print(f"  Term Structure Strategy — {args.coin} ({args.model})")
    print(f"{'='*60}")
    print(f"  Horizons used      : {args.horizons}")
    print(f"  Min agreeing       : {args.min_agreeing}")
    print(f"  Target vol         : {args.target_vol:.1%}")
    print(f"  Max leverage       : {args.max_leverage}x")
    print(f"  Stop-loss          : {args.stop_loss:.1%}")
    print(f"")
    print(f"  Total Return       : {metrics['total_return']:+.2%}")
    print(f"  Annualized Return  : {metrics['annualized_return']:+.2%}")
    print(f"  Sharpe             : {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate           : {metrics['win_rate']:.1%}")
    print(f"  # Trades           : {metrics['n_trades']}")
    pf = metrics["profit_factor"]
    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
    print(f"  Profit Factor      : {pf_str}")
    print(f"  Circuit breaker    : {'TRIGGERED' if metrics['halted'] else 'OK'}")

    bh_return = (prices[-1] - prices[0]) / prices[0]
    print(f"\n  Buy & Hold         : {bh_return:+.2%}")

    # Plot
    if args.output_plot:
        plot_path = args.output_plot
    else:
        plot_path = str(preds_dir / f"term_struct_{args.coin}_{args.model}.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    dates_pd = pd.to_datetime(dates)
    ax.plot(dates_pd, equity, linewidth=1.5, color="tab:blue",
            label=f"Term Structure ({args.model})")
    bh_equity = [args.initial_capital * (1 + (prices[i] - prices[0]) / prices[0])
                 for i in range(len(prices))]
    ax.plot(dates_pd, bh_equity, linewidth=1.5, linestyle=":", color="black",
            label="Buy & Hold")
    ax.axhline(y=args.initial_capital, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5)

    ax.set_title(f"Term Structure Strategy: {args.coin}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nEquity curve -> {plot_path}")


if __name__ == "__main__":
    main()
