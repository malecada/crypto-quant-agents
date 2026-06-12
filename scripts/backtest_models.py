#!/usr/bin/env python
"""Backtest prediction models using Krypto-v0-style strategies.

Reads eval_predictions.csv (from evaluate_models.py) and runs:
- DirectionalSignal: always trade based on prediction direction
- ThresholdSignal: only trade when prediction exceeds threshold
- EnsembleConsensus: only trade when both RF and ARIMA agree

Usage:
    python scripts/backtest_models.py --input data/eval_predictions.csv
    python scripts/backtest_models.py --input data/eth/eval_predictions.csv --threshold 0.02
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Strategies (ported from Krypto-v0) ──────────────────────────────


def directional_signal(prediction, actual_prev):
    """Long if pred > prev, short otherwise. Always trades."""
    return +1.0 if prediction > actual_prev else -1.0


def threshold_signal(prediction, actual_prev, threshold=0.01):
    """Only trade if |pred - prev| / prev > threshold."""
    pct_diff = abs(prediction - actual_prev) / actual_prev if actual_prev != 0 else 0
    if pct_diff <= threshold:
        return 0.0
    return +1.0 if prediction > actual_prev else -1.0


def ensemble_consensus(pred_a, actual_prev_a, pred_b, actual_prev_b):
    """Only trade when both models agree on direction."""
    dir_a = +1 if pred_a > actual_prev_a else -1
    dir_b = +1 if pred_b > actual_prev_b else -1
    if dir_a == dir_b:
        return float(dir_a)
    return 0.0


# ── Backtest engine (same as existing, but works with positions directly) ─


def run_model_backtest(
    dates, actuals, positions,
    initial_capital=10_000.0,
    fee_rate=0.001,
    slippage=0.001,
    short_cost=0.0005,
    price_impact=0.001,
):
    """Run backtest from position array. Returns metrics dict + equity curve."""
    equity = [initial_capital]
    daily_returns = []

    for i in range(1, len(dates)):
        actual_prev = actuals[i - 1]
        actual_i = actuals[i]

        if np.isnan(actual_prev) or np.isnan(actual_i) or actual_prev == 0:
            daily_returns.append(0.0)
            equity.append(equity[-1])
            continue

        pos = positions[i]
        if abs(pos) < 1e-9:
            daily_returns.append(0.0)
            equity.append(equity[-1])
            continue

        price_return = (actual_i - actual_prev) / actual_prev
        gross_ret = pos * price_return

        abs_pos = abs(pos)
        cost = (2 * fee_rate + slippage) * abs_pos
        # Price impact: quadratic in position size (larger trades move price more)
        cost += price_impact * abs_pos * abs_pos
        if pos < 0:
            cost += short_cost * abs_pos

        net_ret = gross_ret - cost
        daily_returns.append(net_ret)
        equity.append(equity[-1] * (1 + net_ret))

    returns = np.array(daily_returns)
    pos_arr = np.array(positions[1:])  # align with returns

    traded_mask = np.abs(pos_arr) > 1e-9
    traded_returns = returns[traded_mask]

    total_return = (equity[-1] - initial_capital) / initial_capital
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    # Sharpe
    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    if len(traded_returns) > 1:
        excess = traded_returns - daily_rf
        std_ex = np.std(excess, ddof=1)
        sharpe = float(np.mean(excess) / std_ex * np.sqrt(252)) if std_ex > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = np.where(running_max > 0, (running_max - eq) / running_max, 0)
    max_dd = float(np.max(dd))

    # Win rate
    n_trades = int(traded_mask.sum())
    wins = int(np.sum(traded_returns > 0))
    win_rate = wins / n_trades if n_trades > 0 else 0

    # Profit factor
    gross_profit = float(np.sum(traded_returns[traded_returns > 0]))
    gross_loss = float(np.abs(np.sum(traded_returns[traded_returns < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    # Directional accuracy
    if len(traded_returns) > 0:
        correct_dir = 0
        total_dir = 0
        for i in range(1, len(dates)):
            if abs(positions[i]) > 1e-9:
                actual_move = actuals[i] - actuals[i - 1]
                if (positions[i] > 0 and actual_move > 0) or (positions[i] < 0 and actual_move < 0):
                    correct_dir += 1
                total_dir += 1
        dir_accuracy = correct_dir / total_dir if total_dir > 0 else 0
    else:
        dir_accuracy = 0

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "profit_factor": pf,
        "directional_accuracy": dir_accuracy,
    }, equity


# ── Main ────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Backtest model predictions with Krypto-v0 strategies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to eval_predictions.csv.")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--threshold", type=float, default=0.01,
                    help="Threshold for ThresholdSignal (e.g. 0.01 = 1%%).")
    p.add_argument("--position-size", type=float, default=0.01,
                    help="Fraction of capital per trade (0.01 = 1%%).")
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--short-cost", type=float, default=0.0005)
    p.add_argument("--price-impact", type=float, default=0.001,
                    help="Price impact coefficient (quadratic in position size).")
    p.add_argument("--output-plot", default=None,
                    help="Equity curve plot path. Default: same dir as input.")
    return p.parse_args()


def main():
    args = parse_args()

    # Load predictions
    df = pd.read_csv(args.input, parse_dates=["date"])
    print(f"\nLoaded {len(df)} rows from {args.input}")

    has_rf = {"rf_prediction", "rf_actual"}.issubset(df.columns)
    has_arima = {"arima_prediction", "arima_actual"}.issubset(df.columns)

    if not has_rf and not has_arima:
        print("ERROR: No model prediction columns found.")
        sys.exit(1)

    models = []
    if has_rf:
        models.append(("RF", "rf_prediction", "rf_actual"))
    if has_arima:
        models.append(("ARIMA", "arima_prediction", "arima_actual"))

    print(f"Models found: {', '.join(m[0] for m in models)}")
    print(f"Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"N days: {len(df)}")

    pos_size = args.position_size
    cost_kwargs = dict(
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        short_cost=args.short_cost,
        price_impact=args.price_impact,
    )

    all_results = []

    # Run per-model strategies
    for model_name, pred_col, actual_col in models:
        preds = df[pred_col].values
        actuals = df[actual_col].values
        dates = df["date"].values

        # DirectionalSignal
        positions = [0.0]  # day 0 has no signal
        for i in range(1, len(dates)):
            positions.append(directional_signal(preds[i], actuals[i - 1]) * pos_size)

        metrics, equity = run_model_backtest(dates, actuals, positions, **cost_kwargs)
        all_results.append({
            "strategy": "DirectionalSignal",
            "model": model_name,
            "metrics": metrics,
            "equity": equity,
            "dates": dates,
        })

        # ThresholdSignal
        positions = [0.0]
        for i in range(1, len(dates)):
            positions.append(threshold_signal(preds[i], actuals[i - 1], args.threshold) * pos_size)

        metrics, equity = run_model_backtest(dates, actuals, positions, **cost_kwargs)
        all_results.append({
            "strategy": f"ThresholdSignal({args.threshold:.1%})",
            "model": model_name,
            "metrics": metrics,
            "equity": equity,
            "dates": dates,
        })

    # EnsembleConsensus (only if both models available)
    if has_rf and has_arima:
        rf_preds = df["rf_prediction"].values
        rf_actuals = df["rf_actual"].values
        arima_preds = df["arima_prediction"].values
        arima_actuals = df["arima_actual"].values
        dates = df["date"].values

        positions = [0.0]
        for i in range(1, len(dates)):
            positions.append(ensemble_consensus(
                rf_preds[i], rf_actuals[i - 1],
                arima_preds[i], arima_actuals[i - 1],
            ) * pos_size)

        metrics, equity = run_model_backtest(dates, rf_actuals, positions, **cost_kwargs)
        all_results.append({
            "strategy": "EnsembleConsensus",
            "model": "RF+ARIMA",
            "metrics": metrics,
            "equity": equity,
            "dates": dates,
        })

    # Buy & Hold benchmark
    ref_actual_col = models[0][2]
    ref_actuals = df[ref_actual_col].values
    bh_return = (ref_actuals[-1] - ref_actuals[0]) / ref_actuals[0]
    bh_equity = [args.initial_capital * (1 + (ref_actuals[i] - ref_actuals[0]) / ref_actuals[0])
                 for i in range(len(ref_actuals))]

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  Strategy Comparison")
    print(f"{'=' * 100}")

    header = (
        f"{'Strategy':<25s} {'Model':<10s} {'Tot.Ret.':>10s} {'Ann.Ret.':>10s} "
        f"{'Sharpe':>8s} {'MaxDD':>8s} {'WinRate':>8s} {'DirAcc':>8s} "
        f"{'#Trades':>8s} {'PF':>8s}"
    )
    print(f"{'-' * len(header)}")
    print(header)
    print(f"{'-' * len(header)}")

    for r in all_results:
        m = r["metrics"]
        pf = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
        print(
            f"{r['strategy']:<25s} {r['model']:<10s} "
            f"{m['total_return']:>+10.2%} {m['annualized_return']:>+10.2%} "
            f"{m['sharpe_ratio']:>8.2f} {m['max_drawdown']:>8.2%} "
            f"{m['win_rate']:>8.1%} {m['directional_accuracy']:>8.1%} "
            f"{m['n_trades']:>8d} {pf:>8s}"
        )

    print(f"{'-' * len(header)}")
    print(f"{'Buy & Hold':<25s} {'':10s} {bh_return:>+10.2%}")
    print(f"{'-' * len(header)}")

    print(f"\nCost assumptions:")
    print(f"  Position size  : {args.position_size:.2%} of capital per trade")
    print(f"  Fee per side   : {args.fee_rate:.2%}")
    print(f"  Slippage       : {args.slippage:.2%}")
    print(f"  Short cost/day : {args.short_cost:.2%}")
    print(f"  Price impact   : {args.price_impact:.4f} (quadratic in position size)")

    n_days = len(df) - 1
    if n_days < 100:
        print(f"NOTE: Only {n_days} trading days — annualized metrics may be unreliable.")

    # ── Per-model analysis ───────────────────────────────────────────
    for model_name, pred_col, actual_col in models:
        preds = df[pred_col].values
        actuals = df[actual_col].values

        errors = preds[1:] - actuals[1:]
        abs_errors = np.abs(errors)
        pct_errors = abs_errors / actuals[1:] * 100

        # Directional accuracy
        correct = sum(
            1 for i in range(1, len(actuals))
            if (preds[i] > actuals[i - 1]) == (actuals[i] > actuals[i - 1])
        )
        dir_acc = correct / (len(actuals) - 1)

        print(f"\n--- {model_name} Prediction Quality ---")
        print(f"  Mean Absolute Error  : ${np.mean(abs_errors):,.2f}")
        print(f"  Median Absolute Error: ${np.median(abs_errors):,.2f}")
        print(f"  Mean % Error         : {np.mean(pct_errors):.2f}%")
        print(f"  Directional Accuracy : {dir_acc:.1%} ({correct}/{len(actuals) - 1})")

        # Correlation
        corr = np.corrcoef(preds[1:], actuals[1:])[0, 1]
        print(f"  Prediction-Actual r  : {corr:.4f}")

        # Predicted vs actual direction agreement over time windows
        for window in [30, 60, 90]:
            if len(actuals) > window:
                recent_correct = sum(
                    1 for i in range(len(actuals) - window, len(actuals))
                    if (preds[i] > actuals[i - 1]) == (actuals[i] > actuals[i - 1])
                )
                print(f"  Dir. accuracy (last {window}d): {recent_correct / window:.1%}")

    # ── Plot equity curves ───────────────────────────────────────────
    plot_path = args.output_plot or str(Path(args.input).parent / "backtest_models_equity.png")

    fig, ax = plt.subplots(figsize=(14, 7))

    for r in all_results:
        dates = pd.to_datetime(r["dates"])
        ax.plot(dates, r["equity"], linewidth=1.2,
                label=f"{r['strategy']} ({r['model']})")

    dates = pd.to_datetime(df["date"].values)
    ax.plot(dates, bh_equity, linewidth=1.5, color="black", linestyle=":",
            label="Buy & Hold")

    ax.axhline(y=args.initial_capital, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5)

    ax.set_title("Model Backtest: Equity Curves")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nEquity curve plot -> {plot_path}")


if __name__ == "__main__":
    main()
