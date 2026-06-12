#!/usr/bin/env python
"""Solid baseline trading strategy using RF + ARIMA predictions.

A principled composite strategy designed as a meaningful benchmark for the
multi-agent LLM system. Combines ensemble agreement, signal persistence,
volatility targeting, Kelly sizing, conditional leverage, and risk management.

Usage:
    python scripts/baseline_strategy.py --input data/eval_predictions.csv
    python scripts/baseline_strategy.py --input data/eth/eval_predictions.csv \
        --max-leverage 3 --target-vol 0.10 --persistence 2
"""

from __future__ import annotations

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


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline composite strategy backtest on model predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to eval_predictions.csv.")
    p.add_argument("--initial-capital", type=float, default=10_000.0)

    # Strategy parameters
    p.add_argument("--persistence", type=int, default=2,
                    help="Days of consecutive signal agreement before entry.")
    p.add_argument("--target-vol", type=float, default=0.10,
                    help="Target annualized portfolio volatility (0.10 = 10%%).")
    p.add_argument("--vol-lookback", type=int, default=20,
                    help="Days for rolling volatility estimate.")
    p.add_argument("--kelly-fraction", type=float, default=0.5,
                    help="Kelly scaling (0.5 = half-Kelly).")
    p.add_argument("--max-leverage", type=float, default=3.0,
                    help="Maximum leverage (conditional on confidence).")
    p.add_argument("--stop-loss", type=float, default=0.03,
                    help="Per-trade stop-loss as fraction (0.03 = 3%%).")
    p.add_argument("--max-portfolio-dd", type=float, default=0.15,
                    help="Portfolio circuit breaker drawdown from peak.")
    p.add_argument("--vol-cap-pct", type=float, default=0.95,
                    help="Skip trading when realized vol exceeds this percentile.")
    p.add_argument("--confidence-ref-return", type=float, default=0.02,
                    help="Predicted return that maps to full confidence (0.02 = 2%%).")

    # Cost parameters
    p.add_argument("--fee-rate", type=float, default=0.001,
                    help="Exchange fee per side (0.001 = 0.1%%).")
    p.add_argument("--slippage", type=float, default=0.001,
                    help="Slippage per trade.")
    p.add_argument("--spread", type=float, default=0.0005,
                    help="Bid-ask spread (half-spread per side).")
    p.add_argument("--price-impact", type=float, default=0.001,
                    help="Price impact coefficient (quadratic in position).")
    p.add_argument("--funding-rate", type=float, default=0.0001,
                    help="Daily funding rate for leveraged perpetuals.")

    p.add_argument("--output-plot", default=None,
                    help="Equity curve plot path. Default: same dir as input.")
    return p.parse_args()


def load_predictions(path: Path) -> pd.DataFrame:
    """Load eval_predictions.csv and validate required columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    required = {"rf_prediction", "rf_actual", "arima_prediction", "arima_actual"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    return df.sort_values("date").reset_index(drop=True)


def generate_ensemble_signals(df: pd.DataFrame) -> np.ndarray:
    """Generate +1/-1/0 ensemble signals where both models must agree."""
    rf_preds = df["rf_prediction"].values
    rf_actuals = df["rf_actual"].values
    arima_preds = df["arima_prediction"].values
    arima_actuals = df["arima_actual"].values

    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        rf_dir = 1 if rf_preds[i] > rf_actuals[i - 1] else -1
        arima_dir = 1 if arima_preds[i] > arima_actuals[i - 1] else -1
        if rf_dir == arima_dir:
            signals[i] = rf_dir
    return signals


def apply_persistence_filter(signals: np.ndarray, days: int) -> np.ndarray:
    """Zero out signals that haven't persisted for `days` consecutive bars."""
    if days <= 1:
        return signals.copy()

    filtered = np.zeros_like(signals)
    for i in range(days - 1, len(signals)):
        window = signals[i - days + 1 : i + 1]
        if np.all(window == 1):
            filtered[i] = 1
        elif np.all(window == -1):
            filtered[i] = -1
    return filtered


def compute_realized_vol(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling annualized realized volatility from log returns."""
    log_ret = np.full(len(prices), np.nan)
    log_ret[1:] = np.log(prices[1:] / prices[:-1])

    vol = np.full(len(prices), np.nan)
    for i in range(lookback, len(prices)):
        window = log_ret[i - lookback + 1 : i + 1]
        window = window[~np.isnan(window)]
        if len(window) >= 2:
            vol[i] = np.std(window, ddof=1) * np.sqrt(252)
    return vol


def vol_regime_mask(vol: np.ndarray, percentile_cap: float) -> np.ndarray:
    """Return boolean mask: True = OK to trade, False = vol too high.

    Uses strict expanding-window quantile (history up to but NOT including
    the current bar) to prevent lookahead.
    """
    mask = np.ones(len(vol), dtype=bool)
    for i in range(len(vol)):
        if np.isnan(vol[i]):
            mask[i] = False
            continue
        history = vol[:i]  # Exclude current bar to avoid lookahead
        history = history[~np.isnan(history)]
        if len(history) < 20:
            continue
        threshold = np.quantile(history, percentile_cap)
        if vol[i] > threshold:
            mask[i] = False
    return mask


def compute_confidence(df: pd.DataFrame, ref_return: float) -> np.ndarray:
    """Confidence score in [0, 1] based on predicted return magnitude."""
    rf_preds = df["rf_prediction"].values
    rf_actuals = df["rf_actual"].values
    arima_preds = df["arima_prediction"].values
    arima_actuals = df["arima_actual"].values

    confidence = np.zeros(len(df))
    for i in range(1, len(df)):
        rf_ret = abs(rf_preds[i] - rf_actuals[i - 1]) / rf_actuals[i - 1]
        arima_ret = abs(arima_preds[i] - arima_actuals[i - 1]) / arima_actuals[i - 1]
        avg = (rf_ret + arima_ret) / 2
        confidence[i] = min(1.0, avg / ref_return)
    return confidence


def vol_targeted_size(
    signal: int,
    confidence: float,
    realized_vol: float,
    target_vol: float,
    kelly_fraction: float,
) -> float:
    """Compute position size using vol targeting + Kelly + confidence."""
    if signal == 0 or np.isnan(realized_vol) or realized_vol <= 0:
        return 0.0
    base = target_vol / realized_vol
    return float(signal) * kelly_fraction * base * confidence


def apply_leverage(base_size: float, confidence: float, max_leverage: float) -> float:
    """Scale position by conditional leverage based on confidence."""
    if base_size == 0:
        return 0.0
    lev = 1 + (max_leverage - 1) * confidence
    sized = base_size * lev
    if abs(sized) > max_leverage:
        sized = np.sign(sized) * max_leverage
    return float(sized)


def build_positions(
    signals: np.ndarray,
    vol_ok: np.ndarray,
    confidence: np.ndarray,
    realized_vol: np.ndarray,
    target_vol: float,
    kelly_fraction: float,
    max_leverage: float,
) -> np.ndarray:
    """Build position series with exit-only-on-flip semantics."""
    positions = np.zeros(len(signals))
    current_pos = 0.0
    current_dir = 0

    for i in range(len(signals)):
        sig = int(signals[i])

        if current_dir == 0 and sig != 0 and vol_ok[i]:
            base = vol_targeted_size(
                sig, confidence[i], realized_vol[i], target_vol, kelly_fraction,
            )
            current_pos = apply_leverage(base, confidence[i], max_leverage)
            current_dir = sig

        elif current_dir != 0 and sig != 0 and sig != current_dir and vol_ok[i]:
            base = vol_targeted_size(
                sig, confidence[i], realized_vol[i], target_vol, kelly_fraction,
            )
            current_pos = apply_leverage(base, confidence[i], max_leverage)
            current_dir = sig

        positions[i] = current_pos

    return positions


def run_baseline_backtest(
    dates: np.ndarray,
    prices: np.ndarray,
    positions: np.ndarray,
    initial_capital: float,
    fee_rate: float,
    slippage: float,
    spread: float,
    price_impact: float,
    funding_rate: float,
    stop_loss: float,
    max_portfolio_dd: float,
) -> tuple[list, dict]:
    """Run baseline backtest with full risk and cost model."""
    equity = [initial_capital]
    daily_returns = []
    effective_positions = [0.0]
    prev_pos = 0.0
    entry_equity = initial_capital
    peak_equity = initial_capital
    halted = False

    for i in range(1, len(dates)):
        p_prev = prices[i - 1]
        p_curr = prices[i]

        if np.isnan(p_prev) or np.isnan(p_curr) or p_prev == 0:
            daily_returns.append(0.0)
            equity.append(equity[-1])
            effective_positions.append(prev_pos)
            continue

        if halted:
            daily_returns.append(0.0)
            equity.append(equity[-1])
            effective_positions.append(0.0)
            prev_pos = 0.0
            continue

        target_pos = positions[i]
        trade_notional = abs(target_pos - prev_pos)

        if target_pos != prev_pos and target_pos != 0:
            entry_equity = equity[-1]
        if target_pos == 0:
            entry_equity = equity[-1]

        price_return = (p_curr - p_prev) / p_prev
        gross_ret = target_pos * price_return

        fee_cost = (2 * fee_rate + slippage + 2 * spread) * trade_notional
        impact_cost = price_impact * trade_notional * trade_notional
        holding_cost = funding_rate * abs(target_pos)

        total_cost = fee_cost + impact_cost + holding_cost
        net_ret = gross_ret - total_cost

        daily_returns.append(net_ret)
        new_equity = equity[-1] * (1 + net_ret)

        if target_pos != 0 and entry_equity > 0:
            trade_dd = (entry_equity - new_equity) / entry_equity
            if trade_dd >= stop_loss:
                target_pos = 0.0

        equity.append(new_equity)
        effective_positions.append(target_pos)
        prev_pos = target_pos

        peak_equity = max(peak_equity, new_equity)
        dd_from_peak = (peak_equity - new_equity) / peak_equity if peak_equity > 0 else 0
        if dd_from_peak >= max_portfolio_dd:
            halted = True

    returns = np.array(daily_returns)
    eff_pos = np.array(effective_positions[1:])
    traded_mask = np.abs(eff_pos) > 1e-9
    traded_returns = returns[traded_mask]

    total_return = (equity[-1] - initial_capital) / initial_capital
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    if len(traded_returns) > 1:
        excess = traded_returns - daily_rf
        std_ex = np.std(excess, ddof=1)
        sharpe = float(np.mean(excess) / std_ex * np.sqrt(252)) if std_ex > 0 else 0
    else:
        sharpe = 0

    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = np.where(running_max > 0, (running_max - eq) / running_max, 0)
    max_dd = float(np.max(dd))

    n_trades = int(np.sum(np.diff(eff_pos) != 0))
    wins = int(np.sum(traded_returns > 0))
    win_rate = wins / len(traded_returns) if len(traded_returns) > 0 else 0

    gross_profit = float(np.sum(traded_returns[traded_returns > 0]))
    gross_loss = float(np.abs(np.sum(traded_returns[traded_returns < 0])))
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0)

    metrics = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
        "profit_factor": pf,
        "halted": halted,
    }
    return equity, metrics


def main():
    args = parse_args()
    df = load_predictions(Path(args.input))
    print(f"\nLoaded {len(df)} rows from {args.input}")
    print(f"Date range: {df['date'].min():%Y-%m-%d} to {df['date'].max():%Y-%m-%d}")

    # Signal pipeline
    raw_signals = generate_ensemble_signals(df)
    filtered = apply_persistence_filter(raw_signals, args.persistence)

    print(f"\nRaw ensemble signals: long={int((raw_signals==1).sum())}  "
          f"short={int((raw_signals==-1).sum())}  flat={int((raw_signals==0).sum())}")
    print(f"After {args.persistence}-day persistence: long={int((filtered==1).sum())}  "
          f"short={int((filtered==-1).sum())}  flat={int((filtered==0).sum())}")

    # Volatility and regime filter
    ref_prices = df["rf_actual"].values
    realized_vol = compute_realized_vol(ref_prices, args.vol_lookback)
    vol_ok = vol_regime_mask(realized_vol, args.vol_cap_pct)

    tradeable = (filtered != 0) & vol_ok
    print(f"After vol regime filter: active bars = {int(tradeable.sum())}")

    # Confidence + position building
    confidence = compute_confidence(df, args.confidence_ref_return)
    positions = build_positions(
        filtered, vol_ok, confidence, realized_vol,
        args.target_vol, args.kelly_fraction, args.max_leverage,
    )

    pos_changes = int(np.sum(np.diff(positions) != 0))
    nonzero_bars = int(np.sum(np.abs(positions) > 1e-9))
    print(f"\nPosition state: {nonzero_bars} bars in position, {pos_changes} entry/flip events")

    # Run backtest
    prices = df["rf_actual"].values
    dates = df["date"].values

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

    # Parameter summary
    print(f"\n{'=' * 60}")
    print(f"  Strategy Parameters")
    print(f"{'=' * 60}")
    print(f"  Persistence days   : {args.persistence}")
    print(f"  Target vol         : {args.target_vol:.1%} annualized")
    print(f"  Vol lookback       : {args.vol_lookback} days")
    print(f"  Kelly fraction     : {args.kelly_fraction}")
    print(f"  Max leverage       : {args.max_leverage}x")
    print(f"  Stop-loss          : {args.stop_loss:.1%}")
    print(f"  Portfolio DD limit : {args.max_portfolio_dd:.1%}")
    print(f"  Vol cap percentile : {args.vol_cap_pct:.0%}")

    print(f"\n{'=' * 60}")
    print(f"  Cost Assumptions")
    print(f"{'=' * 60}")
    print(f"  Fee per side       : {args.fee_rate:.2%}")
    print(f"  Slippage           : {args.slippage:.2%}")
    print(f"  Bid-ask spread     : {args.spread:.2%}")
    print(f"  Price impact       : {args.price_impact:.4f} (quadratic)")
    print(f"  Funding rate/day   : {args.funding_rate:.2%}")

    print(f"\n{'=' * 60}")
    print(f"  Baseline Strategy Results")
    print(f"{'=' * 60}")
    print(f"  Total Return       : {metrics['total_return']:+.2%}")
    print(f"  Annualized Return  : {metrics['annualized_return']:+.2%}")
    print(f"  Sharpe Ratio       : {metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"  Win Rate           : {metrics['win_rate']:.1%}")
    print(f"  # Trades           : {metrics['n_trades']}")
    pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "inf"
    print(f"  Profit Factor      : {pf_str}")
    print(f"  Circuit breaker    : {'TRIGGERED' if metrics['halted'] else 'OK'}")

    bh_return = (prices[-1] - prices[0]) / prices[0]
    print(f"\n  Buy & Hold         : {bh_return:+.2%}")

    # Plot equity curve
    plot_path = args.output_plot or str(Path(args.input).parent / "baseline_equity.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    dates_pd = pd.to_datetime(dates)

    ax.plot(dates_pd, equity, linewidth=1.5, label="Baseline Strategy", color="tab:blue")

    bh_equity = [args.initial_capital * (1 + (prices[i] - prices[0]) / prices[0])
                 for i in range(len(prices))]
    ax.plot(dates_pd, bh_equity, linewidth=1.5, linestyle=":", color="black",
            label="Buy & Hold")

    ax.axhline(y=args.initial_capital, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5, label="Initial Capital")

    ax.set_title("Baseline Strategy vs Buy & Hold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nEquity curve plot -> {plot_path}")


if __name__ == "__main__":
    main()
