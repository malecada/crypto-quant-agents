#!/usr/bin/env python
"""Baseline Strategy V2: Multi-Horizon Term Structure Consensus.

Reads LGB multi-horizon predictions and backtests per-coin with:
- Term structure consensus (h=7 + h=14 must agree on direction)
- Minimum 7-day hold period
- Vol-targeted Kelly sizing with conditional leverage
- Per-trade stop-loss + portfolio circuit breaker

Usage:
    python scripts/baseline_strategy_v2.py --pred-dir data/multi_5coins_v2
    python scripts/baseline_strategy_v2.py --pred-dir data/multi_2coins_v2 --coins bitcoin ethereum
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

from tradingagents.strategies.v2_sizing import (
    generate_term_structure_signals,
    compute_realized_vol,
    vol_regime_mask,
    vol_targeted_size,
    apply_leverage,
    apply_trend_filter,
    build_positions_with_hold,
)


# ── Data Loading ─────────────────────────────────────────────────────


def load_horizon_predictions(pred_dir: Path, horizons: list[int]) -> pd.DataFrame:
    """Load and merge prediction CSVs for multiple horizons.

    Returns DataFrame with columns: date, coin_id, ref_price, actual_h7,
    pred_h7, actual_h14, pred_h14, etc.
    """
    dfs = []
    for h in horizons:
        path = pred_dir / f"preds_lgb_h{h}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        df = pd.read_csv(path, parse_dates=["date"])
        df = df.rename(columns={
            "prediction": f"pred_h{h}",
            "actual": f"actual_h{h}",
        })
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(
            df.drop(columns=["ref_price"]),
            on=["date", "coin_id"],
            how="inner",
        )
    return merged.sort_values(["coin_id", "date"]).reset_index(drop=True)


# ── Signal Generation, Sizing, Trend Filter ──────────────────────────
# Imported from tradingagents.strategies.v2_sizing — see the import block
# near the top of this module. The functions previously defined inline
# here now live in that module so the live trading cycle can reuse them.


def run_coin_backtest(
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
    take_profit: float = 0.0,
) -> tuple[list, dict]:
    """Run backtest for a single coin with full cost and risk model."""
    equity = [initial_capital]
    daily_returns = []
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
            continue

        if halted:
            daily_returns.append(0.0)
            equity.append(equity[-1])
            prev_pos = 0.0
            continue

        target_pos = positions[i]
        trade_notional = abs(target_pos - prev_pos)

        if target_pos != prev_pos and target_pos != 0:
            entry_equity = equity[-1]
        if target_pos == 0 and prev_pos != 0:
            entry_equity = equity[-1]

        price_return = (p_curr - p_prev) / p_prev
        gross_ret = target_pos * price_return

        fee_cost = (2 * fee_rate + slippage + 2 * spread) * trade_notional
        impact_cost = price_impact * trade_notional * trade_notional
        holding_cost = funding_rate * abs(target_pos)
        total_cost = fee_cost + impact_cost + holding_cost
        net_ret = gross_ret - total_cost

        new_equity = equity[-1] * (1 + net_ret)

        if target_pos != 0 and entry_equity > 0:
            trade_dd = (entry_equity - new_equity) / entry_equity
            if trade_dd >= stop_loss:
                target_pos = 0.0
            trade_up = (new_equity - entry_equity) / entry_equity
            if take_profit > 0 and trade_up >= take_profit:
                target_pos = 0.0

        daily_returns.append(net_ret)
        equity.append(new_equity)
        prev_pos = target_pos

        peak_equity = max(peak_equity, new_equity)
        dd_from_peak = (peak_equity - new_equity) / peak_equity if peak_equity > 0 else 0
        if dd_from_peak >= max_portfolio_dd:
            halted = True

    # Compute metrics
    returns = np.array(daily_returns)
    total_return = (equity[-1] - initial_capital) / initial_capital
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0

    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    traded_mask = np.abs(np.array([positions[i] for i in range(1, len(dates))])) > 1e-9
    traded_returns = returns[traded_mask]

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

    n_trades = int(np.sum(np.abs(np.diff(positions)) > 1e-9))
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


# ── CLI ──────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline Strategy V2: Multi-Horizon Term Structure Consensus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pred-dir", required=True, help="Directory with preds_lgb_h*.csv files.")
    p.add_argument("--coins", nargs="+", default=None,
                    help="Subset of coins to trade (default: all in CSVs).")
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14],
                    help="Horizons for consensus.")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--min-hold", type=int, default=7,
                    help="Minimum days to hold a position before allowing exit.")
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
    p.add_argument("--symmetric", action="store_true",
                    help="Use symmetric consensus (both horizons must agree for longs too).")
    p.add_argument("--early-exit-loss", type=float, default=0.015,
                    help="Loss threshold for early exit from losing positions.")
    p.add_argument("--trend-sma", type=int, default=30,
                    help="SMA period for trend filter (0 to disable).")
    p.add_argument("--trend-multiplier", type=float, default=1.5,
                    help="Position scaling when aligned/against trend.")
    p.add_argument("--arima-filter", action="store_true",
                    help="Require ARIMA h=1 agreement as additional consensus filter.")
    p.add_argument("--output-plot", default=None)
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()
    pred_dir = Path(args.pred_dir)

    print(f"\n{'=' * 70}")
    print(f"  Baseline Strategy V2: Term Structure Consensus")
    print(f"{'=' * 70}")

    # Load predictions
    merged = load_horizon_predictions(pred_dir, args.horizons)

    # Optionally load ARIMA h=1 for additional consensus filter
    arima_df = None
    if args.arima_filter:
        arima_path = pred_dir / "preds_arima_h1.csv"
        if not arima_path.exists():
            print(f"  WARNING: --arima-filter set but {arima_path} not found, ignoring")
            args.arima_filter = False
        else:
            arima_df = pd.read_csv(arima_path, parse_dates=["date"])
            arima_df = arima_df.rename(columns={"prediction": "arima_pred", "actual": "arima_actual"})
            if "ref_price" in arima_df.columns:
                arima_df = arima_df.drop(columns=["ref_price"])
            merged = merged.merge(arima_df, on=["date", "coin_id"], how="left")

    coins = args.coins or sorted(merged["coin_id"].unique())
    print(f"  Pred dir    : {pred_dir}")
    print(f"  Horizons    : {args.horizons}")
    print(f"  ARIMA filter: {'yes' if args.arima_filter else 'no'}")
    print(f"  Asymmetric  : {'no (symmetric)' if args.symmetric else 'yes'}")
    print(f"  Trend SMA   : {args.trend_sma}d (multiplier {args.trend_multiplier}x)" if args.trend_sma > 0 else "  Trend SMA   : disabled")
    print(f"  Early exit  : {args.early_exit_loss:.1%} loss threshold")
    print(f"  Coins       : {', '.join(coins)}")
    print(f"  Min hold    : {args.min_hold} days")
    print(f"  Max leverage: {args.max_leverage}x")
    print(f"  Date range  : {merged['date'].min():%Y-%m-%d} to {merged['date'].max():%Y-%m-%d}")

    cost_kwargs = dict(
        fee_rate=args.fee_rate, slippage=args.slippage, spread=args.spread,
        price_impact=args.price_impact, funding_rate=args.funding_rate,
        stop_loss=args.stop_loss, max_portfolio_dd=args.max_portfolio_dd,
    )

    all_results = {}
    all_equity = {}
    all_bh = {}

    for coin in coins:
        df_coin = merged[merged["coin_id"] == coin].sort_values("date").reset_index(drop=True)
        if len(df_coin) < 30:
            print(f"\n  {coin}: skipped (only {len(df_coin)} rows)")
            continue

        # Signals
        signals, confidence = generate_term_structure_signals(
            df_coin, args.horizons, args.confidence_ref_return,
            asymmetric=not args.symmetric,
        )

        # ARIMA veto: zero out signals where ARIMA h=1 disagrees
        if args.arima_filter and "arima_pred" in df_coin.columns:
            ref = df_coin["ref_price"].values
            arima_pred = df_coin["arima_pred"].values
            for i in range(len(signals)):
                if signals[i] != 0 and not np.isnan(arima_pred[i]) and ref[i] > 0:
                    arima_dir = 1 if arima_pred[i] > ref[i] else -1
                    if arima_dir != int(signals[i]):
                        signals[i] = 0
                        confidence[i] = 0

        # Volatility
        prices = df_coin["ref_price"].values
        dates = df_coin["date"].values
        realized_vol = compute_realized_vol(prices, args.vol_lookback)
        vol_ok = vol_regime_mask(realized_vol, args.vol_cap_pct)

        # Positions
        positions = build_positions_with_hold(
            signals, vol_ok, confidence, realized_vol, prices,
            args.target_vol, args.kelly_fraction, args.max_leverage,
            args.min_hold, args.early_exit_loss,
        )

        # Trend filter
        if args.trend_sma > 0:
            positions = apply_trend_filter(
                positions, prices, args.trend_sma, args.trend_multiplier,
            )

        # Use ref_price for the backtest (price at prediction time)
        equity, metrics = run_coin_backtest(
            dates, prices, positions,
            initial_capital=args.initial_capital,
            **cost_kwargs,
        )

        # Buy & Hold
        bh_ret = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0

        all_results[coin] = metrics
        all_equity[coin] = equity
        all_bh[coin] = bh_ret

        n_signals = int(np.sum(signals != 0))
        n_agree = int(np.sum(signals != 0))
        print(f"\n  {coin}: signals={n_signals}  trades={metrics['n_trades']}  "
              f"return={metrics['total_return']:+.2%}  B&H={bh_ret:+.2%}")

    # ── Per-Coin Results Table ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Per-Coin Results")
    print(f"{'=' * 70}")

    header = (f"{'Coin':<12s} {'Return':>10s} {'Ann.Ret':>10s} {'Sharpe':>8s} "
              f"{'MaxDD':>8s} {'WinRate':>8s} {'#Trades':>8s} {'vs B&H':>10s}")
    print(f"  {'-' * len(header)}")
    print(f"  {header}")
    print(f"  {'-' * len(header)}")

    for coin in coins:
        if coin not in all_results:
            continue
        m = all_results[coin]
        bh = all_bh[coin]
        vs_bh = m["total_return"] - bh
        pf_str = f"{m['profit_factor']:.2f}" if m["profit_factor"] != float("inf") else "inf"
        print(f"  {coin:<12s} {m['total_return']:>+10.2%} {m['annualized_return']:>+10.2%} "
              f"{m['sharpe_ratio']:>8.2f} {m['max_drawdown']:>8.2%} "
              f"{m['win_rate']:>8.1%} {m['n_trades']:>8d} {vs_bh:>+10.2%}")

    print(f"  {'-' * len(header)}")

    # Buy & Hold row
    for coin in coins:
        if coin in all_bh:
            print(f"  {coin + ' B&H':<12s} {all_bh[coin]:>+10.2%}")

    # ── Equal-Weight Portfolio ───────────────────────────────────────
    if len(all_equity) > 1:
        min_len = min(len(eq) for eq in all_equity.values())
        port_equity = np.zeros(min_len)
        for eq in all_equity.values():
            port_equity += np.array(eq[:min_len])
        port_equity /= len(all_equity)

        port_return = (port_equity[-1] - args.initial_capital) / args.initial_capital
        port_daily = np.diff(port_equity) / port_equity[:-1]
        port_daily = port_daily[~np.isnan(port_daily)]
        daily_rf = (1 + 0.045) ** (1 / 252) - 1
        if len(port_daily) > 1:
            excess = port_daily - daily_rf
            std_ex = np.std(excess, ddof=1)
            port_sharpe = float(np.mean(excess) / std_ex * np.sqrt(252)) if std_ex > 0 else 0
        else:
            port_sharpe = 0
        running_max = np.maximum.accumulate(port_equity)
        port_dd = np.where(running_max > 0, (running_max - port_equity) / running_max, 0)
        port_max_dd = float(np.max(port_dd))

        print(f"\n{'=' * 70}")
        print(f"  Equal-Weight Portfolio ({len(all_equity)} coins)")
        print(f"{'=' * 70}")
        print(f"  Return     : {port_return:+.2%}")
        print(f"  Sharpe     : {port_sharpe:.2f}")
        print(f"  Max DD     : {port_max_dd:.2%}")

    # ── Cost Assumptions ─────────────────────────────────────────────
    print(f"\n  Cost: fee={args.fee_rate:.2%}/side  slip={args.slippage:.2%}  "
          f"spread={args.spread:.2%}  impact={args.price_impact:.4f}  "
          f"funding={args.funding_rate:.2%}/day")

    # ── Plot ─────────────────────────────────────────────────────────
    plot_path = args.output_plot or str(pred_dir / "baseline_v2_equity.png")

    fig, ax = plt.subplots(figsize=(16, 8))

    for coin in coins:
        if coin not in all_equity:
            continue
        eq = all_equity[coin]
        # Get dates for this coin
        df_coin = merged[merged["coin_id"] == coin].sort_values("date")
        dates_pd = pd.to_datetime(df_coin["date"].values)
        # equity has len(dates)+1 entries (index 0 = initial capital before first bar)
        # eq[1:] aligns with dates[0:] only when len(eq)-1 == len(dates_pd)
        # In practice eq may be shorter if halted; trim both to the same length
        n_plot = min(len(eq) - 1, len(dates_pd))
        ax.plot(dates_pd[:n_plot], eq[1:n_plot + 1], linewidth=1.2,
                label=f"{coin} ({all_results[coin]['total_return']:+.1%})")

    if len(all_equity) > 1:
        # Portfolio line
        ref_coin = coins[0] if coins[0] in all_equity else list(all_equity.keys())[0]
        df_ref = merged[merged["coin_id"] == ref_coin].sort_values("date")
        port_dates = pd.to_datetime(df_ref["date"].values)
        n_port = min(min_len - 1, len(port_dates))
        ax.plot(port_dates[:n_port], port_equity[1:n_port + 1], linewidth=2, color="black",
                linestyle="-", label=f"Portfolio ({port_return:+.1%})")

    ax.axhline(y=args.initial_capital, color="gray", linestyle="--",
               linewidth=0.8, alpha=0.5)

    ax.set_title("Baseline V2: Term Structure Consensus (per coin)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (USD)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\n  Equity plot -> {plot_path}")


if __name__ == "__main__":
    main()
