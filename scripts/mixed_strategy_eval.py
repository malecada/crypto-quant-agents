#!/usr/bin/env python
"""Mix-and-match per-coin: V2 quant for some coins, LLM hybrid for others.

Per-coin policy thesis: BTC consistently underperformed in LLM
backtests across P1-P5. V2 quant on BTC gives Sharpe 2.42 for the
89-day window. ETH/BNB benefit more from LLM signal. A simple
per-coin policy — quant for BTC, hybrid LLM for ETH — should
outperform either uniform strategy.

Usage:
    python scripts/mixed_strategy_eval.py \\
        --quant-coins bitcoin --quant-pred-dir data/multi_2coins_v2 \\
        --llm-coins ethereum --llm-signals-dir data/agent_signals_pit_p4 \\
        --start 2026-01-16 --end 2026-04-15 \\
        --hybrid-pred-dir data/multi_2coins_v2 \\
        --hybrid-agree-weight 2.0 --hybrid-disagree-weight 0.3 --hybrid-conf-cap 2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def _slice_window(dates: np.ndarray, equity: list[float], positions: np.ndarray,
                  start: str, end: str) -> tuple[np.ndarray, np.ndarray]:
    """Slice equity curve + positions to date window. Returns (rets, positions_in_window)."""
    ts = pd.to_datetime(dates, utc=True).tz_localize(None)
    win_start = pd.to_datetime(start)
    win_end = pd.to_datetime(end)
    mask = (ts >= win_start) & (ts <= win_end)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), np.array([])
    i0, i1 = max(idx[0], 1), idx[-1]
    eq = np.asarray(equity)[i0:i1 + 1]
    pos = positions[i0:i1 + 1]
    rets = np.diff(eq) / eq[:-1]
    return rets, pos[1:]


def quant_returns(coin: str, pred_dir: Path, args) -> tuple[np.ndarray, np.ndarray]:
    """Run V2 quant, slice to window, return (daily_returns, positions_in_window)."""
    merged = load_horizon_predictions(pred_dir, args.horizons)
    df_coin = merged[merged["coin_id"] == coin].sort_values("date").reset_index(drop=True)
    if len(df_coin) < 30:
        raise RuntimeError(f"{coin}: only {len(df_coin)} prediction rows in {pred_dir}")
    signals, confidence = generate_term_structure_signals(
        df_coin, args.horizons, args.confidence_ref_return, asymmetric=False,
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
    equity, _ = run_coin_backtest(
        dates, prices, positions, initial_capital=args.initial_capital,
        fee_rate=args.fee_rate, slippage=args.slippage, spread=args.spread,
        price_impact=args.price_impact, funding_rate=args.funding_rate,
        stop_loss=args.stop_loss, max_portfolio_dd=args.max_portfolio_dd,
    )
    return _slice_window(dates, equity, positions, args.start, args.end)


def llm_hybrid_returns(coin: str, signals_dir: Path, args) -> tuple[np.ndarray, np.ndarray]:
    """Run LLM hybrid via backtest_system_v2 module, return (rets, positions)."""
    # Import here to avoid pulling LLM deps unless needed
    from backtest_system_v2 import (  # type: ignore
        load_signal_csv, fetch_prices, signals_to_positions_v2,
    )
    signals_df = load_signal_csv(signals_dir, coin, args.start, args.end)
    prices_df = fetch_prices(coin, args.start, args.end)
    merged = signals_df.merge(prices_df, on="date", how="inner").sort_values("date")
    if len(merged) < 30:
        raise RuntimeError(f"{coin}: only {len(merged)} aligned rows")
    dates = merged["date"].values
    prices = merged["prices"].values.astype(float)
    realized_vol = compute_realized_vol(prices, args.vol_lookback)
    vol_ok = vol_regime_mask(realized_vol, args.vol_cap_pct)
    args._current_coin = coin
    positions = signals_to_positions_v2(merged, prices, realized_vol, vol_ok, args)
    equity, _ = run_coin_backtest(
        dates, prices, positions, initial_capital=args.initial_capital,
        fee_rate=args.fee_rate, slippage=args.slippage, spread=args.spread,
        price_impact=args.price_impact, funding_rate=args.funding_rate,
        stop_loss=args.stop_loss, max_portfolio_dd=args.max_portfolio_dd,
    )
    rets = np.diff(np.asarray(equity)) / np.asarray(equity)[:-1]
    pos = positions[1:]
    # Slice to window
    ts = pd.to_datetime(dates, utc=True).tz_localize(None)
    win_start = pd.to_datetime(args.start)
    win_end = pd.to_datetime(args.end)
    mask_full = (ts >= win_start) & (ts <= win_end)
    # rets/pos arrays are dates[1:]; align mask accordingly
    return rets[mask_full[1:]], pos[mask_full[1:]]


def metrics_from_returns(rets: np.ndarray, positions: np.ndarray, label: str) -> dict:
    if len(rets) == 0:
        return {"label": label, "error": "empty"}
    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    traded = rets[np.abs(positions) > 1e-9] if len(positions) == len(rets) else rets
    sharpe = 0.0
    if len(traded) > 1:
        ex = traded - daily_rf
        std = np.std(ex, ddof=1)
        sharpe = float(np.mean(ex) / std * np.sqrt(252)) if std > 0 else 0.0
    eq = np.cumprod(1 + rets)
    rm = np.maximum.accumulate(eq)
    max_dd = float(np.max((rm - eq) / rm)) if rm[-1] > 0 else 0.0
    total_ret = float(eq[-1] - 1)
    ann_ret = (1 + total_ret) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0.0
    return {
        "label": label, "n_days": int(len(rets)),
        "total_return": total_ret, "annualized_return": ann_ret,
        "sharpe_ratio": sharpe, "max_drawdown": max_dd,
        "n_trades": int(np.sum(np.abs(np.diff(positions)) > 1e-9)) if len(positions) > 1 else 0,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Mix-and-match per-coin V2 quant vs LLM hybrid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--quant-coins", nargs="+", default=[])
    p.add_argument("--quant-pred-dir", default="data/multi_2coins_v2")
    p.add_argument("--llm-coins", nargs="+", default=[])
    p.add_argument("--llm-signals-dir", default="data/agent_signals_pit_p4")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
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
    p.add_argument("--early-exit-loss", type=float, default=0.015)
    p.add_argument("--trend-sma", type=int, default=30)
    p.add_argument("--trend-multiplier", type=float, default=1.5)
    p.add_argument("--confidence-ref-return", type=float, default=0.02)
    p.add_argument("--fee-rate", type=float, default=0.001)
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--spread", type=float, default=0.0005)
    p.add_argument("--price-impact", type=float, default=0.001)
    p.add_argument("--funding-rate", type=float, default=0.0001)
    # LLM hybrid params
    p.add_argument("--drop-low-confidence", action="store_true")
    p.add_argument("--hybrid-pred-dir", default=None)
    p.add_argument("--hybrid-agree-weight", type=float, default=2.0)
    p.add_argument("--hybrid-disagree-weight", type=float, default=0.3)
    p.add_argument("--hybrid-conf-cap", type=float, default=2.0)
    p.add_argument("--hybrid-horizons", nargs="+", type=int, default=[7, 14])
    p.add_argument("--high-confidence-boost", type=float, default=1.0)
    p.add_argument("--output-json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n{'=' * 70}")
    print(f"  Mixed-strategy eval: {args.start} → {args.end}")
    print(f"  Quant coins: {args.quant_coins}  (preds: {args.quant_pred_dir})")
    print(f"  LLM coins  : {args.llm_coins}    (signals: {args.llm_signals_dir})")
    print(f"  Hybrid     : aw={args.hybrid_agree_weight} cap={args.hybrid_conf_cap} dw={args.hybrid_disagree_weight}")
    print(f"{'=' * 70}\n")

    per_coin_metrics: dict = {}
    rets_streams: list[np.ndarray] = []
    coin_labels: list[str] = []

    for coin in args.quant_coins:
        rets, pos = quant_returns(coin, Path(args.quant_pred_dir), args)
        m = metrics_from_returns(rets, pos, f"{coin} (quant)")
        per_coin_metrics[coin] = m
        rets_streams.append(rets)
        coin_labels.append(f"{coin}/quant")
        print(f"  {coin:<14} (quant)  ret={m['total_return']*100:+.2f}%  "
              f"sharpe={m['sharpe_ratio']:+.2f}  maxDD={m['max_drawdown']*100:.2f}%")

    for coin in args.llm_coins:
        rets, pos = llm_hybrid_returns(coin, Path(args.llm_signals_dir), args)
        m = metrics_from_returns(rets, pos, f"{coin} (llm)")
        per_coin_metrics[coin] = m
        rets_streams.append(rets)
        coin_labels.append(f"{coin}/llm")
        print(f"  {coin:<14} (llm)    ret={m['total_return']*100:+.2f}%  "
              f"sharpe={m['sharpe_ratio']:+.2f}  maxDD={m['max_drawdown']*100:.2f}%")

    if not rets_streams:
        return

    # Equal-weight portfolio. Align all streams to min length.
    min_len = min(len(r) for r in rets_streams)
    mat = np.stack([r[:min_len] for r in rets_streams])
    port_rets = mat.mean(axis=0)
    port_eq = np.cumprod(1 + port_rets)

    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    ex = port_rets - daily_rf
    std = np.std(ex, ddof=1)
    sharpe = float(np.mean(ex) / std * np.sqrt(252)) if std > 0 else 0.0
    rm = np.maximum.accumulate(port_eq)
    max_dd = float(np.max((rm - port_eq) / rm))
    total = float(port_eq[-1] - 1)
    ann = (1 + total) ** (252 / min_len) - 1

    print(f"\n{'=' * 70}")
    print(f"  Portfolio (equal-weight, {len(rets_streams)} legs, {min_len} bars)")
    print(f"{'=' * 70}")
    print(f"  Return   : {total*100:+.2f}%")
    print(f"  Ann.Ret  : {ann*100:+.2f}%")
    print(f"  Sharpe   : {sharpe:+.2f}")
    print(f"  Max DD   : {max_dd*100:.2f}%")

    per_coin_metrics["_portfolio"] = {
        "total_return": total, "annualized_return": ann,
        "sharpe_ratio": sharpe, "max_drawdown": max_dd, "n_bars": int(min_len),
        "legs": coin_labels,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(per_coin_metrics, f, indent=2)
        print(f"\n  Metrics JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
