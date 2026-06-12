#!/usr/bin/env python
"""Regime breakdown of V2 strategy returns.

Replays baseline_strategy_v2 against two prediction directories
(baseline + PIT), captures daily PnL per coin, labels each day by BTC
drawdown regime (bull / sideways / bear), and reports per-regime
Sharpe for each variant.

Usage:
    python scripts/regime_breakdown.py \
        --baseline-dir data/multi_2c_5yr_baseline \
        --pit-dir       data/multi_2c_5yr_pit
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_strategy_v2 import (  # noqa: E402
    apply_trend_filter,
    build_positions_with_hold,
    compute_realized_vol,
    generate_term_structure_signals,
    load_horizon_predictions,
    vol_regime_mask,
)


# Costs from V2 default
FEE = 0.0010
SLIPPAGE = 0.0010
SPREAD = 0.0005
IMPACT = 0.0010
FUNDING = 0.0001
INITIAL = 10_000.0
STOP = 0.03
PORT_DD = 0.15


def regime_label(prices: np.ndarray, window: int = 365, bull_dd: float = 0.10,
                 bear_dd: float = 0.30) -> np.ndarray:
    """Label each day by drawdown from rolling `window`-day high.

    Bull: DD < bull_dd. Sideways: bull_dd <= DD < bear_dd. Bear: DD >= bear_dd.
    """
    s = pd.Series(prices)
    rolling_max = s.rolling(window=window, min_periods=1).max()
    dd = (rolling_max - s) / rolling_max
    labels = np.full(len(s), "bull", dtype=object)
    labels[(dd >= bull_dd) & (dd < bear_dd)] = "sideways"
    labels[dd >= bear_dd] = "bear"
    return labels


def replay_coin(
    coin_df: pd.DataFrame, horizons: list[int],
) -> pd.DataFrame:
    """Replay strategy for a single coin and return DataFrame with
    columns: date, price, position, daily_return."""
    df = coin_df.sort_values("date").reset_index(drop=True)
    signals, confidence = generate_term_structure_signals(
        df, horizons, confidence_ref=0.05, asymmetric=False,
    )

    prices = df["ref_price"].astype(float).values
    dates = pd.to_datetime(df["date"]).values

    realized_vol = compute_realized_vol(prices, lookback=30)
    vol_ok = vol_regime_mask(realized_vol, percentile_cap=0.95)
    sized = build_positions_with_hold(
        signals=signals, vol_ok=vol_ok, confidence=confidence,
        realized_vol=realized_vol, prices=prices,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    sized = apply_trend_filter(sized, prices, sma_period=30, multiplier=1.5)

    daily_returns = np.zeros(len(prices))
    prev_pos = 0.0
    entry_equity = INITIAL
    equity = INITIAL
    peak = INITIAL
    halted = False

    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if np.isnan(p0) or np.isnan(p1) or p0 == 0 or halted:
            prev_pos = sized[i] if not halted else 0.0
            continue
        target = sized[i]
        notional = abs(target - prev_pos)
        if target != prev_pos and target != 0:
            entry_equity = equity
        gross = target * (p1 - p0) / p0
        cost = (2 * FEE + SLIPPAGE + 2 * SPREAD) * notional + IMPACT * notional ** 2 + FUNDING * abs(target)
        net = gross - cost
        new_eq = equity * (1 + net)
        if target != 0 and entry_equity > 0:
            trade_dd = (entry_equity - new_eq) / entry_equity
            if trade_dd >= STOP:
                target = 0.0
        daily_returns[i] = net
        equity = new_eq
        peak = max(peak, equity)
        if peak > 0 and (peak - equity) / peak >= PORT_DD:
            halted = True
        prev_pos = target

    return pd.DataFrame({
        "date": dates, "price": prices, "position": sized,
        "daily_return": daily_returns,
    })


def replay_variant(pred_dir: Path, horizons: list[int]) -> dict[str, pd.DataFrame]:
    merged = load_horizon_predictions(pred_dir, horizons)
    out: dict[str, pd.DataFrame] = {}
    for coin in sorted(merged["coin_id"].unique()):
        coin_df = merged[merged["coin_id"] == coin].copy()
        out[coin] = replay_coin(coin_df, horizons)
    return out


def per_regime_sharpe(returns: np.ndarray, mask: np.ndarray, traded: np.ndarray) -> tuple[float, float, int]:
    """Return (sharpe, mean_daily_return, n_traded_days) for the masked days."""
    sub = returns[mask & traded]
    if len(sub) < 2:
        return float("nan"), float("nan"), int(np.sum(mask & traded))
    daily_rf = (1 + 0.045) ** (1 / 252) - 1
    excess = sub - daily_rf
    std = np.std(excess, ddof=1)
    sharpe = float(np.mean(excess) / std * np.sqrt(252)) if std > 0 else 0.0
    return sharpe, float(np.mean(sub)), int(len(sub))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", required=True)
    p.add_argument("--pit-dir", required=True)
    p.add_argument("--horizons", nargs="+", type=int, default=[7, 14])
    p.add_argument("--rolling-high-window", type=int, default=365)
    p.add_argument("--bull-dd", type=float, default=0.10)
    p.add_argument("--bear-dd", type=float, default=0.30)
    return p.parse_args()


def main():
    args = parse_args()
    base = replay_variant(Path(args.baseline_dir), args.horizons)
    pit = replay_variant(Path(args.pit_dir), args.horizons)

    # Build BTC regime label using BTC actual prices across the full OOS window.
    btc_base = base.get("bitcoin")
    if btc_base is None:
        raise RuntimeError("BTC predictions required for regime labelling")
    regime = regime_label(
        btc_base["price"].values,
        window=args.rolling_high_window,
        bull_dd=args.bull_dd,
        bear_dd=args.bear_dd,
    )
    btc_dates = pd.to_datetime(btc_base["date"])

    print(f"\n{'=' * 70}\n  Regime Breakdown — V2 strategy on {len(btc_dates)} OOS days\n{'=' * 70}")
    print(f"  Window for rolling high: {args.rolling_high_window}d")
    print(f"  Regimes: bull DD<{args.bull_dd:.0%}, sideways <{args.bear_dd:.0%}, bear ≥{args.bear_dd:.0%}\n")

    regimes = ["bull", "sideways", "bear"]
    n_per_regime = {r: int(np.sum(regime == r)) for r in regimes}
    print(f"  Regime day counts: {n_per_regime}\n")

    rows = []
    for coin in ["bitcoin", "ethereum"]:
        if coin not in base or coin not in pit:
            continue
        # Align via date intersection
        bdf = base[coin].set_index("date")
        pdf = pit[coin].set_index("date")
        common = bdf.index.intersection(pdf.index)
        bdf = bdf.loc[common].sort_index()
        pdf = pdf.loc[common].sort_index()

        # Map BTC regime to coin dates
        btc_regime_series = pd.Series(regime, index=btc_base["date"]).reindex(common).values

        b_ret = bdf["daily_return"].values
        p_ret = pdf["daily_return"].values
        b_pos = np.abs(bdf["position"].values) > 1e-9
        p_pos = np.abs(pdf["position"].values) > 1e-9

        for r in regimes:
            mask = btc_regime_series == r
            b_sh, b_mu, b_n = per_regime_sharpe(b_ret, mask, b_pos)
            p_sh, p_mu, p_n = per_regime_sharpe(p_ret, mask, p_pos)
            rows.append({
                "coin": coin, "regime": r, "n_days": int(np.sum(mask)),
                "n_trades_base": b_n, "n_trades_pit": p_n,
                "sharpe_base": b_sh, "sharpe_pit": p_sh,
                "delta_sharpe": p_sh - b_sh,
                "mean_ret_base_bps": b_mu * 10_000 if not np.isnan(b_mu) else np.nan,
                "mean_ret_pit_bps": p_mu * 10_000 if not np.isnan(p_mu) else np.nan,
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()

    # Portfolio-level regime breakdown (equal-weight)
    print(f"\n{'=' * 70}\n  Portfolio (equal-weight) Sharpe per regime\n{'=' * 70}")
    if "bitcoin" in base and "ethereum" in base:
        bbtc = base["bitcoin"].set_index("date")["daily_return"]
        beth = base["ethereum"].set_index("date")["daily_return"]
        pbtc = pit["bitcoin"].set_index("date")["daily_return"]
        peth = pit["ethereum"].set_index("date")["daily_return"]
        common = bbtc.index.intersection(beth.index).intersection(pbtc.index).intersection(peth.index)
        port_base = 0.5 * (bbtc.loc[common].values + beth.loc[common].values)
        port_pit = 0.5 * (pbtc.loc[common].values + peth.loc[common].values)
        btc_regime_series = pd.Series(regime, index=btc_base["date"]).reindex(common).values

        port_rows = []
        for r in regimes:
            mask = btc_regime_series == r
            traded_mask = np.ones(len(port_base), dtype=bool)
            b_sh, b_mu, _ = per_regime_sharpe(port_base, mask, traded_mask)
            p_sh, p_mu, _ = per_regime_sharpe(port_pit, mask, traded_mask)
            port_rows.append({
                "regime": r, "n_days": int(np.sum(mask)),
                "sharpe_base": b_sh, "sharpe_pit": p_sh,
                "delta_sharpe": p_sh - b_sh,
                "mean_ret_base_bps": b_mu * 10_000 if not np.isnan(b_mu) else np.nan,
                "mean_ret_pit_bps": p_mu * 10_000 if not np.isnan(p_mu) else np.nan,
            })
        port_df = pd.DataFrame(port_rows)
        print(port_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()


if __name__ == "__main__":
    main()
