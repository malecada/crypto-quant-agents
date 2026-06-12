#!/usr/bin/env python
"""BT7 — Core ablation table for the hybrid quant+LLM stack.

Reads cached hybrid signals (CSV per coin from generate_hybrid_signals.py),
re-computes the hybrid `position` under various ablations, then runs
the V2 cost+risk pipeline on each and reports Sharpe / return / MaxDD.

Configurations:
  full           — original multiplier × effective_weight as emitted
  no_layer2      — multiplier=1.0 everywhere (pure Layer 1 quant)
  no_regime      — effective_weight uses single 0.5 weight per bar
                   (kills regime-conditional weighting)
  no_unlock_veto — set unlock_flag=False (keeps any signal)
  no_uncertainty — uncertainty dampener disabled (k=0)
  no_edge        — edge dampener disabled (k=0)
  static_50      — fixed effective_weight=0.5
  static_100     — fixed effective_weight=1.0 (max LLM influence)
  llm_only       — quant_magnitude replaced by sign(direction) × 0.5

Usage:
    python scripts/ablate_hybrid.py \
        --signals-dir data/hybrid_signals_p1 \
        --coins bitcoin ethereum \
        --start 2026-01-16 --end 2026-04-15
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
from tradingagents.strategies.effective_weight import (  # noqa: E402
    DEFAULT_REGIME_WEIGHT, compute_effective_weight,
)


COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)


def _load_prices(coin: str, end: str) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coin, end)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def _signed(direction: str, mag: float) -> float:
    sign = {"long": 1, "short": -1, "flat": 0}.get(direction, 0)
    return float(sign * abs(mag))


def _compose_position(
    quant_mag: float, direction: str, multiplier: float,
    effective_weight: float,
) -> float:
    """Composition formula identical to strategies.modulator.apply_modulator."""
    sign = {"long": 1, "short": -1, "flat": 0}.get(direction, 0)
    base = float(sign * abs(quant_mag))
    return float(base * (1.0 + effective_weight * (multiplier - 1.0)))


def _reweight(
    regime: str, uncertainty: float, edge, unlock_flag: bool,
    mode: str,
) -> float:
    """Recompute effective_weight under the named ablation."""
    if mode == "no_regime":
        # Single neutral weight regardless of regime
        return 0.0 if unlock_flag else 0.5
    if mode == "no_unlock_veto":
        return compute_effective_weight(regime, uncertainty, edge, False)
    if mode == "no_uncertainty":
        return compute_effective_weight(
            regime, 0.0, edge, unlock_flag, uncertainty_dampener_k=0.0,
        )
    if mode == "no_edge":
        return compute_effective_weight(
            regime, uncertainty, None, unlock_flag, edge_dampener_k=0.0,
        )
    if mode == "static_50":
        return 0.0 if unlock_flag else 0.5
    if mode == "static_100":
        return 0.0 if unlock_flag else 1.0
    # default — use the row's stored effective_weight where present
    return float("nan")


def _build_v2_baseline_positions(sig: pd.DataFrame, prices_df: pd.DataFrame) -> np.ndarray:
    """Apply the production V2 pipeline to the raw LGB consensus signals.

    Lets us compare hybrid against the same V2 sizing baseline used in
    the published Sharpe-2.69 result, with the LGB signal series already
    aligned to the hybrid window.
    """
    from tradingagents.strategies.v2_sizing import (
        apply_trend_filter, build_positions_with_hold, compute_realized_vol,
        vol_regime_mask,
    )
    n = len(sig)
    sign = sig["quant_direction"].map({"long": 1, "short": -1, "flat": 0}).fillna(0).values
    conf = sig["quant_magnitude"].abs().values  # confidence from generate_term_structure_signals
    raw_signal = sign * np.sign(conf)
    px = prices_df["Close"].astype(float).values[:n]
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=raw_signal, vol_ok=mask, confidence=conf,
        realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    return apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)


def _build_positions(sig: pd.DataFrame, mode: str) -> np.ndarray:
    n = len(sig)
    out = np.zeros(n, dtype=float)
    for i, row in enumerate(sig.itertuples(index=False)):
        direction = row.quant_direction
        qmag = float(row.quant_magnitude or 0.0)
        mult = float(row.llm_multiplier or 1.0)
        unc = float(row.llm_uncertainty or 0.0)
        edge = (
            None if pd.isna(getattr(row, "rolling_llm_edge", np.nan))
            else float(row.rolling_llm_edge)
        )
        unlock = bool(getattr(row, "unlock_flag", False))
        regime = str(getattr(row, "regime", "sideways"))
        stored_w = float(row.effective_weight or 0.0)

        if mode == "full":
            out[i] = _compose_position(qmag, direction, mult, stored_w)
        elif mode == "no_layer2":
            out[i] = _signed(direction, qmag)
        elif mode == "llm_only":
            sign = {"long": 1, "short": -1, "flat": 0}.get(direction, 0)
            scale = 0.5
            # Keep multiplier shape but with constant magnitude
            w = stored_w
            out[i] = sign * scale * (1.0 + w * (mult - 1.0))
        else:
            w = _reweight(regime, unc, edge, unlock, mode)
            out[i] = _compose_position(qmag, direction, mult, w)
    return out


def _run_ablation(
    coin: str, signals_csv: Path, start: str, end: str, mode: str,
) -> dict:
    sig = pd.read_csv(signals_csv, parse_dates=["date"])
    sig["date"] = sig["date"].dt.tz_localize(None).dt.normalize()
    sig = sig[(sig["date"] >= start) & (sig["date"] <= end)].copy()
    sig = sig.dropna(subset=["position"]).reset_index(drop=True)

    prices = _load_prices(coin, end)
    merged = sig.merge(prices[["Date", "Close"]], left_on="date", right_on="Date")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)

    if mode == "v2_baseline_pipeline":
        positions = _build_v2_baseline_positions(merged, merged)
    else:
        positions = _build_positions(sig, mode)
    pos = positions[: len(merged)]
    px = merged["Close"].astype(float).values
    dates = merged["date"].values

    equity, m = run_coin_backtest(
        dates=dates, prices=px, positions=pos,
        initial_capital=10_000.0, **COSTS,
    )
    return {
        "mode": mode,
        "sharpe_ratio": float(m.get("sharpe_ratio", float("nan"))),
        "total_return": float(m.get("total_return", float("nan"))),
        "max_drawdown": float(m.get("max_drawdown", float("nan"))),
        "win_rate": float(m.get("win_rate", float("nan"))),
        "n_trades": int(m.get("n_trades", 0)),
        "mean_position": float(np.mean(np.abs(pos))),
        "n_active": int((np.abs(pos) > 1e-9).sum()),
    }


MODES = [
    "full", "no_layer2", "v2_baseline_pipeline", "no_regime", "no_unlock_veto",
    "no_uncertainty", "no_edge", "static_50", "static_100", "llm_only",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--signals-dir", required=True)
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    out_dir = Path(args.output_dir or f"{args.signals_dir}/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for coin in args.coins:
        sig_csv = Path(args.signals_dir) / f"{coin}_{args.start}_{args.end}.csv"
        if not sig_csv.exists():
            print(f"[skip] {coin}: missing {sig_csv}")
            continue
        results[coin] = []
        for mode in MODES:
            r = _run_ablation(coin, sig_csv, args.start, args.end, mode)
            results[coin].append(r)

    # Print table
    print(f"\n{'=' * 92}")
    print(f"  Ablation table — hybrid quant+LLM ({args.start} -> {args.end})")
    print(f"{'=' * 92}\n")
    for coin in args.coins:
        if coin not in results:
            continue
        print(f"  {coin}")
        full_sr = next(r["sharpe_ratio"] for r in results[coin] if r["mode"] == "full")
        print(f"    {'mode':<18} {'SR':>7}  {'ΔSR':>7}  {'ret':>8}  {'MaxDD':>7}  "
              f"{'win':>5}  {'|pos|':>6}  {'active':>6}")
        for r in results[coin]:
            d = r["sharpe_ratio"] - full_sr
            print(f"    {r['mode']:<18} {r['sharpe_ratio']:>7.2f}  {d:>+7.2f}  "
                  f"{r['total_return']:>7.1%}  {r['max_drawdown']:>6.1%}  "
                  f"{r['win_rate']:>4.0%}   {r['mean_position']:>6.3f}  "
                  f"{r['n_active']:>6d}")
        print()

    out_path = out_dir / "ablation.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
