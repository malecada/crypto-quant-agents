#!/usr/bin/env python
"""BT10 — Regime-conditional eval for V2 quant baseline walk-forward.

Per BACKTESTING_METHODOLOGY.md §6: take the per-bar daily returns from
walkforward_v2.py, label each bar with its detected regime (HMM-3 +
heuristic via tradingagents.strategies.regime.detect_regime), then
compute per-regime Sharpe / return / hit rate. Stratifies the
aggregated 4.5-yr Sharpe by regime so we can test:

  H1 (FINSABER): does the V2 baseline alpha persist across regimes,
                 or is it regime-dependent?
  H2 (modulator thesis): is there a regime where LLM modulation has
                         room to add value (e.g. sideways, where
                         narrative drives short-term)?

Uses BT12-style regime-conditional bootstrap: separate returns by
regime, block-bootstrap within each regime, report CIs.

Usage:
    python scripts/regime_breakdown_v2.py \\
        --returns data/walkforward_v2_2coin/daily_returns.csv \\
        --coins bitcoin ethereum \\
        --output-dir data/walkforward_v2_2coin
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

from scripts.bootstrap_sharpe import (  # type: ignore  # noqa: E402
    DAILY_RF, sharpe, stationary_bootstrap_sample,
)
from tradingagents.strategies.regime import (  # noqa: E402
    build_regime_features, hurst_exponent, smooth_label_sequence,
)
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402


REGIMES = ("bull", "sideways", "bear")


def _label_series_vectorized(coin: str, dates: list[str], end: str) -> list[str]:
    """Vectorized regime labelling — load OHLCV once, label every bar.

    Uses the same heuristic_label thresholds as detect_regime but
    rolling over the full price series at once. Optional HMM agreement
    pass kept off here for speed (heuristic dominates anyway per
    Phase 1 finding that HMM mis-clusters on price-only features).
    """
    df = _load_crypto_ohlcv(coin, end)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("Date").reset_index(drop=True)
    prices = pd.Series(df["Close"].values, index=df["Date"].values)

    log_ret = np.log(prices / prices.shift(1))
    rv20 = log_ret.rolling(20).std() * np.sqrt(252)
    rv_p90 = rv20.quantile(0.9)

    # 30-day log return
    ret_30 = np.log(prices / prices.shift(30))

    # Rolling 200-bar Hurst (cheap approximation: pre-compute once)
    n = len(prices)
    hurst = np.full(n, 0.5, dtype=float)
    for i in range(200, n):
        hurst[i] = hurst_exponent(prices.values[i - 200:i + 1])

    labels = np.full(n, "sideways", dtype=object)
    for i in range(30, n):
        r30 = ret_30.iloc[i]
        rv_i = rv20.iloc[i]
        h_i = hurst[i]
        if pd.isna(r30) or pd.isna(rv_i):
            continue
        # heuristic_label thresholds — identical to regime.py
        if r30 < -0.10 and rv_i > rv_p90 * 0.7:
            labels[i] = "bear"
        elif r30 < -0.05:
            labels[i] = "bear"
        elif r30 > 0.10 and h_i > 0.5:
            labels[i] = "bull"
        elif r30 > 0.05 and h_i > 0.5:
            labels[i] = "bull"
        else:
            labels[i] = "sideways"

    smoothed = smooth_label_sequence(labels, window=3)
    by_date = {pd.Timestamp(d): smoothed[i] for i, d in enumerate(prices.index)}

    out = []
    for d in dates:
        ts = pd.Timestamp(d)
        out.append(str(by_date.get(ts, "sideways")))
    return out


def _bootstrap_sr_ci(returns: np.ndarray, n_iter: int, block: int,
                     seed: int) -> tuple[float, float, float, float, float]:
    if len(returns) < 5:
        sr = sharpe(returns) if len(returns) > 1 else 0.0
        return sr, sr, sr, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = sharpe(returns)
    samples = np.empty(n_iter)
    for k in range(n_iter):
        samples[k] = sharpe(stationary_bootstrap_sample(returns, block, rng))
    return (point,
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
            float((samples > 0).mean()),
            float((samples > 1).mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--returns", required=True,
                   help="daily_returns.csv from walkforward_v2.py "
                        "(columns: date, coin, ret)")
    p.add_argument("--coins", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-iter", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = pd.read_csv(args.returns, parse_dates=["date"])
    daily["date"] = daily["date"].dt.tz_localize(None).dt.normalize()
    daily["date_str"] = daily["date"].dt.strftime("%Y-%m-%d")

    print(f"\n{'=' * 84}")
    print(f"  BT10 regime-conditional eval  ({len(daily)} bar-coin obs)")
    print(f"{'=' * 84}\n")

    summary: dict = {}
    for coin in args.coins:
        sub = daily[daily["coin"] == coin].sort_values("date").reset_index(drop=True)
        if sub.empty:
            print(f"[skip] {coin}")
            continue

        print(f"  {coin}: labelling {len(sub)} bars (vectorized) ...")
        end = sub["date_str"].max()
        sub["regime"] = _label_series_vectorized(coin, sub["date_str"].tolist(), end)
        # Save labelled CSV
        sub[["date", "ret", "regime"]].to_csv(
            out_dir / f"daily_returns_labelled_{coin}.csv", index=False,
        )

        print(f"\n  {coin} regime breakdown:")
        rows = []
        for regime in REGIMES + ("__all__",):
            if regime == "__all__":
                rets = sub["ret"].values
            else:
                rets = sub[sub["regime"] == regime]["ret"].values
            n = len(rets)
            if n == 0:
                continue
            sr, lo, hi, p_gt0, p_gt1 = _bootstrap_sr_ci(
                rets, args.n_iter, args.block_size, args.seed,
            )
            mean_ret = float(np.mean(rets))
            cum_ret = float(np.prod(1 + rets) - 1)
            hit_rate = float((rets > 0).mean())
            rows.append({
                "regime": regime, "n_bars": n,
                "sharpe": sr, "ci95_lo": lo, "ci95_hi": hi,
                "p_sr_gt_0": p_gt0, "p_sr_gt_1": p_gt1,
                "mean_daily_ret": mean_ret, "cum_return": cum_ret,
                "hit_rate": hit_rate,
            })
            print(f"    {regime:<10} n={n:>4d}  SR={sr:>+5.2f} "
                  f"CI=[{lo:>+5.2f},{hi:>+5.2f}]  P(>0)={p_gt0:.3f} "
                  f"P(>1)={p_gt1:.3f}  hit={hit_rate:.0%}  cum={cum_ret:+.2%}")
        summary[coin] = rows
        print()

    with open(out_dir / "regime_breakdown.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote: {out_dir / 'regime_breakdown.json'}")


if __name__ == "__main__":
    main()
