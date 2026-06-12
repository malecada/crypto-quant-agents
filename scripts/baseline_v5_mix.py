#!/usr/bin/env python
"""V5 MIX — canonical production strategy (4-coin per-coin-routed portfolio).

The strongest validated strategy in the TradingAgents thesis (§20):
portfolio Sharpe +3.2 / +787% / -4.9% max DD over a 4.5-year walk-forward.

Architecture:
  - V2 signal+sizing core (term-structure h7+h14 consensus → vol-targeted Kelly
    → conditional leverage → SMA30 trend filter → adaptive hold + stops).
  - Per-coin feature routing: each coin's LGB predictions come from the feature
    set that maximised its standalone Sharpe (§20). BTC/BNB use the 78-feature
    canonical pool; ETH/SOL use the 193-feature extended pool (+ Coinglass +
    PIT on-chain).
  - Equal-weight portfolio across the four coins (§19 confirmed weight
    optimization does not beat 1/N out-of-sample).
  - No regime overlay (§16 confirmed regime is a DD-reducer, not an alpha
    source, and the bundled NH-HMM is degenerate).

Attribution caveat (§21): a portfolio-level random-entry placebo attributes
~90% of the +3.2 SR to the V2 sizing layer + 4-coin diversification and only
~10% to LGB prediction signal. V5 MIX is best understood as a vol-targeted
multi-asset momentum strategy with a modest ML enhancement — not an
ML-prediction strategy. This makes it robust to prediction degradation.

This script consumes pre-generated walk-forward prediction CSVs (it does not
retrain). Generate them first with:
    scripts/evaluate_models_multi.py --coins bitcoin ethereum --horizons 7 14 \\
        --output-dir data/multi_2coins_walkforward                  # BTC route
    scripts/evaluate_models_multi.py ... --onchain-pit \\
        --output-dir data/multi_2coins_pit_wf                       # ETH route
    scripts/evaluate_models_multi.py --coins bitcoin ethereum binancecoin ... \\
        --output-dir data/multi_3coins_bnb_wf                       # BNB route
    scripts/evaluate_models_multi.py --coins bitcoin ethereum solana ... \\
        --onchain-pit --output-dir data/multi_3coins_sol_pit_wf     # SOL route

Usage:
    python scripts/baseline_v5_mix.py --start 2021-11-07 --end 2026-04-15
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

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

ANN = np.sqrt(252)
COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, take_profit=0.0, max_portfolio_dd=0.15,
)
EARLY_EXIT_DEFAULT = 0.015  # V2 canonical: matches build_positions_with_hold default

# --- 8-coin expansion: coin tiers + per-coin cost function -----------------
CORE_COINS = ("bitcoin", "ethereum", "binancecoin", "solana")
SATELLITE_COINS = ("ripple", "dogecoin", "cardano", "tron")
SATELLITE_HAIRCUT = 1.5            # conservative slippage/impact multiplier
SATELLITE_COST_KEYS = ("slippage", "price_impact")


def costs_for_coin(coin: str, sat_haircut: float = SATELLITE_HAIRCUT) -> dict:
    """Return the cost dict for a coin.

    Core coins get the legacy ``COSTS`` verbatim. Satellite coins get
    ``slippage`` and ``price_impact`` scaled by ``sat_haircut`` (default 1.5,
    a margin-of-safety for lower-cap perps). All other cost keys are shared.
    """
    c = dict(COSTS)
    if coin in SATELLITE_COINS:
        for k in SATELLITE_COST_KEYS:
            c[k] = COSTS[k] * sat_haircut
    return c


# --- 8-coin expansion: core/satellite portfolio weights --------------------
# Core coins 15% each (60% total), satellites 10% each (40% total).
PORTFOLIO_WEIGHTS = {
    "bitcoin": 0.15, "ethereum": 0.15, "binancecoin": 0.15, "solana": 0.15,
    "ripple": 0.10, "dogecoin": 0.10, "cardano": 0.10, "tron": 0.10,
}

# Canonical signal config that produced the published SR +3.18 (single source
# of truth for live parity — tests/execution/live assert the live LiveConfig
# defaults equal these).
V5_CONFIDENCE_REF = 0.05
V5_ASYMMETRIC = True


def portfolio_return(df: pd.DataFrame, weights: dict) -> pd.Series:
    """Weighted daily portfolio return series.

    ``df`` columns are per-coin daily return series. Weights are restricted
    to the columns present in ``df`` and renormalized to sum to 1, so a
    subset run (e.g. a 4-core-coin regression check) still produces a valid
    portfolio — and an equal-weight subset reproduces ``df.mean(axis=1)``.
    """
    cols = [c for c in weights if c in df.columns]
    if not cols:
        raise ValueError("no weighted coins present in df")
    w = pd.Series({c: weights[c] for c in cols}, dtype=float)
    w = w / w.sum()
    return (df[cols] * w).sum(axis=1)

# Per-coin feature routing → prediction directory (§20). The string after the
# arrow documents the feature set; the directory is what's actually loaded.
DEFAULT_ROUTING = {
    "bitcoin":     "data/multi_2coins_walkforward",   # 78f canonical  (frozen §20)
    "ethereum":    "data/multi_2coins_pit_wf",        # 193f extended  (frozen §20)
    "binancecoin": "data/multi_3coins_bnb_wf",        # 78f canonical  (frozen §20)
    "solana":      "data/multi_3coins_sol_pit_wf",    # 193f extended  (frozen §20)
    "ripple":      "data/multi_3coins_xrp_wf",        # 78f canonical  (T7 routed)
    "dogecoin":    "data/multi_3coins_doge_wf",       # 78f canonical  (T7 routed)
    "cardano":     "data/multi_3coins_ada_pit_wf",    # 193f extended  (T7 routed)
    "tron":        "data/multi_3coins_trx_wf",        # 78f canonical  (T7 routed)
}


def _load_preds(pred_dir: Path, coin: str) -> pd.DataFrame:
    p7 = pd.read_csv(pred_dir / "preds_lgb_h7.csv", parse_dates=["date"])
    p14 = pd.read_csv(pred_dir / "preds_lgb_h14.csv", parse_dates=["date"])
    for d in (p7, p14):
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    p7 = p7[p7["coin_id"] == coin].rename(columns={"prediction": "pred_h7"})
    p14 = p14[p14["coin_id"] == coin].rename(columns={"prediction": "pred_h14"})[["date", "pred_h14"]]
    return p7.merge(p14, on="date").sort_values("date").reset_index(drop=True)


def _v2_positions(
    merged: pd.DataFrame,
    kelly_fraction: float = 0.5,
    early_exit_loss: float = EARLY_EXIT_DEFAULT,
) -> np.ndarray:
    sig, conf = generate_term_structure_signals(
        merged, [7, 14], V5_CONFIDENCE_REF, asymmetric=V5_ASYMMETRIC,
    )
    px = merged["Close"].astype(float).values
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=sig, vol_ok=mask, confidence=conf, realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=kelly_fraction, max_leverage=3.0,
        min_hold=7, early_exit_loss=early_exit_loss,
    )
    return apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)


def _metrics(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    sd = r.std()
    return {
        "sharpe": float(r.mean() / sd * ANN) if sd > 0 else 0.0,
        "total_return": float(eq.iloc[-1] - 1.0),
        "max_drawdown": dd,
        "ann_vol": float(sd * ANN),
        "n_bars": int(len(r)),
    }


def run_coin(
    coin: str,
    pred_dir: Path,
    start: str,
    end: str,
    kelly_fraction: float = 0.5,
    early_exit_loss: float = EARLY_EXIT_DEFAULT,
    costs_override: dict[str, float] | None = None,
) -> pd.Series:
    """Run V2 sizing on one coin's routed predictions → daily return series.

    Early exit loss is forwarded to the position builder.
    Costs override (if supplied) replaces the COSTS dict passed to the engine —
    callers can override stop_loss and take_profit per-call.
    """
    preds = _load_preds(pred_dir, coin)
    preds = preds[(preds["date"] >= start) & (preds["date"] <= end)]
    if preds.empty:
        raise ValueError(f"{coin}: no predictions in [{start}, {end}] under {pred_dir}")
    ohlcv = _load_crypto_ohlcv(coin, end)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.tz_localize(None).dt.normalize()
    merged = preds.merge(ohlcv[["Date", "Close"]], left_on="date", right_on="Date")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    merged["ref_price"] = merged["Close"]

    pos = _v2_positions(
        merged, kelly_fraction=kelly_fraction, early_exit_loss=early_exit_loss,
    )
    costs = dict(COSTS if costs_override is None else costs_override)
    equity, _m = run_coin_backtest(
        dates=merged["date"].values, prices=merged["Close"].values,
        positions=pos, initial_capital=10_000.0, **costs,
    )
    eq = np.asarray(equity, dtype=float)
    rets = eq[1:] / eq[:-1] - 1.0
    return pd.Series(rets, index=pd.to_datetime(merged["date"].values[1:]), name=coin)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", default="2021-11-07")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--output-dir", default="data/v5_mix_production")
    p.add_argument("--routing-json", default=None,
                   help="Optional JSON {coin: pred_dir} overriding DEFAULT_ROUTING")
    p.add_argument("--kelly", type=float, default=0.5,
                   help="Kelly fraction for V2 sizing (default 0.5 = backtest canonical, "
                        "use 0.25 for live margin re-run)")
    p.add_argument("--data-root", default=None,
                   help="Override TRADINGAGENTS_DATA_ROOT for this run "
                        "(sandbox parity replay)")
    p.add_argument("--sat-haircut", type=float, default=SATELLITE_HAIRCUT,
                   help="Satellite-coin slippage/impact multiplier "
                        "(default 1.5; sweep 1.0/1.5/2.0 for sensitivity)")
    args = p.parse_args()

    if args.data_root:
        os.environ["TRADINGAGENTS_DATA_ROOT"] = args.data_root

    routing = DEFAULT_ROUTING
    if args.routing_json:
        routing = json.loads(Path(args.routing_json).read_text())

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 78}")
    print(f"  V5 MIX — {len(routing)}-coin portfolio (core/satellite weighted)")
    print(f"  window: {args.start} → {args.end}   sat-haircut: {args.sat_haircut:.2f}x")
    print(f"{'=' * 78}\n")

    coin_rets: dict[str, pd.Series] = {}
    for coin, pdir in routing.items():
        r = run_coin(coin, PROJECT_ROOT / pdir, args.start, args.end,
                     kelly_fraction=args.kelly,
                     costs_override=costs_for_coin(coin, args.sat_haircut))
        coin_rets[coin] = r
        m = _metrics(r)
        feat = "193f extended" if "pit" in pdir else "78f canonical"
        tier = "satellite" if coin in SATELLITE_COINS else "core     "
        print(f"  {coin:12s} [{feat:14s}] [{tier}] SR={m['sharpe']:+.2f}  "
              f"ret={m['total_return']:+8.1%}  maxDD={m['max_drawdown']:6.1%}  ({pdir})")

    df = pd.DataFrame(coin_rets).dropna().sort_index()
    port = portfolio_return(df, PORTFOLIO_WEIGHTS)  # core/satellite weighted
    pm = _metrics(port)

    print(f"\n  {'-' * 74}")
    print(f"  {len(coin_rets)}-coin V5 MIX portfolio (core/satellite weighted, {pm['n_bars']} bars):")
    print(f"    Sharpe            : {pm['sharpe']:+.3f}")
    print(f"    Compounded return : {pm['total_return']:+.1%}")
    print(f"    Max drawdown      : {pm['max_drawdown']:.1%}")
    print(f"    Annualized vol    : {pm['ann_vol']:.1%}")
    print(f"  {'-' * 74}\n")

    df["portfolio"] = port
    df.to_csv(out_dir / "daily_returns.csv")
    summary = {
        "window": {"start": args.start, "end": args.end},
        "routing": routing,
        "per_coin": {c: _metrics(coin_rets[c]) for c in coin_rets},
        "portfolio": pm,
        "correlation": df[list(coin_rets)].corr().to_dict(),
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Wrote: {out_dir / 'daily_returns.csv'}")
    print(f"  Wrote: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
