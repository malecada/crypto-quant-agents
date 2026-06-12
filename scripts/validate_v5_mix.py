#!/usr/bin/env python
"""V5 MIX validation — Deflated Sharpe Ratio + portfolio-level random-entry placebo.

The 4-coin V5 MIX headline (portfolio SR +3.25, §20) is the survivor of a search
across ~12-15 strategy variants this session. Two corrections are applied:

  1. **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): adjusts the
     observed SR for selection bias from the multiple-backtest search. Reported
     across a range of n_trials so the reader can see sensitivity.

  2. **Portfolio-level random-entry placebo** (BT11 extension): for K
     permutations, replace every coin's LGB direction call with a random ±1/0
     draw matching its empirical signal mix, keep the V2 sizing pipeline intact,
     rebuild the 25% EW portfolio, and collect the null SR distribution. If the
     observed +3.25 sits deep in the right tail, the edge is real signal; if the
     null mean is already high, the edge is sizing mechanics.

Per-coin feature routing (matches §20): BTC=78f, ETH=193f, BNB=78f, SOL=193f.

Usage:
    python scripts/validate_v5_mix.py --n-perms 1000
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
from tradingagents.strategies.v2_sizing import (  # noqa: E402
    apply_trend_filter, build_positions_with_hold, compute_realized_vol,
    generate_term_structure_signals, vol_regime_mask,
)
from tradingagents.strategies.v3.backtest.dsr import (  # noqa: E402
    deflated_sharpe_ratio, expected_max_sharpe, variance_of_sr,
)

ANN = np.sqrt(252)
COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)

# Per-coin feature routing → prediction directory (§20).
ROUTING = {
    "bitcoin":     "data/multi_2coins_walkforward",   # V2-78f canonical
    "ethereum":    "data/multi_2coins_pit_wf",        # V4-B-193f extended
    "binancecoin": "data/multi_3coins_bnb_wf",        # V2-78f canonical
    "solana":      "data/multi_3coins_sol_pit_wf",    # V4-B-193f extended
}

START, END = "2021-11-07", "2026-04-15"

# Strategy variants searched this session — used as n_trials for DSR.
# Conservative core count; DSR also reported for inflated counts.
SEARCH_TRIALS_CORE = 12  # V2, V3-base, V3+ext, V4-A nh-hmm, V4-A heur, V4-B,
                         # V4-B+regime, V5 2-coin, V5 4-coin, + 3 weight schemes


def _load_preds(pred_dir: Path, coin: str) -> pd.DataFrame:
    p7 = pd.read_csv(pred_dir / "preds_lgb_h7.csv", parse_dates=["date"])
    p14 = pd.read_csv(pred_dir / "preds_lgb_h14.csv", parse_dates=["date"])
    for d in (p7, p14):
        d["date"] = pd.to_datetime(d["date"]).dt.tz_localize(None).dt.normalize()
    p7 = p7[p7["coin_id"] == coin].rename(columns={"prediction": "pred_h7"})
    p14 = p14[p14["coin_id"] == coin].rename(columns={"prediction": "pred_h14"})[["date", "pred_h14"]]
    return p7.merge(p14, on="date").sort_values("date").reset_index(drop=True)


def _v2_pipeline(merged: pd.DataFrame, signals: np.ndarray, conf: np.ndarray) -> np.ndarray:
    px = merged["Close"].astype(float).values
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=signals, vol_ok=mask, confidence=conf, realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    return apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)


def _daily_returns(merged: pd.DataFrame, positions: np.ndarray) -> pd.Series:
    equity, _m = run_coin_backtest(
        dates=merged["date"].values, prices=merged["Close"].values,
        positions=positions, initial_capital=10_000.0, **COSTS,
    )
    eq = np.asarray(equity, dtype=float)
    rets = eq[1:] / eq[:-1] - 1.0
    idx = pd.to_datetime(merged["date"].values[1:])
    return pd.Series(rets, index=idx)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-perms", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="data/v5_validation")
    args = p.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # ── Load + observed per-coin pipeline ─────────────────────────────
    coin_merged: dict[str, pd.DataFrame] = {}
    coin_obs_sig: dict[str, np.ndarray] = {}
    coin_obs_conf: dict[str, np.ndarray] = {}
    coin_obs_rets: dict[str, pd.Series] = {}
    coin_sig_mix: dict[str, tuple] = {}

    for coin, pdir in ROUTING.items():
        preds = _load_preds(PROJECT_ROOT / pdir, coin)
        preds = preds[(preds["date"] >= START) & (preds["date"] <= END)]
        ohlcv = _load_crypto_ohlcv(coin, END)
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.tz_localize(None).dt.normalize()
        merged = preds.merge(ohlcv[["Date", "Close"]], left_on="date", right_on="Date")
        merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
        merged["ref_price"] = merged["Close"]
        coin_merged[coin] = merged

        sig, conf = generate_term_structure_signals(merged, [7, 14], 0.05, asymmetric=True)
        coin_obs_sig[coin] = sig
        coin_obs_conf[coin] = conf
        coin_obs_rets[coin] = _daily_returns(merged, _v2_pipeline(merged, sig, conf))

        n = len(sig)
        coin_sig_mix[coin] = (
            float((sig > 0).sum() / n),
            float((sig < 0).sum() / n),
            float((sig == 0).sum() / n),
        )

    # Observed portfolio: 25% EW of per-coin daily returns
    obs_df = pd.DataFrame(coin_obs_rets).dropna().sort_index()
    obs_port = obs_df.mean(axis=1)  # equal-weight
    sr_obs_perbar = float(obs_port.mean() / obs_port.std())
    sr_obs_ann = sr_obs_perbar * ANN
    n_bars = len(obs_port)

    print(f"\n{'=' * 84}")
    print(f"  V5 MIX VALIDATION  ({START} → {END})  n_bars={n_bars}  K={args.n_perms}")
    print(f"{'=' * 84}")
    print(f"\n  Observed 4-coin V5 MIX portfolio SR (annualized): {sr_obs_ann:+.3f}")

    # ── 1. Deflated Sharpe Ratio ──────────────────────────────────────
    var_sr = variance_of_sr(obs_port.values)
    se_sr = float(np.sqrt(var_sr))
    print(f"\n  [1] Deflated Sharpe Ratio (per-bar SR units)")
    print(f"      observed per-bar SR = {sr_obs_perbar:.5f}   SE(SR) = {se_sr:.5f}")
    dsr_rows = []
    for n_trials in (5, SEARCH_TRIALS_CORE, 25, 50, 100):
        e_max = expected_max_sharpe(n_trials, var_sr)
        dsr = deflated_sharpe_ratio(sr_obs_perbar, e_max, se_sr)
        dsr_rows.append({"n_trials": n_trials, "e_max_sr": e_max, "dsr": dsr})
        flag = "  ← session core" if n_trials == SEARCH_TRIALS_CORE else ""
        print(f"      n_trials={n_trials:>4}  E[max SR|null]={e_max:.5f}  "
              f"DSR=Φ(z)={dsr:.4f}{flag}")
    print(f"      (DSR > 0.95 → SR survives multiple-testing correction at that n_trials)")

    # ── 2. Portfolio-level random-entry placebo ───────────────────────
    print(f"\n  [2] Portfolio random-entry placebo  (K={args.n_perms})")
    for coin, (pl, ps, pf) in coin_sig_mix.items():
        print(f"      {coin:12s} signal mix long/short/flat = {pl:.0%}/{ps:.0%}/{pf:.0%}")

    choices = np.array([1.0, -1.0, 0.0])
    sr_perm = np.empty(args.n_perms)
    ret_perm = np.empty(args.n_perms)
    for k in range(args.n_perms):
        perm_rets = {}
        for coin, merged in coin_merged.items():
            pl, ps, pf = coin_sig_mix[coin]
            n = len(coin_obs_sig[coin])
            rand_sig = rng.choice(choices, size=n, p=[pl, ps, pf])
            rand_pos = _v2_pipeline(merged, rand_sig, np.abs(coin_obs_conf[coin]))
            perm_rets[coin] = _daily_returns(merged, rand_pos)
        pdf = pd.DataFrame(perm_rets).dropna().sort_index()
        pp = pdf.mean(axis=1)
        sr_perm[k] = float(pp.mean() / pp.std() * ANN) if pp.std() > 0 else 0.0
        ret_perm[k] = float((1 + pp).prod() - 1)

    p_value = float((sr_perm >= sr_obs_ann).mean())
    null_mean = float(sr_perm.mean())
    print(f"\n      Observed portfolio SR : {sr_obs_ann:+.3f}")
    print(f"      Random-entry null     : mean={null_mean:+.3f}  std={sr_perm.std(ddof=1):.3f}")
    print(f"        p05={np.quantile(sr_perm, 0.05):+.3f}  med={np.quantile(sr_perm, 0.50):+.3f}  "
          f"p95={np.quantile(sr_perm, 0.95):+.3f}  p99={np.quantile(sr_perm, 0.99):+.3f}  "
          f"max={sr_perm.max():+.3f}")
    print(f"      p-value (SR_perm >= SR_obs) : {p_value:.4f}")
    sig_sr = sr_obs_ann - null_mean
    pct_signal = sig_sr / sr_obs_ann * 100 if sr_obs_ann > 0 else float("nan")
    print(f"      Alpha attribution: signal={sig_sr:+.3f} SR ({pct_signal:.0f}%)  "
          f"sizing-floor={null_mean:+.3f} SR ({100-pct_signal:.0f}%)")

    # ── Verdict ───────────────────────────────────────────────────────
    print(f"\n  {'=' * 80}")
    dsr_core = next(r["dsr"] for r in dsr_rows if r["n_trials"] == SEARCH_TRIALS_CORE)
    verdict_dsr = "SURVIVES" if dsr_core > 0.95 else "FAILS"
    verdict_pl = "SURVIVES" if p_value < 0.05 else "FAILS"
    print(f"  DSR @ n_trials={SEARCH_TRIALS_CORE}: {dsr_core:.4f} → {verdict_dsr} multiple-testing correction")
    print(f"  Placebo p-value: {p_value:.4f} → signal {verdict_pl} random-entry null")
    print(f"  {'=' * 80}\n")

    summary = {
        "n_bars": n_bars,
        "sr_observed_annualized": sr_obs_ann,
        "sr_observed_perbar": sr_obs_perbar,
        "se_sr": se_sr,
        "var_sr": var_sr,
        "dsr": dsr_rows,
        "placebo": {
            "n_perms": args.n_perms,
            "sr_observed": sr_obs_ann,
            "null_mean": null_mean,
            "null_std": float(sr_perm.std(ddof=1)),
            "null_p05": float(np.quantile(sr_perm, 0.05)),
            "null_median": float(np.quantile(sr_perm, 0.50)),
            "null_p95": float(np.quantile(sr_perm, 0.95)),
            "null_p99": float(np.quantile(sr_perm, 0.99)),
            "null_max": float(sr_perm.max()),
            "p_value": p_value,
            "signal_sr": sig_sr,
            "sizing_floor_sr": null_mean,
            "pct_from_signal": pct_signal,
        },
        "signal_mix": {c: {"long": m[0], "short": m[1], "flat": m[2]}
                       for c, m in coin_sig_mix.items()},
    }
    with open(out_dir / "v5_validation.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    np.save(out_dir / "placebo_sr_null.npy", sr_perm)
    print(f"  Wrote: {out_dir / 'v5_validation.json'}")
    print(f"  Wrote: {out_dir / 'placebo_sr_null.npy'}")


if __name__ == "__main__":
    main()
