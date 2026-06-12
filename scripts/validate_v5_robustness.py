#!/usr/bin/env python
"""V5 MIX robustness battery — regime decomposition + CPCV + cost sensitivity.

Three robustness checks on the 4-coin V5 MIX (§20-21):

  A. Regime decomposition — per-coin daily returns split by each coin's own
     heuristic regime label; portfolio returns split by BTC regime (market
     proxy). Tests for regime concentration.

  B. CPCV — strategy-layer combinatorial purged cross-validation on the
     portfolio daily-return series (n_groups=8, test_groups=2, embargo=14 →
     28 test combinations). Tests SR stability across non-contiguous
     subwindows; reports PBO-proxy (fraction of test folds with SR < 0).

  C. Cost sensitivity — re-runs every per-coin backtest at 1×/2×/3× the
     baseline transaction-cost assumptions and rebuilds the portfolio. Tests
     whether the +3.2 SR survives pessimistic execution.

Usage:
    python scripts/validate_v5_robustness.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.baseline_strategy_v2 import run_coin_backtest  # type: ignore  # noqa: E402
from scripts.baseline_v5_mix import (  # noqa: E402
    DEFAULT_ROUTING, _load_preds, _v2_positions, _metrics, ANN,
)
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v3.backtest.cpcv import cpcv_splits  # noqa: E402
from tradingagents.strategies.v3.regime.hmm_v2 import heuristic_label  # noqa: E402

START, END = "2021-11-07", "2026-04-15"

_BASE_COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)
# stop_loss + max_portfolio_dd are risk limits, not costs — never scale them.
_COST_KEYS = ("fee_rate", "slippage", "spread", "price_impact", "funding_rate")


def _scaled_costs(mult: float) -> dict:
    c = dict(_BASE_COSTS)
    for k in _COST_KEYS:
        c[k] = _BASE_COSTS[k] * mult
    return c


def _coin_merged(coin: str) -> pd.DataFrame:
    preds = _load_preds(PROJECT_ROOT / DEFAULT_ROUTING[coin], coin)
    preds = preds[(preds["date"] >= START) & (preds["date"] <= END)]
    ohlcv = _load_crypto_ohlcv(coin, END)
    ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.tz_localize(None).dt.normalize()
    merged = preds.merge(ohlcv[["Date", "Close"]], left_on="date", right_on="Date")
    merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
    merged["ref_price"] = merged["Close"]
    return merged


def _coin_returns(merged: pd.DataFrame, costs: dict) -> pd.Series:
    pos = _v2_positions(merged)
    equity, _m = run_coin_backtest(
        dates=merged["date"].values, prices=merged["Close"].values,
        positions=pos, initial_capital=10_000.0, **costs,
    )
    eq = np.asarray(equity, dtype=float)
    return pd.Series(eq[1:] / eq[:-1] - 1.0, index=pd.to_datetime(merged["date"].values[1:]))


def _sharpe(r: np.ndarray) -> float:
    s = r.std()
    return float(r.mean() / s * ANN) if s > 0 else 0.0


def main() -> None:
    out_dir = PROJECT_ROOT / "data" / "v5_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all per-coin merged frames + baseline-cost returns once.
    merged = {c: _coin_merged(c) for c in DEFAULT_ROUTING}
    base_rets = {c: _coin_returns(merged[c], _BASE_COSTS) for c in DEFAULT_ROUTING}
    df = pd.DataFrame(base_rets).dropna().sort_index()
    port = df.mean(axis=1)

    summary: dict = {}

    # ── A. Regime decomposition ───────────────────────────────────────
    print(f"\n{'=' * 84}")
    print(f"  A. REGIME DECOMPOSITION  ({len(port)} bars)")
    print(f"{'=' * 84}\n")

    # Label each coin's dates by its own regime; cache BTC labels for portfolio.
    coin_labels: dict[str, pd.Series] = {}
    for coin in DEFAULT_ROUTING:
        ohlcv = _load_crypto_ohlcv(coin, END)
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"]).dt.tz_localize(None).dt.normalize()
        prices = ohlcv.set_index("Date").sort_index()["Close"]
        labels = {}
        for d in base_rets[coin].index:
            sub = prices[prices.index <= d]
            labels[d] = "sideways" if len(sub) < 30 else heuristic_label(sub)[0]
        coin_labels[coin] = pd.Series(labels)

    regime_rows = []
    for coin in DEFAULT_ROUTING:
        r = base_rets[coin]
        lab = coin_labels[coin]
        print(f"  {coin}  (full SR={_sharpe(r.values):+.2f})")
        for regime in ("bull", "sideways", "bear"):
            mask = (lab == regime).reindex(r.index, fill_value=False)
            rr = r[mask]
            if len(rr) == 0:
                continue
            regime_rows.append({
                "scope": coin, "regime": regime, "n_bars": int(len(rr)),
                "pct_bars": len(rr) / len(r), "sharpe": _sharpe(rr.values),
                "total_ret": float((1 + rr).prod() - 1),
            })
            print(f"    {regime:<9} n={len(rr):>4}  {len(rr)/len(r)*100:>5.1f}%  "
                  f"SR={_sharpe(rr.values):+.2f}  ret={float((1+rr).prod()-1)*100:+.1f}%")
        print()

    # Portfolio split by BTC regime (market-beta proxy)
    btc_lab = coin_labels["bitcoin"].reindex(port.index, method="ffill")
    print(f"  PORTFOLIO split by BTC regime  (full SR={_sharpe(port.values):+.2f})")
    for regime in ("bull", "sideways", "bear"):
        mask = btc_lab == regime
        rr = port[mask]
        if len(rr) == 0:
            continue
        regime_rows.append({
            "scope": "portfolio_by_btc_regime", "regime": regime,
            "n_bars": int(len(rr)), "pct_bars": len(rr) / len(port),
            "sharpe": _sharpe(rr.values), "total_ret": float((1 + rr).prod() - 1),
        })
        print(f"    {regime:<9} n={len(rr):>4}  {len(rr)/len(port)*100:>5.1f}%  "
              f"SR={_sharpe(rr.values):+.2f}  ret={float((1+rr).prod()-1)*100:+.1f}%")
    summary["regime_decomposition"] = regime_rows

    # ── B. CPCV on the portfolio return series ────────────────────────
    print(f"\n{'=' * 84}")
    print(f"  B. CPCV — strategy-layer combinatorial purged CV (portfolio returns)")
    print(f"{'=' * 84}\n")
    port_arr = port.values
    n = len(port_arr)
    splits = list(cpcv_splits(n_samples=n, n_groups=8, test_groups=2, embargo=14, min_train=252))
    fold_srs = []
    for sp in splits:
        test_r = port_arr[sp.test_idx]
        if len(test_r) > 1:
            fold_srs.append(_sharpe(test_r))
    fold_srs = np.array(fold_srs)
    pbo_proxy = float((fold_srs < 0).mean())
    print(f"  {len(fold_srs)} test folds (n_groups=8, test_groups=2, embargo=14)")
    print(f"  Fold SR:  mean={fold_srs.mean():+.2f}  median={np.median(fold_srs):+.2f}  "
          f"std={fold_srs.std(ddof=1):.2f}")
    print(f"            min={fold_srs.min():+.2f}  max={fold_srs.max():+.2f}  "
          f"p05={np.quantile(fold_srs,0.05):+.2f}  p95={np.quantile(fold_srs,0.95):+.2f}")
    print(f"  %folds SR>0: {(fold_srs>0).mean()*100:.0f}%   %folds SR>1: {(fold_srs>1).mean()*100:.0f}%   "
          f"%folds SR>2: {(fold_srs>2).mean()*100:.0f}%")
    print(f"  PBO proxy (fraction of folds with SR<0): {pbo_proxy:.3f}")
    summary["cpcv"] = {
        "n_folds": int(len(fold_srs)),
        "sr_mean": float(fold_srs.mean()), "sr_median": float(np.median(fold_srs)),
        "sr_std": float(fold_srs.std(ddof=1)),
        "sr_min": float(fold_srs.min()), "sr_max": float(fold_srs.max()),
        "frac_sr_gt_0": float((fold_srs > 0).mean()),
        "frac_sr_gt_1": float((fold_srs > 1).mean()),
        "frac_sr_gt_2": float((fold_srs > 2).mean()),
        "pbo_proxy": pbo_proxy,
    }

    # ── C. Cost sensitivity ───────────────────────────────────────────
    print(f"\n{'=' * 84}")
    print(f"  C. COST SENSITIVITY — portfolio SR at 1x / 2x / 3x baseline costs")
    print(f"{'=' * 84}\n")
    cost_rows = []
    for mult in (1.0, 2.0, 3.0):
        costs = _scaled_costs(mult)
        rets = {c: _coin_returns(merged[c], costs) for c in DEFAULT_ROUTING}
        cdf = pd.DataFrame(rets).dropna().sort_index()
        cport = cdf.mean(axis=1)
        m = _metrics(cport)
        cost_rows.append({"cost_mult": mult, **m})
        print(f"  {mult:.0f}x costs:  SR={m['sharpe']:+.3f}  ret={m['total_return']:+.1%}  "
              f"maxDD={m['max_drawdown']:.1%}  annVol={m['ann_vol']:.1%}")
    summary["cost_sensitivity"] = cost_rows

    base_sr = cost_rows[0]["sharpe"]
    sr_3x = cost_rows[2]["sharpe"]
    print(f"\n  SR degradation 1x→3x costs: {base_sr:+.3f} → {sr_3x:+.3f}  "
          f"({(sr_3x-base_sr)/base_sr*100:+.0f}%)")

    with open(out_dir / "v5_robustness.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Wrote: {out_dir / 'v5_robustness.json'}")


if __name__ == "__main__":
    main()
