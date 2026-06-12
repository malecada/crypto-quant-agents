#!/usr/bin/env python
"""V4-B analysis — feature importance + per-regime decomposition.

Part 1 (feature importance): fits one LGB regressor per coin on the 193-feature
extended pool (PIT on-chain + Coinglass + technical + cross-asset) for h=14,
extracts gain importance, and contrasts BTC vs ETH feature rankings. Goal: see
WHY extended features help ETH (§17 V4-B SR +0.88 → +1.80) but hurt BTC
(+1.57 → +1.19) — which columns each coin's LGB actually relies on.

Part 2 (per-regime decomposition): splits V5 MIX daily returns (BTC=V2-78f,
ETH=V4-B-193f) by heuristic regime label (bull/sideways/bear) and reports
per-regime Sharpe, mean return, and bar count. Goal: test whether the V4-B
ETH alpha is regime-robust or concentrated.

Usage:
    python scripts/analyze_v4b.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.models import lgb_model  # noqa: E402
from tradingagents.models.model_utils import build_pooled_dataset, data_transform  # noqa: E402
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v3.regime.hmm_v2 import heuristic_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HORIZON = 14
TRADE_DATE = "2026-04-15"
LOOKBACK_DAYS = 2200


def feature_importance_analysis() -> pd.DataFrame:
    """Fit per-coin LGB on the 193-feature pool, return importance comparison."""
    logger.info("Building 193-feature pooled dataset (add_onchain_pit=True)...")
    pooled_raw = build_pooled_dataset(
        coin_universe=["bitcoin", "ethereum"],
        lookback_days=LOOKBACK_DAYS,
        horizons=[7, HORIZON],
        trade_date=TRADE_DATE,
        add_technical=True,
        add_cross_asset=True,
        add_onchain=True,
        add_onchain_pit=True,
    )
    logger.info("Pooled raw shape: %s", pooled_raw.shape)

    per_coin_imp: dict[str, pd.Series] = {}
    for coin in ("bitcoin", "ethereum"):
        sub = pooled_raw[pooled_raw["coin_id"] == coin].drop(columns=["coin_id"])
        first_future = sub.index.max() + pd.Timedelta(days=1)
        ref, _ = data_transform(sub, first_day_future=first_future,
                                include_future_row=True, horizons=[7, HORIZON])
        ref["coin_id"] = coin
        ref = ref.set_index("date") if "date" in ref.columns else ref
        bundle = lgb_model.fit_pooled_full(ref, horizon=HORIZON)
        booster = bundle["booster"]
        names = bundle["feature_names"]
        # gain importance
        imp = pd.Series(booster.feature_importances_, index=names, name=coin)
        per_coin_imp[coin] = imp.sort_values(ascending=False)
        logger.info("[%s] LGB fit on %d rows, %d features", coin,
                    bundle["n_train_rows"], len(names))

    btc_imp = per_coin_imp["bitcoin"]
    eth_imp = per_coin_imp["ethereum"]
    # Normalize to fractions (gain importance sums vary)
    btc_frac = btc_imp / btc_imp.sum()
    eth_frac = eth_imp / eth_imp.sum()
    comp = pd.DataFrame({"btc_frac": btc_frac, "eth_frac": eth_frac}).fillna(0.0)
    comp["diff_eth_minus_btc"] = comp["eth_frac"] - comp["btc_frac"]

    def _group(col: str) -> str:
        if col.startswith("oc_"):
            return "PIT-onchain/coinglass"
        if col.startswith("ti_"):
            return "technical-indicator"
        if col.startswith("xa_"):
            return "cross-asset"
        if col.startswith("deriv_"):
            return "derivatives-raw"
        if col.startswith("lag"):
            return "price-lag"
        if col.startswith("day_") or col in ("Day", "Month", "Year"):
            return "calendar"
        if col in ("prices", "open", "high", "low", "total_volumes", "daily_return",
                   "high_low_spread", "open_close_spread") or col.startswith(("ma_", "vol_")):
            return "ohlc-mechanics"
        if col == "coin_int":
            return "coin-id"
        return "other"

    comp["group"] = [_group(c) for c in comp.index]

    print("\n" + "=" * 86)
    print("  PART 1 — FEATURE IMPORTANCE (per-coin LGB, h=14, gain)")
    print("=" * 86)

    print("\n  Importance mass by feature group:")
    grp = comp.groupby("group")[["btc_frac", "eth_frac"]].sum().sort_values("eth_frac", ascending=False)
    print(f"    {'group':<26} {'BTC%':>8} {'ETH%':>8} {'ETH-BTC':>9}")
    for g, row in grp.iterrows():
        print(f"    {g:<26} {row['btc_frac']*100:>7.1f}% {row['eth_frac']*100:>7.1f}% "
              f"{(row['eth_frac']-row['btc_frac'])*100:>+8.1f}%")

    print("\n  Top 20 features — BITCOIN:")
    for c, v in btc_frac.head(20).items():
        print(f"    {c:<42} {v*100:>6.2f}%   [{_group(c)}]")

    print("\n  Top 20 features — ETHEREUM:")
    for c, v in eth_frac.head(20).items():
        print(f"    {c:<42} {v*100:>6.2f}%   [{_group(c)}]")

    print("\n  Features ETH relies on MUCH more than BTC (top 15 by diff):")
    for c, row in comp.sort_values("diff_eth_minus_btc", ascending=False).head(15).iterrows():
        print(f"    {c:<42} ETH={row['eth_frac']*100:>5.2f}%  BTC={row['btc_frac']*100:>5.2f}%  "
              f"Δ={row['diff_eth_minus_btc']*100:>+6.2f}%   [{row['group']}]")

    print("\n  Features BTC relies on MUCH more than ETH (top 15 by diff):")
    for c, row in comp.sort_values("diff_eth_minus_btc").head(15).iterrows():
        print(f"    {c:<42} BTC={row['btc_frac']*100:>5.2f}%  ETH={row['eth_frac']*100:>5.2f}%  "
              f"Δ={-row['diff_eth_minus_btc']*100:>+6.2f}%   [{row['group']}]")

    out = PROJECT_ROOT / "data" / "v4b_analysis"
    out.mkdir(parents=True, exist_ok=True)
    comp.sort_values("eth_frac", ascending=False).to_csv(out / "feature_importance.csv")
    print(f"\n  Wrote: {out / 'feature_importance.csv'}")
    return comp


def per_regime_decomposition() -> pd.DataFrame:
    """Split V5 MIX daily returns by heuristic regime label per coin."""
    print("\n" + "=" * 86)
    print("  PART 2 — PER-REGIME DECOMPOSITION (V5 MIX daily returns)")
    print("=" * 86)

    sources = {
        "bitcoin": "data/walkforward_v4_v2repro/daily_returns.csv",   # V2-78f
        "ethereum": "data/walkforward_v4b_pit_noregime/daily_returns.csv",  # V4-B-193f
    }

    rows = []
    coin_daily: dict[str, pd.Series] = {}
    for coin, path in sources.items():
        dr = pd.read_csv(PROJECT_ROOT / path, parse_dates=["date"])
        dr = dr[dr["coin"] == coin].set_index("date")["ret"].sort_index()
        coin_daily[coin] = dr

        # Load full price history and label each date's regime.
        ohlcv = _load_crypto_ohlcv(coin, TRADE_DATE)
        ohlcv["Date"] = pd.to_datetime(ohlcv["Date"])
        prices = ohlcv.set_index("Date").sort_index()["Close"]
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        labels = {}
        for d in dr.index:
            sub = prices[prices.index <= d]
            if len(sub) < 30:
                labels[d] = "sideways"
            else:
                lbl, _conf, _h = heuristic_label(sub)
                labels[d] = lbl
        label_series = pd.Series(labels)

        for regime in ("bull", "sideways", "bear"):
            mask = label_series == regime
            r = dr[mask.reindex(dr.index, fill_value=False)]
            if len(r) == 0:
                continue
            sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
            rows.append({
                "coin": coin, "regime": regime, "n_bars": int(len(r)),
                "pct_bars": len(r) / len(dr),
                "mean_daily_ret": float(r.mean()),
                "sharpe": sr,
                "total_ret": float((1 + r).prod() - 1),
            })

    df = pd.DataFrame(rows)
    print()
    for coin in ("bitcoin", "ethereum"):
        sub = df[df["coin"] == coin]
        full = coin_daily[coin]
        full_sr = float(full.mean() / full.std() * np.sqrt(252))
        print(f"  {coin}  (full-window SR={full_sr:+.2f}, {len(full)} bars)")
        print(f"    {'regime':<10} {'n_bars':>7} {'%bars':>7} {'mean_ret':>10} {'sharpe':>8} {'total_ret':>10}")
        for _, row in sub.iterrows():
            print(f"    {row['regime']:<10} {row['n_bars']:>7d} {row['pct_bars']*100:>6.1f}% "
                  f"{row['mean_daily_ret']*100:>9.3f}% {row['sharpe']:>+8.2f} {row['total_ret']*100:>+9.1f}%")
        print()

    out = PROJECT_ROOT / "data" / "v4b_analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_regime_decomposition.csv", index=False)
    print(f"  Wrote: {out / 'per_regime_decomposition.csv'}")
    return df


def main() -> None:
    feature_importance_analysis()
    per_regime_decomposition()


if __name__ == "__main__":
    main()
