#!/usr/bin/env python
"""BT8 — Expanding-window walk-forward backtest of V3 quant stack.

Mirrors ``scripts/walkforward_v2.py`` protocol for direct comparability:
quarterly test blocks (63 bars) over 2021-11 → 2026-04, expanding-window
training, per-quarter aggregate stats + bootstrap CI on concatenated daily
returns.

V3-specific differences vs V2 BT8:
  - V3 runner is called with ``retrain_per_bar=True, retrain_cadence=63`` so
    the MultiHorizonEnsemble is refit at the start of every quarter on all
    available history through ``as_of - 21 days`` (purge guard for h=21).
  - NH-HMM regime bundle is loaded once from
    ``data/checkpoints/regime_hmm_v3_{coin}.pkl`` and held fixed (matches the
    CPCV protocol in ``evaluate_v3_cpcv.py`` — HMM fit on long history is not
    the subject of the WF evaluation).
  - V3 sizing layer = vol-target + CDAP (no SMA30 bolt-on by default).

Usage:
    python scripts/walkforward_v3.py \\
        --coins bitcoin ethereum \\
        --start 2021-11-01 --end 2026-04-15 \\
        --quarter-bars 63 \\
        --output-dir data/walkforward_v3_2coin
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v3.backtest.runner_v3 import run_v3_backtest  # noqa: E402
from tradingagents.strategies.v3.config import V3Config  # noqa: E402
from tradingagents.strategies.v3.features.extended import (  # noqa: E402
    build_extended_global_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _load_required_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required pickle missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ohlcv(coin: str, end: str) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coingecko_id=coin, curr_date=end)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def _quarter_blocks(dates: pd.DatetimeIndex, q_bars: int) -> list[tuple[int, int]]:
    n = len(dates)
    out = []
    for s in range(0, n, q_bars):
        e = min(s + q_bars, n)
        if e - s >= max(20, q_bars // 2):
            out.append((s, e))
    return out


def _bootstrap_ci(rets: np.ndarray, n_iter: int, block: int, rng: np.random.Generator) -> tuple[float, float, float, float]:
    """Returns (CI95 lo, CI95 hi, P(SR>0), P(SR>1)) via stationary bootstrap."""
    from scripts.bootstrap_sharpe import sharpe, stationary_bootstrap_sample  # type: ignore
    samples = np.empty(n_iter)
    for k in range(n_iter):
        samples[k] = sharpe(stationary_bootstrap_sample(rets, block, rng))
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    p_gt_0 = float((samples > 0).mean())
    p_gt_1 = float((samples > 1).mean())
    return lo, hi, p_gt_0, p_gt_1


def _per_quarter_stats(
    block_dates: np.ndarray, block_rets: np.ndarray, initial_capital: float
) -> dict:
    if len(block_rets) == 0:
        return {"sharpe": float("nan"), "total_return": float("nan"),
                "max_dd": float("nan"), "win_rate": float("nan"), "n_trades": 0}
    eq = initial_capital * np.cumprod(1.0 + block_rets)
    total_return = float(eq[-1] / initial_capital - 1.0)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = float(dd.min())
    mu = float(np.mean(block_rets))
    sigma = float(np.std(block_rets, ddof=1)) if len(block_rets) > 1 else 0.0
    sharpe = (mu / sigma) * np.sqrt(252) if sigma > 0 else 0.0
    win_rate = float(np.mean(block_rets > 0))
    n_trades = int(np.sum(np.abs(np.diff(np.sign(block_rets), prepend=0)) > 0))
    return {
        "sharpe": float(sharpe),
        "total_return": total_return,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "n_trades": n_trades,
    }


def _build_extended_for_coin(
    coin: str, ohlcv: pd.DataFrame, prices: pd.Series,
    micro: pd.DataFrame, deriv: pd.DataFrame,
    btc_ohlcv: pd.DataFrame | None, eth_ohlcv: pd.DataFrame | None,
) -> pd.DataFrame:
    """Wrapper: assemble cross-asset inputs and build the extended feature frame."""
    btc_p = btc_ohlcv["Close"] if btc_ohlcv is not None else None
    eth_p = eth_ohlcv["Close"] if eth_ohlcv is not None else None
    btc_v = btc_ohlcv["Volume"] if btc_ohlcv is not None else None
    return build_extended_global_features(
        coin=coin,
        prices=prices,
        ohlcv=ohlcv.reset_index().rename(columns={"index": "Date"}),
        microstructure_features=micro,
        derivatives_features=deriv,
        btc_prices=btc_p,
        eth_prices=eth_p,
        btc_volume=btc_v,
        include_pit_onchain=True,
    )


def run_coin_walkforward(
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    q_bars: int,
    cfg: V3Config,
    micro_dir: Path,
    deriv_dir: Path,
    regime_dir: Path,
    models_dir: Path,
    initial_capital: float,
    feature_set: str = "base",
    btc_ohlcv: pd.DataFrame | None = None,
    eth_ohlcv: pd.DataFrame | None = None,
) -> tuple[list[dict], list[dict], pd.Series]:
    """Run V3 WF for one coin. Returns (per-quarter rows, daily-return rows, eq series)."""
    ohlcv = _load_ohlcv(coin, end.strftime("%Y-%m-%d"))
    prices = ohlcv["Close"]
    returns = prices.pct_change().fillna(0.0)

    micro = _load_optional_parquet(micro_dir / f"{coin}.parquet")
    deriv = _load_optional_parquet(deriv_dir / f"{coin}.parquet")
    regime_bundle = _load_required_pickle(regime_dir / f"regime_hmm_v3_{coin}.pkl")
    mh_bundle = _load_required_pickle(models_dir / f"v3_models_{coin}.pkl")

    logger.info("[%s] OHLCV %d bars %s → %s; running V3 BT8 WF %s → %s (q_bars=%d, feature_set=%s)",
                coin, len(prices), prices.index.min().date(), prices.index.max().date(),
                start.date(), end.date(), q_bars, feature_set)

    global_override = None
    if feature_set == "extended":
        global_override = _build_extended_for_coin(coin, ohlcv, prices, micro, deriv, btc_ohlcv, eth_ohlcv)
        logger.info("[%s] extended feature matrix shape=%s", coin, global_override.shape)

    t0 = time.time()
    result = run_v3_backtest(
        coin=coin,
        prices=prices,
        returns=returns,
        microstructure_features=micro,
        derivatives_features=deriv,
        regime_bundle=regime_bundle,
        multi_horizon_bundle=mh_bundle,
        config=cfg,
        start=start,
        end=end,
        ticker=coin.upper(),
        initial_capital=initial_capital,
        retrain_per_bar=True,
        retrain_cadence=q_bars,
        retrain_members=("lgb",),
        retrain_use_calibration=False,
        global_features_override=global_override,
    )
    elapsed = time.time() - t0
    logger.info("[%s] V3 WF done in %.1fs (%d daily returns)", coin, elapsed, len(result.daily_returns))

    bars = prices.loc[start:end].index
    daily_rets = np.asarray(result.daily_returns, dtype=float)
    if len(daily_rets) != len(bars):
        logger.warning("[%s] daily_returns len %d != bars len %d — trimming",
                       coin, len(daily_rets), len(bars))
        n = min(len(daily_rets), len(bars))
        daily_rets = daily_rets[:n]
        bars = bars[:n]

    blocks = _quarter_blocks(bars, q_bars)
    per_quarter_rows: list[dict] = []
    eq_curve = []
    daily_rows = []
    cum_eq = initial_capital
    for s, e in blocks:
        block_dates = bars[s:e].values
        block_rets = daily_rets[s:e]
        stats = _per_quarter_stats(block_dates, block_rets, initial_capital)
        ts0 = pd.Timestamp(block_dates[0])
        ts1 = pd.Timestamp(block_dates[-1])
        q_label = f"{ts0.strftime('%Y-%m-%d')}_{ts1.strftime('%Y-%m-%d')}"
        per_quarter_rows.append({
            "coin": coin,
            "quarter": q_label,
            "n_bars": int(e - s),
            **stats,
        })
        for i, r in enumerate(block_rets):
            daily_rows.append({
                "date": pd.Timestamp(block_dates[i]).strftime("%Y-%m-%d"),
                "coin": coin,
                "ret": float(r),
            })
        # Per-quarter restart at initial_capital matches BT8 V2 (each quarter is independent)
        # Equity curve here is per-quarter compounded by total_return for plot purposes only.
        cum_eq *= (1.0 + stats["total_return"])
        eq_curve.append({"quarter_end": ts1, "compounded_equity": cum_eq})

    eq_series = pd.Series(
        [e["compounded_equity"] for e in eq_curve],
        index=[e["quarter_end"] for e in eq_curve],
    )
    return per_quarter_rows, daily_rows, eq_series


def aggregate_summary(per_quarter_df: pd.DataFrame, daily_df: pd.DataFrame, coin: str,
                      bs_n_iter: int = 3000, bs_block: int = 5) -> dict:
    sub = per_quarter_df[per_quarter_df["coin"] == coin]
    coin_daily = daily_df[daily_df["coin"] == coin]["ret"].values
    if len(coin_daily) <= 1 or sub.empty:
        return {"coin": coin, "skipped": "insufficient_data"}
    from scripts.bootstrap_sharpe import sharpe  # type: ignore
    rng = np.random.default_rng(42)
    sr_oos = float(sharpe(coin_daily))
    lo, hi, p_gt_0, p_gt_1 = _bootstrap_ci(coin_daily, bs_n_iter, bs_block, rng)
    sr_arr = sub["sharpe"].dropna().values
    ret_arr = sub["total_return"].dropna().values
    return {
        "n_quarters": int(len(sub)),
        "n_daily_bars": int(len(coin_daily)),
        "sr_oos_aggregated": sr_oos,
        "sr_oos_ci95": [lo, hi],
        "p_sr_gt_0_boot": p_gt_0,
        "p_sr_gt_1_boot": p_gt_1,
        "sr_quarter_mean": float(np.mean(sr_arr)) if len(sr_arr) else float("nan"),
        "sr_quarter_median": float(np.median(sr_arr)) if len(sr_arr) else float("nan"),
        "sr_quarter_std": float(np.std(sr_arr, ddof=1)) if len(sr_arr) > 1 else float("nan"),
        "sr_quarter_p25": float(np.quantile(sr_arr, 0.25)) if len(sr_arr) else float("nan"),
        "sr_quarter_p75": float(np.quantile(sr_arr, 0.75)) if len(sr_arr) else float("nan"),
        "frac_sr_gt_0": float((sr_arr > 0).mean()) if len(sr_arr) else float("nan"),
        "frac_sr_gt_1": float((sr_arr > 1).mean()) if len(sr_arr) else float("nan"),
        "frac_sr_gt_2": float((sr_arr > 2).mean()) if len(sr_arr) else float("nan"),
        "geo_total_return": float(np.prod(1 + ret_arr) - 1) if len(ret_arr) else float("nan"),
        "max_quarter_dd": float(sub["max_dd"].min()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    p.add_argument("--start", default="2021-11-01")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--quarter-bars", type=int, default=63)
    p.add_argument("--microstructure-dir", default="data/microstructure")
    p.add_argument("--derivatives-dir", default="data/derivatives")
    p.add_argument("--regime-dir", default="data/checkpoints")
    p.add_argument("--models-dir", default="data/checkpoints")
    p.add_argument("--output-dir", default="data/walkforward_v3_2coin")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--feature-set", choices=("base", "extended"), default="base",
                   help="base = 9 V3 features (default); extended = ~180 features (V2 OHLC + TI + cross-asset + lags + calendar + V3 micro/deriv + PIT on-chain)")
    args = p.parse_args()

    cfg = V3Config()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    all_quarter_rows: list[dict] = []
    all_daily_rows: list[dict] = []
    coin_eq_series: dict[str, pd.Series] = {}

    # Pre-load BTC + ETH OHLCV for cross-asset features (extended feature set)
    btc_ohlcv = eth_ohlcv = None
    if args.feature_set == "extended":
        btc_ohlcv = _load_ohlcv("bitcoin", args.end)
        eth_ohlcv = _load_ohlcv("ethereum", args.end)

    for coin in args.coins:
        try:
            per_q, daily, eq = run_coin_walkforward(
                coin=coin, start=start, end=end, q_bars=args.quarter_bars, cfg=cfg,
                micro_dir=Path(args.microstructure_dir),
                deriv_dir=Path(args.derivatives_dir),
                regime_dir=Path(args.regime_dir),
                models_dir=Path(args.models_dir),
                initial_capital=args.initial_capital,
                feature_set=args.feature_set,
                btc_ohlcv=btc_ohlcv,
                eth_ohlcv=eth_ohlcv,
            )
        except Exception as exc:
            logger.exception("[%s] WF failed: %s", coin, exc)
            continue
        all_quarter_rows.extend(per_q)
        all_daily_rows.extend(daily)
        coin_eq_series[coin] = eq

    if not all_quarter_rows:
        logger.error("No coins produced results — aborting")
        return

    per_quarter_df = pd.DataFrame(all_quarter_rows)
    daily_df = pd.DataFrame(all_daily_rows)
    per_quarter_df.to_csv(out_dir / "quarterly_metrics.csv", index=False)
    daily_df.to_csv(out_dir / "daily_returns.csv", index=False)

    summary = {}
    print(f"\n{'=' * 86}")
    print(f"  V3 Walk-forward ({args.start} -> {args.end})  q={args.quarter_bars} bars")
    print(f"{'=' * 86}\n")
    for coin in args.coins:
        agg = aggregate_summary(per_quarter_df, daily_df, coin)
        if "skipped" in agg:
            print(f"  {coin}: SKIPPED ({agg['skipped']})")
            continue
        summary[coin] = agg
        print(f"  {coin}  ({agg['n_quarters']} quarters, {agg['n_daily_bars']} daily bars)")
        print(f"    OOS Sharpe (aggregated): {agg['sr_oos_aggregated']:+.2f}  "
              f"CI95=[{agg['sr_oos_ci95'][0]:+.2f},{agg['sr_oos_ci95'][1]:+.2f}]  "
              f"P(SR>0)={agg['p_sr_gt_0_boot']:.3f}  P(SR>1)={agg['p_sr_gt_1_boot']:.3f}")
        print(f"    Quarter SR  mean={agg['sr_quarter_mean']:+.2f}  median={agg['sr_quarter_median']:+.2f}  "
              f"std={agg['sr_quarter_std']:.2f}  IQR=[{agg['sr_quarter_p25']:+.2f},{agg['sr_quarter_p75']:+.2f}]")
        print(f"    Frac SR>0: {agg['frac_sr_gt_0']:.0%}  >1: {agg['frac_sr_gt_1']:.0%}  >2: {agg['frac_sr_gt_2']:.0%}")
        print(f"    Compounded return: {agg['geo_total_return']:+.1%}  Max quarter DD: {agg['max_quarter_dd']:.1%}")
        print(f"    {'quarter':<22} {'SR':>7}  {'ret':>8}  {'MaxDD':>7}  {'win':>5}  {'#tr':>4}")
        sub = per_quarter_df[per_quarter_df["coin"] == coin]
        for _, r in sub.iterrows():
            print(f"    {r['quarter']:<22} {r['sharpe']:>+7.2f}  {r['total_return']:>+7.2%}  "
                  f"{r['max_dd']:>6.2%}  {r['win_rate']:>4.0%}   {r['n_trades']:>4d}")
        print()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(14, 6))
    for coin, eq in coin_eq_series.items():
        ax.plot(eq.index, eq.values, marker="o", label=coin, linewidth=1.6)
    ax.set_xlabel("Quarter end")
    ax.set_ylabel("Compounded equity (10k initial per coin)")
    ax.set_title("V3 BT8 Walk-forward — quarterly compounded equity")
    ax.legend(); ax.grid(True, alpha=0.3); fig.autofmt_xdate()
    fig.savefig(out_dir / "walkforward_equity.png", dpi=130, bbox_inches="tight")

    print(f"  Wrote: {out_dir / 'quarterly_metrics.csv'}")
    print(f"  Wrote: {out_dir / 'summary.json'}")
    print(f"  Plot:  {out_dir / 'walkforward_equity.png'}")


if __name__ == "__main__":
    main()
