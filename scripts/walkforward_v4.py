#!/usr/bin/env python
"""V4 — V2 signal/sizing core + NH-HMM regime overlay.

Hypothesis: V3's NH-HMM regime detector may add alpha when bolted onto V2's
production sizing/signal layer rather than replacing them. V3 in isolation
loses to V2 by ΔSR -3.55 / -1.58 even with full feature parity (§15). This
script isolates the regime contribution by keeping V2 100% intact and applying
a regime-conditional position multiplier as the only V3-derived addition.

Multiplier (sign-aware, matches V2 5-level discrete intent):

    | regime    | long pos       | short pos      |
    | bull      | 1.20 × conf    | 0.40 × conf    |
    | sideways  | 0.70 × conf    | 0.70 × conf    |
    | bear      | 0.40 × conf    | 1.20 × conf    |

    × 0.5 when BOCPD changepoint_alert fires.
    confidence ∈ [0.5, 1.0] — base 0.5, max 1.0 from posterior dominance.

Usage:
    python scripts/walkforward_v4.py \\
        --pred-dir data/multi_2coins_walkforward \\
        --coins bitcoin ethereum \\
        --start 2021-11-07 --end 2026-04-15 \\
        --quarter-bars 63 \\
        --output-dir data/walkforward_v4_2coin
"""

from __future__ import annotations

import argparse
import json
import logging
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

from scripts.baseline_strategy_v2 import run_coin_backtest  # type: ignore  # noqa: E402
from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v2_sizing import (  # noqa: E402
    apply_trend_filter, build_positions_with_hold, compute_realized_vol,
    generate_term_structure_signals, vol_regime_mask,
)
from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3  # noqa: E402
from tradingagents.strategies.v3.regime.hmm_v2 import heuristic_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


COSTS = dict(
    fee_rate=0.0004, slippage=0.0005, spread=0.0001,
    price_impact=0.00005, funding_rate=0.0001 / 8,
    stop_loss=0.03, max_portfolio_dd=0.15,
)

# Regime multipliers per (regime_label, position_sign).
_REGIME_MULT = {
    "bull":     {"long": 1.20, "short": 0.40},
    "sideways": {"long": 0.70, "short": 0.70},
    "bear":     {"long": 0.40, "short": 1.20},
}
_CONF_FLOOR = 0.5
_CHANGEPOINT_DAMP = 0.5


def _load_preds(pred_dir: Path, coin: str) -> pd.DataFrame:
    p7 = pd.read_csv(pred_dir / "preds_lgb_h7.csv", parse_dates=["date"])
    p14 = pd.read_csv(pred_dir / "preds_lgb_h14.csv", parse_dates=["date"])
    for df in (p7, p14):
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    p7 = p7[p7["coin_id"] == coin].rename(columns={"prediction": "pred_h7"})
    p14 = p14[p14["coin_id"] == coin].rename(columns={"prediction": "pred_h14"})[["date", "pred_h14"]]
    return p7.merge(p14, on="date").sort_values("date").reset_index(drop=True)


def _load_prices(coin: str, end: str) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coin, end)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df.sort_values("Date").reset_index(drop=True)


def _v2_positions(merged: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sig, conf = generate_term_structure_signals(merged, [7, 14], 0.05, asymmetric=True)
    px = merged["Close"].astype(float).values
    rv = compute_realized_vol(px, lookback=20)
    mask = vol_regime_mask(rv, percentile_cap=0.95)
    pos = build_positions_with_hold(
        signals=sig, vol_ok=mask, confidence=conf, realized_vol=rv, prices=px,
        target_vol=0.10, kelly_fraction=0.5, max_leverage=3.0,
        min_hold=7, early_exit_loss=0.015,
    )
    pos = apply_trend_filter(pos, px, sma_period=30, multiplier=1.5)
    return pos, px


def _compute_regime_overlay(
    prices_series: pd.Series, dates: np.ndarray, positions: np.ndarray,
    bundle=None, method: str = "nh_hmm",
) -> tuple[np.ndarray, list[dict]]:
    """Compute per-bar regime multiplier vector. Returns (multipliers, diagnostics).

    ``method`` selects regime classifier:
      - ``nh_hmm``: detect_regime_v3 (requires bundle). Existing V3 path.
      - ``heuristic``: heuristic_label — deterministic 30d-return + Hurst classifier.
        Faster, no stale-bundle risk; recommended after BTC bundle pathology found.
    """
    mults = np.ones(len(dates), dtype=float)
    diag: list[dict] = []
    for i, (d, pos) in enumerate(zip(dates, positions)):
        as_of = pd.Timestamp(d)
        try:
            if method == "nh_hmm":
                if bundle is None:
                    raise ValueError("nh_hmm method requires bundle")
                regime = detect_regime_v3(prices=prices_series, bundle=bundle, as_of=as_of)
                label, conf, changepoint = regime.label, regime.confidence, regime.changepoint_alert
            elif method == "heuristic":
                sub = prices_series[prices_series.index <= as_of]
                if len(sub) < 30:
                    label, conf, _h = "sideways", 0.3, 0.5
                else:
                    label, conf, _h = heuristic_label(sub)
                changepoint = False
            else:
                raise ValueError(f"unknown method {method}")
        except Exception as exc:
            logger.debug("regime detect failed at %s: %s", as_of, exc)
            diag.append({"date": str(as_of.date()), "label": "FAILED", "mult": 1.0})
            continue

        sign = "long" if pos > 0 else "short" if pos < 0 else "flat"
        if sign == "flat":
            mults[i] = 1.0
            diag.append({"date": str(as_of.date()), "label": label,
                         "conf": float(conf), "sign": "flat", "mult": 1.0})
            continue

        base = _REGIME_MULT.get(label, {"long": 0.7, "short": 0.7})[sign]
        conf_factor = max(_CONF_FLOOR, min(1.0, conf))
        m = base * conf_factor
        if changepoint:
            m *= _CHANGEPOINT_DAMP
        mults[i] = m
        diag.append({
            "date": str(as_of.date()), "label": label, "conf": float(conf),
            "sign": sign, "base": base, "changepoint": bool(changepoint), "mult": float(m),
        })
    return mults, diag


def _quarter_blocks(dates: pd.DatetimeIndex, q_bars: int) -> list[tuple[int, int]]:
    n = len(dates)
    out = []
    for s in range(0, n, q_bars):
        e = min(s + q_bars, n)
        if e - s >= max(20, q_bars // 2):
            out.append((s, e))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", default="data/multi_2coins_walkforward")
    p.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    p.add_argument("--start", default="2021-11-07")
    p.add_argument("--end", default="2026-04-15")
    p.add_argument("--quarter-bars", type=int, default=63)
    p.add_argument("--regime-dir", default="data/checkpoints")
    p.add_argument("--regime-method", choices=("nh_hmm", "heuristic"), default="heuristic",
                   help="Regime classifier: nh_hmm (stale bundle, known pathology on BTC) or heuristic (30d return + Hurst, deterministic)")
    p.add_argument("--output-dir", default="data/walkforward_v4_2coin")
    p.add_argument("--no-regime", action="store_true", help="Disable regime overlay (= V2 baseline reproduction)")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    regime_dir = Path(args.regime_dir)

    all_rows = []
    coin_dates_full: dict[str, list] = {}
    daily_returns_rows: list[dict] = []
    regime_diag_rows: list[dict] = []

    for coin in args.coins:
        preds = _load_preds(Path(args.pred_dir), coin)
        preds = preds[(preds["date"] >= args.start) & (preds["date"] <= args.end)]
        if preds.empty:
            logger.warning("[%s] no preds in window", coin)
            continue

        prices = _load_prices(coin, args.end)
        merged = preds.merge(prices[["Date", "Close"]], left_on="date", right_on="Date")
        merged = merged.dropna(subset=["Close"]).reset_index(drop=True)
        merged = merged.rename(columns={"date": "_date"})
        merged["ref_price"] = merged["Close"]

        pos_full, px_full = _v2_positions(merged)
        dates_full = merged["_date"].values
        coin_dates_full[coin] = list(dates_full)

        # Apply regime overlay
        if not args.no_regime:
            bundle = None
            if args.regime_method == "nh_hmm":
                bundle_path = regime_dir / f"regime_hmm_v3_{coin}.pkl"
                if not bundle_path.exists():
                    logger.warning("[%s] regime bundle missing %s — skipping overlay", coin, bundle_path)
                    continue
                with open(bundle_path, "rb") as f:
                    bundle = pickle.load(f)
            prices_series = pd.Series(px_full, index=pd.to_datetime(dates_full))
            t0 = time.time()
            mults, diag = _compute_regime_overlay(
                prices_series, dates_full, pos_full, bundle=bundle, method=args.regime_method,
            )
            logger.info("[%s] regime overlay (%s): %.1fs over %d bars; "
                        "mean_mult=%.3f, mult_p10=%.3f p90=%.3f",
                        coin, args.regime_method, time.time() - t0, len(mults),
                        float(np.mean(mults)), float(np.quantile(mults, 0.1)), float(np.quantile(mults, 0.9)))
            for r in diag:
                r["coin"] = coin
                regime_diag_rows.append(r)
            pos_full = pos_full * mults

        blocks = _quarter_blocks(merged["_date"], args.quarter_bars)
        for s, e in blocks:
            block_dates = dates_full[s:e]
            block_px = px_full[s:e]
            block_pos = pos_full[s:e]
            equity, m = run_coin_backtest(
                dates=block_dates, prices=block_px, positions=block_pos,
                initial_capital=10_000.0, **COSTS,
            )
            eq_arr = np.asarray(equity, dtype=float)
            if len(eq_arr) > 1:
                rets = eq_arr[1:] / eq_arr[:-1] - 1.0
                for i, r in enumerate(rets):
                    daily_returns_rows.append({
                        "date": pd.Timestamp(block_dates[i + 1]).strftime("%Y-%m-%d"),
                        "coin": coin, "ret": float(r),
                    })
            ts0 = pd.Timestamp(block_dates[0]); ts1 = pd.Timestamp(block_dates[-1])
            q_lbl = f"{ts0.strftime('%Y-%m-%d')}_{ts1.strftime('%Y-%m-%d')}"
            all_rows.append({
                "coin": coin, "quarter": q_lbl, "n_bars": e - s,
                "sharpe": float(m.get("sharpe_ratio", float("nan"))),
                "total_return": float(m.get("total_return", float("nan"))),
                "max_dd": float(m.get("max_drawdown", float("nan"))),
                "win_rate": float(m.get("win_rate", float("nan"))),
                "n_trades": int(m.get("n_trades", 0)),
            })

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "quarterly_metrics.csv", index=False)
    pd.DataFrame(daily_returns_rows).to_csv(out_dir / "daily_returns.csv", index=False)
    if regime_diag_rows:
        pd.DataFrame(regime_diag_rows).to_csv(out_dir / "regime_diagnostics.csv", index=False)

    daily_df = pd.DataFrame(daily_returns_rows)
    from scripts.bootstrap_sharpe import sharpe, stationary_bootstrap_sample  # type: ignore
    rng = np.random.default_rng(42)
    DAILY_RF = (1 + 0.045) ** (1 / 252) - 1
    bs_n_iter = 3000
    bs_block = 5

    label = "V2-reproduction" if args.no_regime else "V4 (V2 + NH-HMM regime overlay)"
    print(f"\n{'=' * 86}")
    print(f"  {label}  ({args.start} → {args.end})  q={args.quarter_bars} bars")
    print(f"{'=' * 86}\n")
    summary = {}
    for coin in args.coins:
        sub = df[df["coin"] == coin]
        if sub.empty:
            continue
        sr_arr = sub["sharpe"].dropna().values
        ret_arr = sub["total_return"].dropna().values
        coin_daily = daily_df[daily_df["coin"] == coin]["ret"].values
        if len(coin_daily) <= 1:
            continue
        sr_oos = float(sharpe(coin_daily))
        bs_samples = np.empty(bs_n_iter)
        for k in range(bs_n_iter):
            bs_samples[k] = sharpe(stationary_bootstrap_sample(coin_daily, bs_block, rng))
        lo = float(np.quantile(bs_samples, 0.025))
        hi = float(np.quantile(bs_samples, 0.975))
        agg = {
            "n_quarters": int(len(sub)),
            "n_daily_bars": int(len(coin_daily)),
            "sr_oos_aggregated": sr_oos,
            "sr_oos_ci95": [lo, hi],
            "p_sr_gt_0_boot": float((bs_samples > 0).mean()),
            "p_sr_gt_1_boot": float((bs_samples > 1).mean()),
            "sr_quarter_mean": float(np.mean(sr_arr)) if len(sr_arr) else float("nan"),
            "sr_quarter_median": float(np.median(sr_arr)) if len(sr_arr) else float("nan"),
            "frac_sr_gt_0": float((sr_arr > 0).mean()) if len(sr_arr) else float("nan"),
            "frac_sr_gt_1": float((sr_arr > 1).mean()) if len(sr_arr) else float("nan"),
            "geo_total_return": float(np.prod(1 + ret_arr) - 1) if len(ret_arr) else float("nan"),
            "max_quarter_dd": float(sub["max_dd"].max()),
        }
        summary[coin] = agg
        print(f"  {coin}  ({agg['n_quarters']} quarters, {agg['n_daily_bars']} daily bars)")
        print(f"    OOS Sharpe (agg): {agg['sr_oos_aggregated']:+.2f}  "
              f"CI95=[{agg['sr_oos_ci95'][0]:+.2f},{agg['sr_oos_ci95'][1]:+.2f}]  "
              f"P(SR>0)={agg['p_sr_gt_0_boot']:.3f}  P(SR>1)={agg['p_sr_gt_1_boot']:.3f}")
        print(f"    Quarter SR mean={agg['sr_quarter_mean']:+.2f}  median={agg['sr_quarter_median']:+.2f}  "
              f"frac>0={agg['frac_sr_gt_0']:.0%}  frac>1={agg['frac_sr_gt_1']:.0%}")
        print(f"    Compounded return: {agg['geo_total_return']:+.1%}  Max quarter DD: {agg['max_quarter_dd']:.1%}")
        for _, r in sub.iterrows():
            print(f"    {r['quarter']:<22} SR={r['sharpe']:>+6.2f}  ret={r['total_return']:>+7.2%}  "
                  f"MaxDD={r['max_dd']:>6.2%}  win={r['win_rate']:>4.0%}   #tr={r['n_trades']:>4d}")
        print()

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Wrote: {out_dir / 'quarterly_metrics.csv'}")
    print(f"  Wrote: {out_dir / 'summary.json'}")
    print(f"  Wrote: {out_dir / 'daily_returns.csv'}")
    if regime_diag_rows:
        print(f"  Wrote: {out_dir / 'regime_diagnostics.csv'}")


if __name__ == "__main__":
    main()
