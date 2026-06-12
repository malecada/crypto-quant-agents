#!/usr/bin/env python
"""Pooled multi-coin, multi-horizon walk-forward evaluation.

Builds a pooled dataset from a coin universe with optional technical,
cross-asset, and on-chain features, then runs walk-forward evaluation
with RF + LightGBM at user-specified horizons. ARIMA is run per-coin
for short horizons only (h <= 3).

Usage examples:
    # Full run: all 10 coins, h ∈ {1,3,7,14}, all models
    python scripts/evaluate_models_multi.py

    # Quick smoke test: 2 coins, 2 horizons, no on-chain
    python scripts/evaluate_models_multi.py \
        --coins bitcoin ethereum --horizons 1 7 --models rf lgb --no-onchain

    # Exclude cross-asset features to isolate their contribution
    python scripts/evaluate_models_multi.py --no-cross-asset
"""
from __future__ import annotations

import argparse
import logging
import sys
import traceback
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

DEFAULT_UNIVERSE = [
    "bitcoin", "ethereum", "binancecoin", "solana", "ripple",
    "cardano", "avalanche-2", "chainlink", "polkadot", "matic-network",
]
DEFAULT_HORIZONS = [1, 3, 7, 14]


def parse_args():
    p = argparse.ArgumentParser(
        description="Pooled multi-coin multi-horizon walk-forward evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--coins", nargs="+", default=DEFAULT_UNIVERSE,
                   help="CoinGecko IDs to include in the pooled dataset.")
    p.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS,
                   help="Forecast horizons in days.")
    p.add_argument("--days", type=int, default=730,
                   help="Lookback window size in days.")
    p.add_argument("--trade-date", default=None,
                   help="Upper date boundary YYYY-MM-DD. Default: today.")
    p.add_argument("--min-train", type=int, default=365,
                   help="Walk-forward training window in unique dates.")
    p.add_argument("--models", nargs="+", default=["rf", "lgb", "arima"],
                   choices=["rf", "lgb", "arima"],
                   help="Which models to evaluate.")
    p.add_argument("--output-dir", default="data/multi",
                   help="Directory for prediction CSVs and summary.")
    p.add_argument("--no-technical", action="store_true",
                   help="Disable stockstats technical indicator features.")
    p.add_argument("--no-cross-asset", action="store_true",
                   help="Disable cross-asset (BTC-anchored) features.")
    p.add_argument("--no-onchain", action="store_true",
                   help="Disable realtime on-chain (funding/TVL/stablecoin) features.")
    p.add_argument("--onchain-pit", action="store_true",
                   help="Enable PIT on-chain features from the bitemporal "
                        "store (MVRV, flows, Puell, TVL). Safe for backtests.")
    return p.parse_args()


def build_pooled_transformed(
    coins: list, horizons: list, days: int, trade_date: str | None,
    add_technical: bool, add_cross_asset: bool, add_onchain: bool,
    add_onchain_pit: bool = False,
) -> pd.DataFrame:
    """Build pooled dataset and apply data_transform per-coin.

    Returns a date-indexed DataFrame with `coin_id` column, ready for
    pooled walk-forward evaluation.
    """
    from tradingagents.models.model_utils import build_pooled_dataset, data_transform

    logger = logging.getLogger(__name__)

    pooled_raw = build_pooled_dataset(
        coin_universe=coins,
        lookback_days=days,
        horizons=horizons,
        trade_date=trade_date,
        add_technical=add_technical,
        add_cross_asset=add_cross_asset,
        add_onchain=add_onchain,
        add_onchain_pit=add_onchain_pit,
    )
    if pooled_raw.empty:
        raise RuntimeError("build_pooled_dataset returned empty DataFrame")

    logger.info(f"Raw pooled shape: {pooled_raw.shape}")

    # Apply data_transform per coin so .shift() respects coin boundaries,
    # then re-concat and set date as index for pooled walk-forward.
    pieces = []
    for coin in pooled_raw["coin_id"].unique():
        sub = pooled_raw[pooled_raw["coin_id"] == coin].drop(columns=["coin_id"])
        if sub.empty:
            continue
        first_future = sub.index.max() + pd.Timedelta(days=1)
        try:
            reframed, _ = data_transform(
                sub, first_future, include_future_row=False, horizons=horizons,
            )
        except Exception as e:
            logger.warning(f"data_transform failed for {coin}: {e}")
            continue
        reframed["coin_id"] = coin
        pieces.append(reframed)

    if not pieces:
        raise RuntimeError("No coins produced transformed data")

    pooled = pd.concat(pieces, ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled = pooled.set_index("date").sort_index()
    logger.info(f"Transformed pooled shape: {pooled.shape}, coins: {pooled['coin_id'].nunique()}")
    return pooled


def run_arima_per_coin(pooled: pd.DataFrame, horizon: int, min_train: int) -> tuple:
    """Run ARIMA per-coin at the given horizon and concatenate predictions.

    ARIMA does not pool (each series has its own process), so we loop over
    coins and call arima_model.walk_forward_horizon for each.

    Returns (pred_df, metrics_dict) — same contract as rf/lgb model_run_pooled.
    """
    from tradingagents.models import arima_model, model_utils as mu, lgb_model

    logger = logging.getLogger(__name__)

    rows = []
    for coin in pooled["coin_id"].unique():
        sub = pooled[pooled["coin_id"] == coin].sort_index()
        if len(sub) < min_train + horizon + 10:
            logger.info(f"  {coin}: too few rows ({len(sub)}), skipping")
            continue

        # Build ARIMA-compatible df: target column `prices` + exog subset
        feat_subset = [
            c for c in arima_model.ARIMA_EXOG_FEATURES if c in sub.columns
        ]
        if not feat_subset:
            logger.warning(f"  {coin}: no ARIMA exog features found, skipping")
            continue

        df_arima = sub[["prices"] + feat_subset].copy().dropna()
        if df_arima.empty or len(df_arima) < min_train + horizon + 10:
            logger.info(f"  {coin}: too few clean rows after dropna, skipping")
            continue

        try:
            preds, actuals, ws = arima_model.walk_forward_horizon(
                df_arima, horizon, min_train_window=min_train,
            )
        except Exception as e:
            logger.warning(f"  {coin}: ARIMA walk-forward failed: {e}")
            continue

        # eval_dates: the date of each actual we predicted
        # preds[i] is the h-step-ahead forecast made at end_train=ws+i,
        # and the actual is the price at index ws+i+horizon-1.
        for i, (p, a) in enumerate(zip(preds, actuals)):
            target_idx = ws + i + horizon - 1
            if target_idx >= len(df_arima):
                break
            rows.append({
                "date": df_arima.index[target_idx],
                "coin_id": coin,
                "prediction": float(p),
                "actual": float(a),
            })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        metrics = {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}
        return pred_df, metrics

    metrics = mu.compute_metrics(pred_df["actual"].values, pred_df["prediction"].values)
    metrics["directional_accuracy"] = lgb_model._dir_acc(pred_df, pooled, horizon)
    return pred_df, metrics


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  Building pooled dataset\n{'='*60}")
    print(f"  Coins       : {len(args.coins)} ({', '.join(args.coins)})")
    print(f"  Horizons    : {args.horizons}")
    print(f"  Days        : {args.days}")
    print(f"  Technical   : {'yes' if not args.no_technical else 'no'}")
    print(f"  Cross-asset : {'yes' if not args.no_cross_asset else 'no'}")
    print(f"  On-chain    : {'yes' if not args.no_onchain else 'no'}")
    print(f"  On-chain PIT: {'yes' if args.onchain_pit else 'no'}")

    try:
        pooled = build_pooled_transformed(
            coins=args.coins,
            horizons=args.horizons,
            days=args.days,
            trade_date=args.trade_date,
            add_technical=not args.no_technical,
            add_cross_asset=not args.no_cross_asset,
            add_onchain=not args.no_onchain,
            add_onchain_pit=args.onchain_pit,
        )
    except Exception as e:
        print(f"ERROR: Failed to build pooled dataset: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"\nPooled shape: {pooled.shape}  "
          f"({pooled['coin_id'].nunique()} coins, "
          f"{pooled.index.nunique()} unique dates)")

    summary = []

    for horizon in args.horizons:
        for model_name in args.models:
            # ARIMA only for short horizons — longer horizons with recursive
            # or multi-step ARIMA collapse in noisy crypto series.
            if model_name == "arima" and horizon > 3:
                continue
            print(f"\n--- {model_name.upper()} h={horizon} ---")
            try:
                if model_name == "rf":
                    from tradingagents.models import rf_model
                    pred_df, metrics = rf_model.model_run_pooled(
                        pooled, horizon, args.min_train,
                    )
                elif model_name == "lgb":
                    from tradingagents.models import lgb_model
                    pred_df, metrics = lgb_model.model_run_pooled(
                        pooled, horizon, args.min_train,
                    )
                elif model_name == "arima":
                    pred_df, metrics = run_arima_per_coin(pooled, horizon, args.min_train)
                else:
                    continue

                csv_path = output_dir / f"preds_{model_name}_h{horizon}.csv"
                pred_df.to_csv(csv_path, index=False)

                summary.append({
                    "model": model_name,
                    "horizon": horizon,
                    "r2": metrics["r2"],
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "dir_acc": metrics.get("directional_accuracy", 0.0),
                    "n_predictions": len(pred_df),
                })
                print(f"  R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}  "
                      f"DirAcc={metrics.get('directional_accuracy', 0):.1%}  "
                      f"n={len(pred_df)}")
                print(f"  Saved -> {csv_path}")
            except Exception as e:
                print(f"  FAILED: {e}")
                traceback.print_exc()

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_path = output_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n{'='*60}\n  Summary\n{'='*60}")
        print(summary_df.to_string(index=False))
        print(f"\nSummary CSV -> {summary_path}")
    else:
        print("\nNo models completed successfully.")


if __name__ == "__main__":
    main()
