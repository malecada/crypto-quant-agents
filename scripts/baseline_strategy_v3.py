#!/usr/bin/env python
"""Baseline Strategy V3 — orchestrates the V3 quant pipeline per coin.

Loads:
  - OHLCV (Binance/CoinGecko via tradingagents.dataflows)
  - Microstructure features (data/microstructure/{coin}.parquet) — optional
  - Derivatives features (data/derivatives/{coin}.parquet) — optional
  - Regime bundle (data/checkpoints/regime_hmm_v3_{coin}.pkl) — required
  - Multi-horizon ensemble (data/checkpoints/v3_models_{coin}.pkl) — required

Runs V3 backtest per coin, aggregates to equal-weighted portfolio, writes
metrics + equity-curve plot to ``--out-dir``.

Usage:
    python scripts/baseline_strategy_v3.py \\
        --coins bitcoin ethereum \\
        --start 2026-01-16 --end 2026-04-15 \\
        --out-dir data/multi_2coins_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("Optional parquet missing: %s — using empty DataFrame", path)
    return pd.DataFrame()


def _load_required_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required pickle missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ohlcv_for_coin(coin: str, days: int = 2500) -> pd.DataFrame:
    df = _load_crypto_ohlcv(coingecko_id=coin, curr_date="2026-04-15")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    parser.add_argument("--start", required=True, help="ISO date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="ISO date YYYY-MM-DD")
    parser.add_argument("--microstructure-dir", default="data/microstructure")
    parser.add_argument("--derivatives-dir", default="data/derivatives")
    parser.add_argument("--regime-dir", default="data/checkpoints")
    parser.add_argument("--models-dir", default="data/checkpoints")
    parser.add_argument("--out-dir", default="data/multi_v3")
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument("--target-vol", type=float, default=0.15)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--signal-deadband", type=float, default=0.02,
                        help="Deadband for consensus signal (default 0.02 to allow realistic signal generation)")
    # Walk-forward retraining flags (matches V2 walk_forward_pooled protocol)
    parser.add_argument("--retrain-per-bar", action="store_true",
                        help="Retrain MultiHorizonEnsemble at each bar (or cadence) using data through bar-1 with purge guard. Matches V2 methodology.")
    parser.add_argument("--retrain-cadence", type=int, default=1,
                        help="Retrain every N bars (default 1=every bar). Set to 7 for weekly retrain.")
    parser.add_argument("--retrain-members", nargs="+", default=["lgb"],
                        help="Ensemble members to train during walk-forward (default: lgb)")
    parser.add_argument("--no-retrain-calibration", action="store_true",
                        help="Disable isotonic calibration during walk-forward retraining (default: calibration off per diagnostic)")
    parser.add_argument("--sma30-filter", action="store_true",
                        help="Apply V2-style SMA30 trend filter as final position multiplier (1.5x aligned, 0.5x against)")
    parser.add_argument("--sma30-multiplier", type=float, default=1.5,
                        help="Aligned-direction multiplier for SMA30 filter (default 1.5 matches V2)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.retrain_per_bar:
        logger.info(
            "Mode: WALK-FORWARD (retrain every %d bar(s), members=%s, calibration=%s)",
            args.retrain_cadence,
            args.retrain_members,
            not args.no_retrain_calibration,
        )
    else:
        logger.info("Mode: FROZEN model (no per-bar retraining)")
    if args.sma30_filter:
        logger.info("SMA30 trend filter ENABLED (multiplier=%.1fx aligned, %.1fx against)", args.sma30_multiplier, 1.0 / args.sma30_multiplier)

    cfg = V3Config(
        target_annual_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )

    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    per_coin_results = {}

    for coin in args.coins:
        logger.info("Running V3 for %s", coin)
        try:
            ohlcv = _load_ohlcv_for_coin(coin)
            ohlcv.index = ohlcv.index.tz_localize("UTC") if ohlcv.index.tz is None else ohlcv.index
            prices = ohlcv["Close"]
            returns = prices.pct_change().fillna(0.0)

            micro_path = Path(args.microstructure_dir) / f"{coin}.parquet"
            deriv_path = Path(args.derivatives_dir) / f"{coin}.parquet"
            regime_path = Path(args.regime_dir) / f"regime_hmm_v3_{coin}.pkl"
            models_path = Path(args.models_dir) / f"v3_models_{coin}.pkl"

            micro = _load_optional_parquet(micro_path)
            deriv = _load_optional_parquet(deriv_path)
            regime_bundle = _load_required_pickle(regime_path)
            mh_bundle = _load_required_pickle(models_path)

            coin_start = max(start_ts, prices.index.min())
            coin_end = min(end_ts, prices.index.max())
            if coin_start >= coin_end:
                logger.warning("No usable bars for %s in [%s, %s]", coin, coin_start, coin_end)
                continue

            result = run_v3_backtest(
                coin=coin,
                prices=prices,
                returns=returns,
                microstructure_features=micro,
                derivatives_features=deriv,
                regime_bundle=regime_bundle,
                multi_horizon_bundle=mh_bundle,
                config=cfg,
                start=coin_start,
                end=coin_end,
                ticker=coin.upper(),
                initial_capital=args.initial_capital,
                signal_deadband=args.signal_deadband,
                retrain_per_bar=args.retrain_per_bar,
                retrain_cadence=args.retrain_cadence,
                retrain_members=tuple(args.retrain_members),
                retrain_use_calibration=not args.no_retrain_calibration,
                sma30_filter=args.sma30_filter,
                sma30_multiplier=args.sma30_multiplier,
            )
            per_coin_results[coin] = result
            logger.info(
                "%s — Sharpe=%.2f Return=%.2f%% MaxDD=%.2f%%",
                coin,
                result.metrics.get("sharpe_ratio", 0.0),
                result.metrics.get("total_return", 0.0) * 100,
                result.metrics.get("max_drawdown", 0.0) * 100,
            )
        except Exception:
            logger.exception("Failed coin %s", coin)
            continue

    if not per_coin_results:
        logger.error("No coins produced results")
        sys.exit(1)

    # Aggregate equal-weighted portfolio
    portfolio_metrics: dict = {"per_coin": {}}
    sharpes, returns_pct, dds = [], [], []
    for coin, result in per_coin_results.items():
        m = result.metrics
        portfolio_metrics["per_coin"][coin] = {
            "sharpe_ratio": float(m.get("sharpe_ratio", 0.0)),
            "total_return": float(m.get("total_return", 0.0)),
            "max_drawdown": float(m.get("max_drawdown", 0.0)),
        }
        sharpes.append(m.get("sharpe_ratio", 0.0))
        returns_pct.append(m.get("total_return", 0.0))
        dds.append(m.get("max_drawdown", 0.0))

    portfolio_metrics["portfolio_avg"] = {
        "sharpe_ratio": float(np.mean(sharpes)),
        "total_return": float(np.mean(returns_pct)),
        "max_drawdown": float(np.mean(dds)),
    }

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for coin, result in per_coin_results.items():
        if hasattr(result, "equity_curve") and result.equity_curve is not None:
            eq = result.equity_curve
            if isinstance(eq, (list, np.ndarray)):
                ax.plot(np.arange(len(eq)), eq, label=coin, alpha=0.7)
    ax.set_title("V3 Equity Curves per Coin")
    ax.set_xlabel("Bar")
    ax.set_ylabel("Equity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plot_path = out_dir / "baseline_v3_equity.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote plot %s", plot_path)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(portfolio_metrics, f, indent=2)
    logger.info("Wrote metrics %s", metrics_path)

    # Print summary table
    print("\nV3 Per-Coin Results")
    print("-" * 60)
    print(f"{'Coin':<15} {'Sharpe':>10} {'Return':>10} {'MaxDD':>10}")
    print("-" * 60)
    for coin, m in portfolio_metrics["per_coin"].items():
        print(
            f"{coin:<15} {m['sharpe_ratio']:>10.2f} "
            f"{m['total_return']*100:>9.2f}% {m['max_drawdown']*100:>9.2f}%"
        )
    print("-" * 60)
    pa = portfolio_metrics["portfolio_avg"]
    print(
        f"{'PORTFOLIO':<15} {pa['sharpe_ratio']:>10.2f} "
        f"{pa['total_return']*100:>9.2f}% {pa['max_drawdown']*100:>9.2f}%"
    )


if __name__ == "__main__":
    main()
