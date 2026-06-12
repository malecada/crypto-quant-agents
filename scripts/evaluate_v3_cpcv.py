#!/usr/bin/env python
"""V3 CPCV evaluation — runs Combinatorial Purged CV on the V3 backtest.

For each coin:
  1. Generate CPCV splits over the date range
  2. Run V3 backtest on each test fold
  3. Collect per-split Sharpe ratios
  4. Compute Deflated Sharpe Ratio adjusting for n_trials

Outputs:
  data/v3_cpcv/{coin}/sharpe_distribution.parquet
  data/v3_cpcv/{coin}/summary.json

NOTE (Phase-7 simplification): by default (--retrain-per-fold not set) models
are NOT retrained per fold. The pre-trained `data/checkpoints/v3_models_{coin}.pkl`
is reused across all splits.

Use --retrain-per-fold to train a fresh MultiHorizonEnsemble on each fold's
train_idx (the correct López de Prado CPCV protocol). The regime bundle is
still loaded from disk (not refit per fold) since the HMM is fitted on long
history and is not the subject of the CPCV evaluation.

Usage:
    python scripts/evaluate_v3_cpcv.py \\
        --coins bitcoin ethereum \\
        --start 2024-05-01 --end 2026-04-15

    # Per-fold retraining (lgb-only, ~3-5 min):
    python scripts/evaluate_v3_cpcv.py \\
        --coins bitcoin ethereum \\
        --start 2024-05-01 --end 2026-04-15 \\
        --retrain-per-fold --retrain-members lgb
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

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv  # noqa: E402
from tradingagents.strategies.v3.backtest.cpcv import cpcv_splits  # noqa: E402
from tradingagents.strategies.v3.backtest.dsr import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    variance_of_sr,
)
from tradingagents.strategies.v3.backtest.runner_v3 import run_v3_backtest  # noqa: E402
from tradingagents.strategies.v3.config import V3Config  # noqa: E402
from tradingagents.strategies.v3.models.multi_horizon import MultiHorizonEnsemble  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Feature column names — must match what _build_v3_features_at produces when
# both microstructure and derivatives data are available.
_FEATURE_COLS = [
    "ret_1d",
    "ret_5d",
    "vol_5d",
    "vol_21d",
    "ofi_proxy",
    "ofi_proxy_w",
    "vol_dispersion",
    "funding_rate",
    "funding_rate_ma7",
]


def _load_optional_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def _load_required_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_ohlcv_for_coin(coin: str, days: int = 2500) -> pd.DataFrame:
    # _load_crypto_ohlcv takes coingecko_id + curr_date (not coin + days)
    # Use end of eval window as curr_date so we get all needed history
    df = _load_crypto_ohlcv(coingecko_id=coin, curr_date="2026-04-15")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    return df


def _build_global_features(
    prices: pd.Series,
    micro: pd.DataFrame,
    deriv: pd.DataFrame,
) -> pd.DataFrame:
    """Build a full-history feature DataFrame aligned to the prices index.

    Produces the same 9 features that _build_v3_features_at generates per bar,
    but computed once for the whole series (vectorised) to avoid O(n²) cost
    when training per fold.

    Returns a DataFrame with DatetimeIndex matching prices.index (tz-aware UTC).
    Rows with NaN (first 21 bars) are kept so slicing by integer index remains
    aligned; callers should dropna() before fitting.
    """
    idx = prices.index

    ret_series = prices.pct_change()
    df = pd.DataFrame(index=idx)
    df["ret_1d"] = ret_series
    df["ret_5d"] = prices.pct_change(5)
    df["vol_5d"] = ret_series.rolling(5).std()
    df["vol_21d"] = ret_series.rolling(21).std()

    # tz-normalise helper
    def _tz_norm(other_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is not None and other_idx.tz is None:
            return other_idx.tz_localize("UTC")
        if idx.tz is None and other_idx.tz is not None:
            return other_idx.tz_localize(None)
        return other_idx

    # Microstructure columns (ofi_proxy, ofi_proxy_w, vol_dispersion)
    micro_cols = ["ofi_proxy", "ofi_proxy_w", "vol_dispersion"]
    if not micro.empty:
        m = micro.copy()
        m.index = _tz_norm(m.index)
        for col in micro_cols:
            if col in m.columns:
                df[col] = m[col].reindex(idx, method="ffill")
            else:
                df[col] = 0.0
    else:
        for col in micro_cols:
            df[col] = 0.0

    # Derivatives columns (funding_rate, funding_rate_ma7)
    deriv_cols = ["funding_rate", "funding_rate_ma7"]
    if not deriv.empty:
        d = deriv.copy()
        d.index = _tz_norm(d.index)
        for col in deriv_cols:
            if col in d.columns:
                df[col] = d[col].reindex(idx, method="ffill")
            else:
                df[col] = 0.0
    else:
        for col in deriv_cols:
            df[col] = 0.0

    df = df.fillna(0.0)
    return df


def _train_fold_mhe(
    features: pd.DataFrame,
    returns: pd.Series,
    train_idx: np.ndarray,
    members: tuple[str, ...] = ("lgb",),
) -> MultiHorizonEnsemble:
    """Train a fresh MultiHorizonEnsemble on the fold's train_idx.

    Args:
        features: Full-history feature DataFrame (price-index aligned).
        returns: Full-history simple-return Series (aligned to features).
        train_idx: Integer indices (into features.index) forming the train fold.
        members: Ensemble member names, e.g. ``("lgb",)`` or
            ``("lgb", "xgb", "catboost")``.

    Returns:
        Fitted MultiHorizonEnsemble.
    """
    X_train = features.iloc[train_idx]
    y_train = returns.iloc[train_idx]

    # Align indices (should already be aligned, but guard against any drift)
    common = X_train.index.intersection(y_train.index)
    X_train = X_train.loc[common]
    y_train = y_train.loc[common]

    mhe = MultiHorizonEnsemble(horizons=(3, 7, 14, 21))
    mhe.fit(X_train, y_train, members=members)
    return mhe


def evaluate_coin_cpcv(
    coin: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg: V3Config,
    microstructure_dir: Path,
    derivatives_dir: Path,
    regime_dir: Path,
    models_dir: Path,
    n_groups: int = 8,
    test_groups: int = 2,
    embargo: int = 14,
    n_trials_for_dsr: int = 12,
    retrain_per_fold: bool = False,
    retrain_members: tuple[str, ...] = ("lgb",),
    sma30_filter: bool = False,
    sma30_multiplier: tuple[float, float] = (1.5, 0.5),
) -> dict:
    """Run CPCV evaluation for one coin. Returns dict of per-split metrics + DSR."""
    ohlcv = _load_ohlcv_for_coin(coin)
    if ohlcv.index.tz is None:
        ohlcv.index = ohlcv.index.tz_localize("UTC")
    prices = ohlcv["Close"]
    returns = prices.pct_change().fillna(0.0)

    # Slice to evaluation window
    mask = (prices.index >= start) & (prices.index <= end)
    bars = prices.index[mask]
    if len(bars) < n_groups * 2 * embargo:
        raise ValueError(f"Too few bars ({len(bars)}) for CPCV with n_groups={n_groups}")

    micro = _load_optional_parquet(microstructure_dir / f"{coin}.parquet")
    deriv = _load_optional_parquet(derivatives_dir / f"{coin}.parquet")
    regime_bundle = _load_required_pickle(regime_dir / f"regime_hmm_v3_{coin}.pkl")

    # Pre-compute global feature matrix (needed for per-fold training)
    global_features: pd.DataFrame | None = None
    if retrain_per_fold:
        logger.info("[%s] Building global feature matrix for per-fold retraining...", coin)
        global_features = _build_global_features(prices, micro, deriv)
        logger.info("[%s] Feature matrix shape: %s", coin, global_features.shape)

    if not retrain_per_fold:
        mh_bundle = _load_required_pickle(models_dir / f"v3_models_{coin}.pkl")
    else:
        mh_bundle = None  # will be built per fold

    splits = list(
        cpcv_splits(
            n_samples=len(bars),
            n_groups=n_groups,
            test_groups=test_groups,
            embargo=embargo,
        )
    )
    logger.info("[%s] %d CPCV splits to evaluate (retrain_per_fold=%s)", coin, len(splits), retrain_per_fold)

    per_split_records = []
    fold_train_times: list[float] = []

    for split_idx, split in enumerate(splits):
        if len(split.test_idx) == 0:
            continue
        test_bars = bars[split.test_idx]
        if len(test_bars) < 5:
            continue
        test_start = test_bars[0]
        test_end = test_bars[-1]

        if retrain_per_fold:
            # Map split.train_idx (relative to bars window) back to global price index.
            # bars is a DatetimeIndex slice of prices.index; split.train_idx are integer
            # offsets into bars. We need the corresponding integer positions in prices.
            train_bars = bars[split.train_idx]

            # Find integer positions of train_bars in prices.index
            # (prices.index is a superset of bars — we need iloc positions)
            price_pos = prices.index.get_indexer(train_bars)
            valid_mask = price_pos >= 0
            price_pos_valid = price_pos[valid_mask]

            if len(price_pos_valid) < 30:
                logger.warning(
                    "[%s] split %d: only %d valid train bars; skipping",
                    coin, split_idx, len(price_pos_valid)
                )
                continue

            t0 = time.perf_counter()
            try:
                fold_mhe = _train_fold_mhe(
                    features=global_features,
                    returns=returns,
                    train_idx=price_pos_valid,
                    members=retrain_members,
                )
            except Exception:
                logger.exception("[%s] split %d: fold training failed; skipping", coin, split_idx)
                continue
            train_elapsed = time.perf_counter() - t0
            fold_train_times.append(train_elapsed)
            logger.debug(
                "[%s] split %d/%d trained in %.1fs (%d train bars)",
                coin, split_idx + 1, len(splits), train_elapsed, len(price_pos_valid)
            )
            active_bundle = fold_mhe
        else:
            active_bundle = mh_bundle

        try:
            result = run_v3_backtest(
                coin=coin,
                prices=prices,
                returns=returns,
                microstructure_features=micro,
                derivatives_features=deriv,
                regime_bundle=regime_bundle,
                multi_horizon_bundle=active_bundle,
                config=cfg,
                start=test_start,
                end=test_end,
                ticker=coin.upper(),
                sma30_filter=sma30_filter,
                sma30_multiplier=sma30_multiplier[0],
            )
            per_split_records.append({
                "split_idx": split_idx,
                "test_start": test_start,
                "test_end": test_end,
                "n_bars": len(test_bars),
                "sharpe_ratio": float(result.metrics.get("sharpe_ratio", 0.0)),
                "total_return": float(result.metrics.get("total_return", 0.0)),
                "max_drawdown": float(result.metrics.get("max_drawdown", 0.0)),
            })
        except Exception:
            logger.exception("[%s] split %d failed", coin, split_idx)
            continue

    if not per_split_records:
        raise RuntimeError(f"No splits completed for {coin}")

    df = pd.DataFrame(per_split_records)
    sharpes = df["sharpe_ratio"].values

    sr_obs = float(np.mean(sharpes))
    var_sr = variance_of_sr(sharpes)
    sr_exp = expected_max_sharpe(n_trials=n_trials_for_dsr, var_sr=max(var_sr, 1e-9))
    dsr = deflated_sharpe_ratio(
        sr_observed=sr_obs,
        sr_expected_under_null=sr_exp,
        se_sr=float(np.sqrt(max(var_sr, 1e-9))),
    )

    summary: dict = {
        "coin": coin,
        "n_splits": len(per_split_records),
        "sharpe_mean": sr_obs,
        "sharpe_median": float(np.median(sharpes)),
        "sharpe_std": float(np.std(sharpes, ddof=1)),
        "sharpe_min": float(np.min(sharpes)),
        "sharpe_max": float(np.max(sharpes)),
        "var_sr": float(var_sr),
        "sr_expected_under_null": float(sr_exp),
        "dsr": float(dsr),
        "n_trials_for_dsr": n_trials_for_dsr,
        "retrain_per_fold": retrain_per_fold,
        "retrain_members": list(retrain_members),
        "sma30_filter": sma30_filter,
        "sma30_multiplier_aligned": sma30_multiplier[0],
        "sma30_multiplier_against": sma30_multiplier[1],
    }
    if fold_train_times:
        summary["fold_train_time_mean_s"] = float(np.mean(fold_train_times))
        summary["fold_train_time_total_s"] = float(np.sum(fold_train_times))

    return {
        "splits_df": df,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--coins", nargs="+", default=["bitcoin", "ethereum"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--microstructure-dir", default="data/microstructure")
    parser.add_argument("--derivatives-dir", default="data/derivatives")
    parser.add_argument("--regime-dir", default="data/checkpoints")
    parser.add_argument("--models-dir", default="data/checkpoints")
    parser.add_argument("--out-dir", default="data/v3_cpcv")
    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--test-groups", type=int, default=2)
    parser.add_argument("--embargo", type=int, default=14)
    parser.add_argument("--n-trials-dsr", type=int, default=12)
    parser.add_argument(
        "--retrain-per-fold",
        action="store_true",
        default=False,
        help=(
            "Train a fresh MultiHorizonEnsemble on each fold's train_idx "
            "(proper López de Prado CPCV protocol). Default: reuse pre-trained model."
        ),
    )
    parser.add_argument(
        "--retrain-members",
        nargs="+",
        default=["lgb"],
        metavar="MEMBER",
        help="Ensemble members to use when --retrain-per-fold is set (default: lgb).",
    )
    parser.add_argument(
        "--sma30-filter",
        action="store_true",
        default=False,
        help="Apply V2 SMA30 trend filter (1.5x aligned, 0.5x against) as final position multiplier.",
    )
    parser.add_argument(
        "--sma30-multiplier",
        type=float,
        nargs=2,
        default=[1.5, 0.5],
        metavar=("ALIGNED_MULT", "AGAINST_MULT"),
        help="SMA30 multipliers: aligned_mult against_mult (default: 1.5 0.5).",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = V3Config()
    start_ts = pd.Timestamp(args.start, tz="UTC")
    end_ts = pd.Timestamp(args.end, tz="UTC")

    retrain_members = tuple(args.retrain_members)

    for coin in args.coins:
        try:
            result = evaluate_coin_cpcv(
                coin=coin,
                start=start_ts,
                end=end_ts,
                cfg=cfg,
                microstructure_dir=Path(args.microstructure_dir),
                derivatives_dir=Path(args.derivatives_dir),
                regime_dir=Path(args.regime_dir),
                models_dir=Path(args.models_dir),
                n_groups=args.n_groups,
                test_groups=args.test_groups,
                embargo=args.embargo,
                n_trials_for_dsr=args.n_trials_dsr,
                retrain_per_fold=args.retrain_per_fold,
                retrain_members=retrain_members,
                sma30_filter=args.sma30_filter,
                sma30_multiplier=tuple(args.sma30_multiplier),
            )
        except Exception:
            logger.exception("Failed coin %s", coin)
            continue

        coin_dir = out_dir / coin
        coin_dir.mkdir(parents=True, exist_ok=True)
        result["splits_df"].to_parquet(coin_dir / "sharpe_distribution.parquet")
        with open(coin_dir / "summary.json", "w") as f:
            json.dump(result["summary"], f, indent=2, default=str)

        s = result["summary"]
        extra = ""
        if "fold_train_time_mean_s" in s:
            extra = f" fold_train_mean={s['fold_train_time_mean_s']:.1f}s total={s['fold_train_time_total_s']:.0f}s"
        logger.info(
            "[%s] n_splits=%d sharpe_mean=%.2f median=%.2f std=%.2f DSR=%.3f%s",
            coin, s["n_splits"], s["sharpe_mean"], s["sharpe_median"],
            s["sharpe_std"], s["dsr"], extra,
        )


if __name__ == "__main__":
    main()
