"""LightGBM model for multi-horizon, multi-coin pooled prediction.

Unlike rf_model and arima_model (which train per-coin single-horizon models),
this module trains one LightGBM per (horizon) across all coins pooled, with
coin identity as a categorical feature. This gives 10x more training data
and lets the model learn cross-asset patterns.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from tradingagents.dataflows.config import get_config
from tradingagents.models import model_utils as mu

logger = logging.getLogger(__name__)


def _build_lgb():
    """Build a LightGBM regressor with config-driven hyperparameters.

    Uses single-threaded fitting (n_jobs=1): walk-forward evaluation trains
    hundreds of small models and the multi-threading overhead dominates the
    per-model fit time on small datasets.
    """
    import lightgbm as lgb

    cfg = get_config().get("prediction_models", {})
    return lgb.LGBMRegressor(
        n_estimators=cfg.get("lgb_n_estimators", 500),
        max_depth=cfg.get("lgb_max_depth", -1),
        learning_rate=cfg.get("lgb_learning_rate", 0.05),
        num_leaves=cfg.get("lgb_num_leaves", 31),
        min_child_samples=cfg.get("lgb_min_child", 20),
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )


def _ensure_date_indexed(pooled_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the pooled df so the index is a datetime of the row's date.

    Handles two cases:
      1. df already has a datetime index (return as-is)
      2. df has a `date` column — set as index

    If neither works, raise ValueError.
    """
    if isinstance(pooled_df.index, pd.DatetimeIndex):
        return pooled_df
    if "date" in pooled_df.columns:
        out = pooled_df.copy()
        out["date"] = pd.to_datetime(out["date"])
        return out.set_index("date")
    raise ValueError(
        "pooled_df must either have a DatetimeIndex or a `date` column"
    )


def _dir_acc(pred_df: pd.DataFrame, pooled_df: pd.DataFrame, horizon: int) -> float:
    """Directional accuracy of predictions vs actuals.

    For each prediction row, look up the same coin's previous day's `prices`
    in pooled_df and compare the predicted direction (up/down) to the actual
    direction.

    Vectorised over coins: builds a per-coin date→prev-price map once, then
    uses a vectorised merge to compute directions for the entire pred_df.
    """
    del horizon  # part of the public contract; not used in this implementation

    if pred_df.empty:
        return 0.0

    pooled_df = _ensure_date_indexed(pooled_df)

    # Build per-coin previous-price series (shift by 1 bar within each coin)
    prev_rows = []
    for coin in pooled_df["coin_id"].unique():
        sub = pooled_df[pooled_df["coin_id"] == coin][["prices"]].sort_index()
        prev = sub["prices"].shift(1)
        tmp = pd.DataFrame({
            "date": sub.index,
            "coin_id": coin,
            "prev_price": prev.values,
        })
        prev_rows.append(tmp)
    prev_df = pd.concat(prev_rows, ignore_index=True)
    prev_df["date"] = pd.to_datetime(prev_df["date"])

    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])

    merged = pred_df.merge(prev_df, on=["date", "coin_id"], how="left")
    merged = merged[merged["prev_price"].notna() & (merged["prev_price"] > 0)]
    if merged.empty:
        return 0.0

    pred_dir = np.where(merged["prediction"].values > merged["prev_price"].values, 1, -1)
    actual_dir = np.where(merged["actual"].values > merged["prev_price"].values, 1, -1)
    correct = int(np.sum(pred_dir == actual_dir))
    total = int(len(merged))

    return correct / total if total > 0 else 0.0


def walk_forward_pooled(
    pooled_df: pd.DataFrame,
    horizon: int,
    min_train_window: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """Walk-forward eval on the pooled multi-coin dataset.

    For each unique date at position >= min_train_window:
      - Train on all coin-rows with date < current
      - Predict all coin-rows with date == current
    Features: all columns except target columns (`prices_h*`), `coin_id`,
    and `date`. `coin_id` is encoded as an integer `coin_int` feature so
    the model can learn per-coin offsets.

    Args:
        pooled_df: Pooled DataFrame from build_pooled_dataset + data_transform
            stacked across coins. May be date-indexed or have a `date` column.
        horizon: Which `prices_h{h}` column to predict.
        min_train_window: Number of initial dates to reserve as training only
            (walk-forward starts after this).

    Returns:
        (predictions_df, metrics_dict) where predictions_df has columns
        [date, coin_id, prediction, actual] and metrics_dict has keys
        r2/mae/rmse/mape/directional_accuracy.
    """
    pooled_df = _ensure_date_indexed(pooled_df)

    target_col = f"prices_h{horizon}"
    if target_col not in pooled_df.columns:
        raise ValueError(
            f"target column {target_col} not in pooled_df "
            f"(have: {[c for c in pooled_df.columns if c.startswith('prices')]})"
        )

    # Columns that must NOT become features
    exclude_cols = {"coin_id"}
    for c in pooled_df.columns:
        if c.startswith("prices_h"):
            exclude_cols.add(c)
    feature_cols = [c for c in pooled_df.columns if c not in exclude_cols]

    # Encode coin_id as categorical integer
    coin_ids = sorted(pooled_df["coin_id"].unique())
    coin_to_int = {c: i for i, c in enumerate(coin_ids)}
    pooled_df = pooled_df.copy()
    pooled_df["coin_int"] = pooled_df["coin_id"].map(coin_to_int).astype(int)
    if "coin_int" not in feature_cols:
        feature_cols.append("coin_int")

    unique_dates = sorted(pooled_df.index.unique())
    if len(unique_dates) <= min_train_window:
        raise ValueError(
            f"Need >{min_train_window} unique dates; got {len(unique_dates)}"
        )

    rows = []
    for i in range(min_train_window, len(unique_dates)):
        cur_date = unique_dates[i]
        train = pooled_df.loc[pooled_df.index < cur_date].dropna(subset=[target_col])
        test = pooled_df.loc[pooled_df.index == cur_date].dropna(subset=[target_col])
        if train.empty or test.empty:
            continue

        X_tr = train[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train[target_col].to_numpy(dtype=np.float32)
        X_te = test[feature_cols].to_numpy(dtype=np.float32)

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = _build_lgb()
        model.fit(X_tr_s, y_tr)
        preds = model.predict(X_te_s)

        test_rows = test.reset_index().rename(columns={"index": "date"})
        for j in range(len(test)):
            rows.append({
                "date": cur_date,
                "coin_id": test_rows.iloc[j]["coin_id"],
                "prediction": float(preds[j]),
                "actual": float(test.iloc[j][target_col]),
                "ref_price": float(test.iloc[j]["prices"]),
            })

    pred_df = pd.DataFrame(rows)
    if pred_df.empty:
        metrics = {"r2": 0.0, "mae": 0.0, "rmse": 0.0, "mape": 0.0, "directional_accuracy": 0.0}
        return pred_df, metrics

    metrics = mu.compute_metrics(pred_df["actual"].values, pred_df["prediction"].values)
    metrics["directional_accuracy"] = _dir_acc(pred_df, pooled_df, horizon)
    return pred_df, metrics


def model_run_pooled(
    pooled_df: pd.DataFrame,
    horizon: int,
    min_train_window: int = 365,
) -> tuple[pd.DataFrame, dict]:
    """Public entrypoint mirroring the contract used by evaluate scripts.

    Currently a thin wrapper around walk_forward_pooled(). Returns the same
    (pred_df, metrics_dict) tuple.
    """
    return walk_forward_pooled(pooled_df, horizon, min_train_window)


# ── Live inference path: fit-once + predict ──────────────────────────


def _select_feature_cols(pooled_df: pd.DataFrame) -> list[str]:
    """Return the canonical feature column list matching walk_forward_pooled.

    Drops `coin_id` and any `prices_h*` target columns. All remaining columns
    are treated as features. Caller is responsible for adding coin_int
    upstream if needed.
    """
    exclude_cols = {"coin_id"}
    for c in pooled_df.columns:
        if c.startswith("prices_h"):
            exclude_cols.add(c)
    return [c for c in pooled_df.columns if c not in exclude_cols]


def fit_pooled_full(
    pooled_df: pd.DataFrame,
    horizon: int,
    feature_cols: list[str] | None = None,
) -> dict:
    """Fit a single LGB regressor on the entire pooled dataset for one horizon.

    Unlike model_run_pooled (walk-forward eval), this fits ONE model on all
    available rows and returns it, so live inference can call .predict()
    directly. Used by the live cycle's daily retrain step.

    Hyperparameters and feature selection match `walk_forward_pooled` exactly:
      - LGB params from `_build_lgb()` (config-driven; defaults n_estimators=500,
        max_depth=-1, learning_rate=0.05, num_leaves=31, min_child_samples=20,
        random_state=42, n_jobs=1)
      - Features: all columns except `coin_id` and any `prices_h*` targets,
        plus an integer-encoded `coin_int` derived from `coin_id` so the model
        can learn per-coin offsets.
      - Min-max scaling fitted on the training rows; the fitted scaler is
        bundled so live inference applies the same transform.

    Args:
        pooled_df: Pooled DataFrame post-`data_transform` (must already have
            the `prices_h{horizon}` target column).
        horizon: Which `prices_h{h}` column to predict.
        feature_cols: Optional override of the feature list. When None we
            replicate walk_forward_pooled's selection rule.

    Returns:
        Dict with:
          - 'booster': fitted sklearn-style LGBMRegressor
          - 'feature_names': list of feature column names used (in fit order)
          - 'horizon': echoed input
          - 'target_col': e.g. 'prices_h7'
          - 'n_train_rows': rows used after dropping NaN target rows
          - 'scaler': fitted MinMaxScaler used at fit time
          - 'coin_to_int': mapping of coin_id -> integer (for live encoding)

    Raises:
        ValueError: target column missing or no rows remain after NaN drop.
    """
    target_col = f"prices_h{horizon}"
    if target_col not in pooled_df.columns:
        raise ValueError(
            f"target column {target_col} not in pooled_df "
            f"(have: {[c for c in pooled_df.columns if c.startswith('prices')]})"
        )

    df = pooled_df.copy()

    # Encode coin_id as integer to match walk_forward_pooled.
    if "coin_id" in df.columns:
        coin_ids = sorted(df["coin_id"].unique())
        coin_to_int = {c: i for i, c in enumerate(coin_ids)}
        df["coin_int"] = df["coin_id"].map(coin_to_int).astype(int)
    else:
        coin_to_int = {}

    if feature_cols is None:
        feature_cols = _select_feature_cols(df)
        if "coin_int" in df.columns and "coin_int" not in feature_cols:
            feature_cols.append("coin_int")

    # Drop NaN rows in the target before fit.
    train = df.dropna(subset=[target_col])
    if train.empty:
        raise ValueError(
            f"No training rows after dropping NaN in {target_col}"
        )

    X = train[feature_cols].to_numpy(dtype=np.float32)
    y = train[target_col].to_numpy(dtype=np.float32)

    scaler = MinMaxScaler()
    X_s = scaler.fit_transform(X)

    booster = _build_lgb()
    booster.fit(X_s, y)

    return {
        "booster": booster,
        "feature_names": list(feature_cols),
        "horizon": int(horizon),
        "target_col": target_col,
        "n_train_rows": int(len(train)),
        "scaler": scaler,
        "coin_to_int": coin_to_int,
    }


def predict_pooled(
    bundle: dict,
    feature_row: pd.DataFrame,
) -> float:
    """Predict y for a single feature row using a bundle from fit_pooled_full.

    Applies the bundled MinMaxScaler before calling the booster, matching
    the fit-time transform. If the bundle has a non-empty `coin_to_int`
    mapping and the feature_row has a `coin_id` column but lacks `coin_int`,
    the integer code is filled in automatically.

    Args:
        bundle: Output of `fit_pooled_full`.
        feature_row: DataFrame with at least the columns in
            bundle['feature_names']. Only the first row is used.

    Returns:
        The predicted target as a Python float.
    """
    feat_names = bundle["feature_names"]
    row = feature_row.copy()

    # Auto-fill coin_int when the bundle was fit with coin encoding and
    # the caller passed coin_id without coin_int.
    if (
        "coin_int" in feat_names
        and "coin_int" not in row.columns
        and "coin_id" in row.columns
        and bundle.get("coin_to_int")
    ):
        row["coin_int"] = row["coin_id"].map(bundle["coin_to_int"]).astype(int)

    X = row[feat_names].to_numpy(dtype=np.float32)
    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)
    pred = bundle["booster"].predict(X)
    return float(pred[0])


# ── Agent-facing single-date multi-horizon forecast ──────────────────


# Historical DirAcc benchmarks from walk-forward evaluation (see THESIS_FINDINGS.md).
# Used to inform agents about prediction quality per (coin, horizon).
_HISTORICAL_DIRACC = {
    "bitcoin":     {7: 0.749, 14: 0.846},
    "ethereum":    {7: 0.744, 14: 0.758},
    "binancecoin": {7: 0.603, 14: 0.675},
    "solana":      {7: 0.584, 14: 0.603},
    "ripple":      {7: 0.488, 14: 0.510},
    "dogecoin":    {7: 0.479, 14: 0.449},
    "cardano":     {7: 0.477, 14: 0.452},
}


def _select_pool(symbol: str, pool_coins: list[str] | None) -> tuple[list[str], str]:
    """Auto-select training pool based on symbol.

    Returns (pool_coins, pool_label). 2-coin for BTC/ETH, 2+1 for altcoins.
    """
    if pool_coins is not None:
        label = f"{' + '.join(pool_coins)} (custom, {len(pool_coins)}-coin)"
        return pool_coins, label

    if symbol in ("bitcoin", "ethereum"):
        return ["bitcoin", "ethereum"], "bitcoin + ethereum (2-coin)"
    return ["bitcoin", "ethereum", symbol], f"bitcoin + ethereum + {symbol} (2+1)"


def _format_dir_acc_note(symbol: str, horizon: int) -> str:
    hist = _HISTORICAL_DIRACC.get(symbol, {}).get(horizon)
    if hist is None:
        return "not benchmarked"
    return f"~{hist:.0%}"


def forecast_next(
    symbol: str,
    horizons: list[int] | None = None,
    pool_coins: list[str] | None = None,
    lookback_days: int | None = None,
    trade_date: str | None = None,
) -> str:
    """Agent-facing LightGBM multi-horizon pooled prediction.

    Trains one LGB per horizon on a pooled dataset of `pool_coins` (auto-
    selected if None) and returns the prediction for `symbol` at each
    horizon. Used by the prediction analyst agent for structured multi-
    horizon signals.

    Args:
        symbol: Target coin CoinGecko ID (e.g. "bitcoin").
        horizons: Forecast horizons in days (default: [7, 14]).
        pool_coins: Training pool. If None, auto-select 2-coin (BTC/ETH)
            for BTC/ETH themselves, or 2+1 pool (BTC+ETH+target) for altcoins.
        lookback_days: Historical window for training.
        trade_date: Upper date bound (YYYY-mm-dd) for PIT correctness.
            Defaults to today.

    Returns:
        Formatted markdown string suitable for LLM consumption with
        predictions, directional agreement, confidence, and historical
        DirAcc benchmarks.
    """
    if horizons is None:
        horizons = [7, 14]

    cfg = get_config().get("prediction_models", {})
    if lookback_days is None:
        lookback_days = cfg.get("lookback_days", 730)

    pool_coins, pool_label = _select_pool(symbol, pool_coins)
    if symbol not in pool_coins:
        pool_coins = list(pool_coins) + [symbol]
        pool_label += f" (+ {symbol} target)"

    try:
        from scripts.evaluate_models_multi import build_pooled_transformed
    except Exception:
        # Fallback: import via path manipulation if scripts isn't a package
        import sys
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        scripts_dir = project_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from evaluate_models_multi import build_pooled_transformed  # type: ignore

    try:
        pooled = build_pooled_transformed(
            coins=pool_coins,
            horizons=horizons,
            days=lookback_days,
            trade_date=trade_date,
            add_technical=True,
            add_cross_asset=True,
            add_onchain=False,
        )
    except Exception as e:
        logger.exception(f"Failed to build pooled dataset for {symbol}")
        return f"[LightGBM] ERROR: Could not build pooled dataset — {e}"

    if pooled.empty:
        return f"[LightGBM] ERROR: Empty pooled dataset for {symbol}"

    # Encode coin_id as integer (same as walk_forward_pooled)
    coin_ids = sorted(pooled["coin_id"].unique())
    coin_to_int = {c: i for i, c in enumerate(coin_ids)}
    pooled = pooled.copy()
    pooled["coin_int"] = pooled["coin_id"].map(coin_to_int).astype(int)

    if symbol not in coin_ids:
        return f"[LightGBM] ERROR: {symbol} missing from pooled coins: {coin_ids}"

    # Isolate the target coin's most recent row for prediction
    target_rows = pooled[pooled["coin_id"] == symbol].sort_index()
    if target_rows.empty:
        return f"[LightGBM] ERROR: No rows for {symbol} after transform"
    last_row = target_rows.iloc[-1]
    ref_price = float(last_row["prices"])

    # Feature columns: everything except targets, coin_id
    exclude = {"coin_id"}
    for c in pooled.columns:
        if c.startswith("prices_h"):
            exclude.add(c)
    feature_cols = [c for c in pooled.columns if c not in exclude]

    # Train one LGB per horizon
    predictions: dict[int, float] = {}
    for h in horizons:
        target_col = f"prices_h{h}"
        if target_col not in pooled.columns:
            continue
        train = pooled.dropna(subset=[target_col])
        if train.empty:
            continue
        X_tr = train[feature_cols].to_numpy(dtype=np.float32)
        y_tr = train[target_col].to_numpy(dtype=np.float32)
        X_te = last_row[feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)

        scaler = MinMaxScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = _build_lgb()
        model.fit(X_tr_s, y_tr)
        predictions[h] = float(model.predict(X_te_s)[0])

    if not predictions:
        return f"[LightGBM] ERROR: No predictions produced (missing target columns)"

    # Build output report
    lines = [
        f"=== LightGBM Multi-Horizon Prediction for {symbol.upper()} ===",
        f"Pool: {pool_label}",
        f"Reference Price: ${ref_price:,.2f}",
        "",
    ]

    directions: dict[int, int] = {}
    pct_moves: dict[int, float] = {}
    for h in sorted(predictions.keys()):
        pred = predictions[h]
        pct = (pred - ref_price) / ref_price * 100 if ref_price > 0 else 0
        direction = "UP" if pred > ref_price else "DOWN"
        directions[h] = 1 if pred > ref_price else -1
        pct_moves[h] = pct
        dir_acc_note = _format_dir_acc_note(symbol, h)
        lines.append(
            f"h={h:>2d} Prediction: ${pred:,.2f} ({pct:+.2f}%) → {direction}   "
            f"[historical DirAcc: {dir_acc_note}]"
        )

    lines.append("")

    # Consensus assessment
    all_dirs = list(directions.values())
    all_same = all(d == all_dirs[0] for d in all_dirs)
    max_pct = max(abs(p) for p in pct_moves.values())

    if all_same:
        dir_word = "bullish" if all_dirs[0] == 1 else "bearish"
        lines.append(f"Consensus: ALL HORIZONS AGREE ({dir_word})")
        if max_pct >= 2.0:
            confidence = "HIGH"
            lines.append(f"Confidence: HIGH (strong magnitude, ≥2% predicted move at longest horizon)")
        else:
            confidence = "MEDIUM"
            lines.append(f"Confidence: MEDIUM (agreement but small magnitude, <2%)")
    else:
        confidence = "LOW"
        lines.append("Consensus: HORIZONS DISAGREE")
        longest_h = max(horizons)
        if longest_h in directions:
            longest_dir = "bullish" if directions[longest_h] == 1 else "bearish"
            lines.append(
                f"Confidence: LOW — prefer h={longest_h} ({longest_dir}) as the stronger long-term signal"
            )
        else:
            lines.append("Confidence: LOW")

    lines.append("")
    lines.append(
        f"Note: h=14 is the primary signal (historical DirAcc 85% for BTC, 76% for ETH). "
        f"h=1 signals are noise (~50%) and are NOT used."
    )

    return "\n".join(lines)
