"""On-chain feature model for cryptocurrency price prediction.

A GradientBoosting model trained on on-chain and derivatives features.
Purpose: isolate the predictive contribution of on-chain data and provide
a distinct signal for the agent pipeline.

Ported from Krypto-v0/src/models/onchain_model.py, adapted to use
TradingAgents' config system and CoinGecko/Binance data vendor.

Note: The original model used on-chain features (funding rate, TVL,
stablecoin mcap, gas prices) from Krypto-v0's dedicated on-chain scraper.
In this port, if on-chain features are not present in the data, the model
falls back to price-derived volatility/volume features as proxies.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

from tradingagents.dataflows.config import get_config
from tradingagents.models import model_utils as mu
from tradingagents.models.prediction import Prediction

logger = logging.getLogger(__name__)

# Primary on-chain features (if available from on-chain data vendor)
ONCHAIN_FEATURE_COLS = [
    "funding_rate",
    "total_tvl",
    "stablecoin_mcap",
    "avg_gas_price",
    "bsc_avg_gas_price",
]

# Fallback features derived from OHLCV data when on-chain features
# are not available
FALLBACK_FEATURE_COLS = [
    "total_volumes",
    "vol_ma_7",
    "vol_ma_30",
    "high_low_spread",
    "open_close_spread",
    "vol_7",
    "vol_14",
    "vol_30",
    "daily_return",
]


# ── Config helpers ─────────────────────────────────────────────────


def _cfg():
    """Return the prediction_models config dict."""
    return get_config().get("prediction_models", {})


# ── Model construction ─────────────────────────────────────────────


def _build_model(**overrides):
    """Instantiate a GradientBoostingRegressor from config."""
    cfg = _cfg()
    params = dict(
        n_estimators=cfg.get("onchain_n_estimators", 500),
        max_depth=cfg.get("onchain_max_depth", 5),
        learning_rate=cfg.get("onchain_learning_rate", 0.1),
        random_state=42,
    )
    params.update(overrides)
    return GradientBoostingRegressor(**params)


# ── Checkpoint I/O ─────────────────────────────────────────────────


def _checkpoint_dir() -> Path:
    return Path(_cfg().get("checkpoint_dir", "./data/checkpoints/"))


def save_checkpoint(model, scaler):
    """Save trained on-chain model and scaler to disk."""
    ckpt = _checkpoint_dir()
    ckpt.mkdir(parents=True, exist_ok=True)
    model_path = ckpt / "onchain_model.joblib"
    scaler_path = ckpt / "onchain_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"OnChain checkpoint saved -> {model_path}")


def load_checkpoint():
    """Load on-chain model and scaler from disk."""
    ckpt = _checkpoint_dir()
    model_path = ckpt / "onchain_model.joblib"
    scaler_path = ckpt / "onchain_scaler.joblib"
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logger.info(f"OnChain checkpoint loaded <- {model_path}")
    return model, scaler


def _checkpoint_exists() -> bool:
    ckpt = _checkpoint_dir()
    return (ckpt / "onchain_model.joblib").exists() and (ckpt / "onchain_scaler.joblib").exists()


# ── Core building blocks ───────────────────────────────────────────


def _select_features(reframed_lags: pd.DataFrame) -> list[str]:
    """Return on-chain features present in the DataFrame, falling back to OHLCV-derived features."""
    # Try primary on-chain features first
    onchain = [c for c in ONCHAIN_FEATURE_COLS if c in reframed_lags.columns]
    if onchain:
        return onchain

    # Fallback to price-derived features
    fallback = [c for c in FALLBACK_FEATURE_COLS if c in reframed_lags.columns]
    if fallback:
        logger.info(
            f"No on-chain features found; using {len(fallback)} OHLCV-derived fallback features"
        )
    return fallback


def prepare_data(df_all, include_future_row=True):
    """Run data_transform and return features, target, and metadata."""
    first_day_future = pd.to_datetime(datetime.now() + timedelta(days=1))
    reframed_lags, df_final = mu.data_transform(
        df_all, first_day_future, include_future_row=include_future_row,
    )
    return reframed_lags, df_final, first_day_future


def train_and_predict(X_train, y_train, X_test, scaler=None):
    """Fit one GradientBoosting model and return predictions on X_test.

    Returns:
        (predictions, model, scaler)
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train_s = scaler.fit_transform(X_train.astype("float32"))
    else:
        X_train_s = scaler.transform(X_train.astype("float32"))
    X_test_s = scaler.transform(X_test.astype("float32"))

    model = _build_model()
    model.fit(X_train_s, y_train.astype("float32"))
    preds = model.predict(X_test_s)
    return preds, model, scaler


def predict_with_confidence(model, scaler, X_test, alpha=None):
    """Return point prediction with approximate confidence interval.

    Uses staged_predict to estimate uncertainty from the learning curve.
    Falls back to a +/-5% heuristic interval if staged prediction is
    unavailable.
    """
    if alpha is None:
        alpha = _cfg().get("prediction_interval_alpha", 0.05)

    X_test_s = scaler.transform(X_test.astype("float32"))
    prediction = model.predict(X_test_s)

    # Use staged predictions to estimate variance
    staged = np.array(list(model.staged_predict(X_test_s)))  # (n_estimators, n_samples)
    if staged.shape[0] > 10:
        # Use last 20% of stages to estimate prediction stability
        tail = staged[int(staged.shape[0] * 0.8):]
        std = np.std(tail, axis=0)
        from scipy import stats
        z = stats.norm.ppf(1 - alpha / 2)
        lower = prediction - z * std
        upper = prediction + z * std
    else:
        # Fallback: 5% symmetric interval
        lower = prediction * 0.95
        upper = prediction * 1.05

    return prediction, lower, upper


# ── Forecast ───────────────────────────────────────────────────────


def _forecast_from_df(df_all, save_checkpoint_flag=False):
    """Retrain on all available data and forecast one step ahead.

    Uses on-chain features (or OHLCV-derived fallback features).
    Returns a Prediction object compatible with the RF/ARIMA interface.
    """
    reframed_lags, df_final, first_day_future = prepare_data(df_all)

    feature_cols = _select_features(reframed_lags)
    if not feature_cols:
        raise ValueError(
            "No usable features found in the data. "
            "Ensure on-chain data is available or OHLCV data contains volume/spread columns."
        )

    target_col = "prices"
    X_all = reframed_lags[feature_cols]
    y_all = reframed_lags[target_col].values.astype("float32")

    # Training rows: all except the last (future) row
    X_train_vals = X_all.values[:-1].astype("float32")
    y_train = y_all[:-1]
    valid = np.isfinite(y_train) & np.all(np.isfinite(X_train_vals), axis=1)
    X_train_vals = X_train_vals[valid]
    y_train = y_train[valid]

    if len(X_train_vals) == 0:
        raise ValueError(
            "No valid training rows after dropping NaN in on-chain/fallback features."
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_vals)

    model = _build_model()
    model.fit(scaler.transform(X_train_vals), y_train)

    if save_checkpoint_flag:
        save_checkpoint(model, scaler)

    pred, lower, upper = predict_with_confidence(
        model, scaler, X_all.values[-1:],
    )

    return Prediction(
        value=float(pred[0]),
        lower=float(lower[0]),
        upper=float(upper[0]),
        model_name="OnChainGBR",
        timestamp=first_day_future.to_pydatetime(),
        features_used=feature_cols,
    )


# ── On-chain data enrichment ──────────────────────────────────────


def _enrich_with_onchain_data(df_model: pd.DataFrame, symbol: str, lookback_days: int) -> pd.DataFrame:
    """Fetch on-chain features and merge them into the model DataFrame.

    Fetches funding rates, TVL, and stablecoin market cap, then joins
    them by date so the GBR model can use real on-chain signals instead
    of falling back to OHLCV-derived proxies.
    """
    from tradingagents.dataflows.onchain import (
        _scrape_funding_rates,
        _scrape_total_tvl,
        _scrape_stablecoin_mcap_history,
    )
    from tradingagents.dataflows.onchain import _resolve_futures_symbol

    # Determine date range from the model DataFrame
    if df_model.index.dtype == "object":
        dates = pd.to_datetime(df_model.index)
    else:
        dates = df_model.index
    past = dates.min().date()
    today = dates.max().date()

    onchain_frames = []

    # 1. Funding rates (daily average)
    try:
        futures_symbol = _resolve_futures_symbol(symbol)
        df_funding = _scrape_funding_rates(past, today, futures_symbol)
        if not df_funding.empty:
            onchain_frames.append(df_funding)
            logger.info(f"Fetched {len(df_funding)} days of funding rate data")
    except Exception as e:
        logger.warning(f"Could not fetch funding rates for on-chain model: {e}")

    # 2. Total DeFi TVL
    try:
        df_tvl = _scrape_total_tvl(past, today)
        if not df_tvl.empty:
            onchain_frames.append(df_tvl)
            logger.info(f"Fetched {len(df_tvl)} days of TVL data")
    except Exception as e:
        logger.warning(f"Could not fetch TVL for on-chain model: {e}")

    # 3. Stablecoin market cap (historical)
    try:
        df_stable = _scrape_stablecoin_mcap_history(past, today)
        if not df_stable.empty:
            onchain_frames.append(df_stable)
            logger.info(f"Fetched {len(df_stable)} days of stablecoin mcap data")
    except Exception as e:
        logger.warning(f"Could not fetch stablecoin mcap for on-chain model: {e}")

    if not onchain_frames:
        return df_model

    # Merge on-chain DataFrames together
    from functools import reduce
    df_onchain = reduce(
        lambda left, right: left.join(right, how="outer"),
        onchain_frames,
    )

    # Align index format: model df may be datetime, on-chain is string dates
    df_model = df_model.copy()
    model_date_strings = pd.to_datetime(df_model.index).strftime("%Y-%m-%d")
    df_model.index = model_date_strings

    # Join and forward-fill (on-chain data may have gaps on weekends)
    df_model = df_model.join(df_onchain, how="left")
    for col in df_onchain.columns:
        if col in df_model.columns:
            df_model[col] = df_model[col].ffill().fillna(0)

    logger.info(
        f"On-chain enrichment: added columns {list(df_onchain.columns)}, "
        f"{df_model[list(df_onchain.columns)].notna().all(axis=1).sum()}/{len(df_model)} rows populated"
    )

    return df_model


# ── Public forecast_next (called by prediction analyst tools) ──────


def forecast_next(
    symbol: str,
    lookback_days: Optional[int] = None,
    trade_date: Optional[str] = None,
) -> str:
    """Fetch data, train Gradient Boosting model, and return a formatted prediction string.

    This is the entry point called by the prediction analyst tools.
    Uses on-chain features when available, otherwise falls back to
    OHLCV-derived volume and volatility features.

    Args:
        symbol: CoinGecko ID (e.g. "bitcoin", "ethereum").
        lookback_days: Number of days of historical data. Defaults to config value.
        trade_date: Upper date boundary (YYYY-mm-dd) for backtesting.
            Prevents look-ahead bias by ensuring the model only sees
            data up to this date. Defaults to today for live usage.

    Returns:
        A formatted string with the prediction, confidence interval,
        direction, and relevant metrics -- suitable for LLM consumption.
    """
    try:
        cfg = _cfg()
        if lookback_days is None:
            lookback_days = cfg.get("lookback_days", 300)

        # Fetch OHLCV data via the data vendor
        df_model = mu.fetch_ohlcv_for_model(symbol, lookback_days, trade_date=trade_date)
        if df_model.empty:
            return (
                f"[OnChainGBR] ERROR: No price data available for '{symbol}'. "
                f"Could not fetch OHLCV data from CoinGecko/Binance."
            )

        # Enrich with actual on-chain features (funding rates, TVL, stablecoin mcap)
        df_model = _enrich_with_onchain_data(df_model, symbol, lookback_days)

        # Get current price for direction/percentage calculation
        current_price = float(df_model["prices"].iloc[-1])

        # Run prediction
        prediction = _forecast_from_df(df_model, save_checkpoint_flag=True)

        # Calculate direction and percentage change
        pct_change = ((prediction.value - current_price) / current_price) * 100
        direction = "UP" if pct_change > 0 else "DOWN"

        # Determine feature source label
        has_onchain = any(f in ONCHAIN_FEATURE_COLS for f in prediction.features_used)
        feature_source = "on-chain data" if has_onchain else "OHLCV-derived volume/volatility features"

        # Format output for the LLM analyst
        lines = [
            f"=== On-Chain Gradient Boosting Prediction for {symbol.upper()} ===",
            f"",
            f"Current Price: ${current_price:,.2f}",
            f"Predicted Price (next day): ${prediction.value:,.2f}",
            f"Direction: {direction} ({pct_change:+.2f}%)",
            f"",
            f"95% Confidence Interval:",
            f"  Lower Bound: ${prediction.lower:,.2f}",
            f"  Upper Bound: ${prediction.upper:,.2f}",
            f"  Interval Width: ${prediction.interval_width:,.2f}",
            f"",
            f"Model Details:",
            f"  Feature source: {feature_source}",
            f"  Features used: {', '.join(prediction.features_used)}",
            f"  Training samples: {lookback_days} days of historical data",
            f"  N estimators: {cfg.get('onchain_n_estimators', 500)}",
            f"  Max depth: {cfg.get('onchain_max_depth', 5)}",
            f"  Learning rate: {cfg.get('onchain_learning_rate', 0.1)}",
            f"  Forecast date: {prediction.timestamp.strftime('%Y-%m-%d')}",
        ]

        # Add confidence assessment
        interval_pct = (prediction.interval_width / current_price) * 100 if current_price > 0 else 0
        if interval_pct < 3:
            confidence_note = "High confidence (narrow prediction interval)"
        elif interval_pct < 8:
            confidence_note = "Moderate confidence"
        else:
            confidence_note = "Low confidence (wide prediction interval)"
        lines.append(f"  Confidence: {confidence_note} (interval is {interval_pct:.1f}% of price)")

        if not has_onchain:
            lines.append("")
            lines.append(
                "Note: On-chain features (funding rate, TVL, stablecoin mcap, gas prices) "
                "were not available. Model used OHLCV-derived features as proxies."
            )

        return "\n".join(lines)

    except Exception as e:
        logger.exception(f"OnChainGBR forecast failed for {symbol}")
        return (
            f"[OnChainGBR] ERROR: Forecast failed for '{symbol}'. "
            f"Reason: {e}"
        )


# ── Walk-forward evaluation (retained for offline evaluation) ──────


def walk_forward_evaluate(reframed_lags, min_train_window=None):
    """Rolling-window evaluation using on-chain/fallback features.

    Returns (predictions, actuals, window_start).
    """
    target_col = "prices"
    feature_cols = _select_features(reframed_lags)
    if not feature_cols:
        return [], [], 0

    if min_train_window is not None:
        window_start = min(min_train_window, len(reframed_lags) - 1)
    elif len(reframed_lags) > 500:
        window_start = int(len(reframed_lags) * 0.9)
    elif len(reframed_lags) > 200:
        window_start = int(len(reframed_lags) * 0.8)
    else:
        window_start = int(len(reframed_lags) * 0.7)

    predictions = []
    actuals = []

    for end_train in range(window_start, len(reframed_lags)):
        train_slice = reframed_lags.iloc[:end_train]
        test_slice = reframed_lags.iloc[end_train: end_train + 1]

        X_tr = train_slice[feature_cols].values.astype("float32")
        y_tr = train_slice[target_col].values.astype("float32")
        X_te = test_slice[feature_cols].values.astype("float32")
        y_te = test_slice[target_col].values.astype("float32")

        sc = MinMaxScaler(feature_range=(0, 1))
        X_tr_s = sc.fit_transform(X_tr)
        X_te_s = sc.transform(X_te)

        model = _build_model()
        model.fit(X_tr_s, y_tr)
        predictions.append(model.predict(X_te_s)[0])
        actuals.append(y_te[0])

    return predictions, actuals, window_start


def model_run(
    df_all: pd.DataFrame,
    min_train_window: int | None = None,
    save_checkpoint_flag: bool = False,
    symbol: str = "bitcoin",
    lookback_days: int = 300,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Orchestrate prepare -> walk-forward evaluate -> metrics for GBR.

    Optionally enriches with on-chain data before evaluation.

    Returns:
        (df_forecast, metrics, result_df) — same contract as rf_model.model_run.
    """
    # Enrich with on-chain features (degrades gracefully if unavailable)
    df_enriched = _enrich_with_onchain_data(df_all.copy(), symbol, lookback_days)

    reframed_lags, df_final, first_day_future = prepare_data(df_enriched)

    predictions, actuals, window_start = walk_forward_evaluate(
        reframed_lags, min_train_window=min_train_window,
    )

    metrics = mu.compute_metrics(actuals, predictions)

    eval_dates = df_final.index[window_start: window_start + len(predictions)]
    result_df = pd.DataFrame(
        {"prediction": predictions, "actual": actuals},
        index=eval_dates,
    )
    result_df.index.name = "date"

    prediction_obj = _forecast_from_df(df_enriched, save_checkpoint_flag=save_checkpoint_flag)
    df_forecast = df_final.copy()
    df_forecast.loc[df_forecast.index[-1], "prices"] = prediction_obj.value

    return df_forecast, metrics, result_df
