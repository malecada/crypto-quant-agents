"""ARIMA forecasting model for cryptocurrency price prediction.

Ported from Krypto-v0/src/models/arima_model.py, adapted to use TradingAgents'
config system and CoinGecko/Binance data vendor.

Note: The original ARIMA model used exogenous features (ETH price, S&P 500,
Google Trends, etc.) that came from Krypto-v0's multi-source scraping pipeline.
In this port, since we only have OHLCV data from the CoinGecko/Binance vendor,
the ARIMA model uses price-derived features as exogenous regressors instead.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tradingagents.dataflows.config import get_config
from tradingagents.models import model_utils as mu
from tradingagents.models.prediction import Prediction

logger = logging.getLogger(__name__)

# Exogenous features for ARIMA -- derived from OHLCV data since the original
# multi-source features (ETH price, S&P 500, etc.) are not available here.
ARIMA_EXOG_FEATURES = [
    "daily_return",
    "high_low_spread",
    "open_close_spread",
    "ma_7",
    "ma_14",
    "vol_7",
    "vol_14",
    "total_volumes",
    "Day",
    "Month",
]


# ── Config helpers ─────────────────────────────────────────────────


def _cfg():
    """Return the prediction_models config dict."""
    return get_config().get("prediction_models", {})


# ── Checkpoint I/O ─────────────────────────────────────────────────


def _checkpoint_dir() -> Path:
    return Path(_cfg().get("checkpoint_dir", "./data/checkpoints/"))


def save_checkpoint(model_fit):
    """Save fitted ARIMA model to disk."""
    ckpt = _checkpoint_dir()
    ckpt.mkdir(parents=True, exist_ok=True)
    model_path = ckpt / "arima_model.pkl"
    model_fit.save(str(model_path))
    logger.info(f"ARIMA checkpoint saved -> {model_path}")


def load_checkpoint():
    """Load ARIMA model from disk.

    Returns:
        Fitted ARIMAResults object.
    """
    from statsmodels.tsa.arima.model import ARIMAResults

    ckpt = _checkpoint_dir()
    model_path = ckpt / "arima_model.pkl"
    model_fit = ARIMAResults.load(str(model_path))
    logger.info(f"ARIMA checkpoint loaded <- {model_path}")
    return model_fit


def _checkpoint_exists() -> bool:
    ckpt = _checkpoint_dir()
    return (ckpt / "arima_model.pkl").exists()


# ── Core building blocks ───────────────────────────────────────────


def prepare_data(df_all, include_future_row=True):
    """Run data_transform and build the ARIMA-specific DataFrame.

    Returns:
        (df_with_date, reframed_lags, df_final, first_day_future)
        where *df_with_date* contains only the target + ARIMA exogenous
        features, date-indexed and NaN-dropped.
    """
    first_day_future = pd.to_datetime(datetime.now() + timedelta(days=1))
    reframed_lags, df_final = mu.data_transform(
        df_all, first_day_future, include_future_row=include_future_row,
    )

    target_col = "prices"
    feature_cols = [c for c in ARIMA_EXOG_FEATURES if c in reframed_lags.columns]
    df = reframed_lags[[target_col] + feature_cols].copy()

    # Reconstruct date index from Year/Month/Day columns
    if all(c in reframed_lags.columns for c in ["Year", "Month", "Day"]):
        date = pd.to_datetime(dict(
            year=reframed_lags["Year"],
            month=reframed_lags["Month"],
            day=reframed_lags["Day"],
        ))
        df_with_date = pd.concat([date, df], axis=1)
        df_with_date.columns = np.append("date", df.columns)
        df_with_date.set_index("date", inplace=True)
    else:
        # Fallback: use a simple positional index
        df_with_date = df.copy()

    df_with_date = df_with_date.dropna()

    # Normalize date index to daily timestamps
    if hasattr(df_with_date.index, 'to_period'):
        try:
            df_with_date.index = (
                pd.DatetimeIndex(df_with_date.index)
                .to_period("D")
                .to_timestamp("D")
            )
        except Exception:
            pass  # Keep existing index if conversion fails

    return df_with_date, reframed_lags, df_final, first_day_future


def train_and_predict(train_df, test_exog):
    """Fit ARIMA on *train_df* and forecast one step using *test_exog*.

    Returns:
        (forecast_value, model_fit)
    """
    cfg = _cfg()
    arima_order = tuple(cfg.get("arima_order", [2, 1, 2]))
    max_iter = cfg.get("arima_max_iter", 500)

    target_col = "prices"
    model = ARIMA(
        train_df[target_col],
        exog=train_df.drop(columns=[target_col]),
        order=arima_order,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
        model_fit = model.fit(method_kwargs={"maxiter": max_iter})
    forecast = model_fit.forecast(steps=1, exog=test_exog)
    return float(forecast.values[0]), model_fit


def predict_with_confidence(model_fit, exog, alpha=None):
    """Return point prediction with native ARIMA confidence interval.

    Returns:
        (prediction, lower, upper)
    """
    if alpha is None:
        alpha = _cfg().get("prediction_interval_alpha", 0.05)
    fc = model_fit.get_forecast(steps=1, exog=exog)
    summary = fc.summary_frame(alpha=alpha)
    prediction = float(summary["mean"].values[0])
    lower = float(summary["mean_ci_lower"].values[0])
    upper = float(summary["mean_ci_upper"].values[0])
    return prediction, lower, upper


def _forecast_from_df(df_all, save_checkpoint_flag=False):
    """Retrain on all available data and forecast one step ahead.

    Returns:
        Prediction object with confidence interval.
    """
    df_with_date, reframed_lags, df_final, first_day_future = prepare_data(df_all)

    target_col = "prices"
    df_past = df_with_date.iloc[:-1, :]
    df_future = df_with_date.iloc[-1:, :]

    _, model_fit = train_and_predict(
        df_past, df_future.drop(columns=target_col),
    )
    if save_checkpoint_flag:
        save_checkpoint(model_fit)

    pred, lower, upper = predict_with_confidence(
        model_fit, df_future.drop(columns=target_col),
    )

    feature_cols = [c for c in ARIMA_EXOG_FEATURES if c in df_with_date.columns]
    return Prediction(
        value=pred,
        lower=lower,
        upper=upper,
        model_name="ARIMA",
        timestamp=first_day_future.to_pydatetime(),
        features_used=feature_cols,
    )


# ── Public forecast_next (called by prediction analyst tools) ──────


def forecast_next(
    symbol: str,
    lookback_days: Optional[int] = None,
    trade_date: Optional[str] = None,
) -> str:
    """Fetch data, train ARIMA model, and return a formatted prediction string.

    This is the entry point called by the prediction analyst tools.

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
                f"[ARIMA] ERROR: No price data available for '{symbol}'. "
                f"Could not fetch OHLCV data from CoinGecko/Binance."
            )

        # Get current price for direction/percentage calculation
        current_price = float(df_model["prices"].iloc[-1])

        # Run prediction
        prediction = _forecast_from_df(df_model, save_checkpoint_flag=True)

        # Calculate direction and percentage change
        pct_change = ((prediction.value - current_price) / current_price) * 100
        direction = "UP" if pct_change > 0 else "DOWN"

        arima_order = tuple(cfg.get("arima_order", [2, 1, 2]))

        # Format output for the LLM analyst
        lines = [
            f"=== ARIMA Price Prediction for {symbol.upper()} ===",
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
            f"  ARIMA Order: {arima_order}",
            f"  Training samples: {lookback_days} days of historical data",
            f"  Exogenous features: {', '.join(prediction.features_used)}",
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

        return "\n".join(lines)

    except Exception as e:
        logger.exception(f"ARIMA forecast failed for {symbol}")
        return (
            f"[ARIMA] ERROR: Forecast failed for '{symbol}'. "
            f"Reason: {e}"
        )


# ── Walk-forward evaluation (retained for offline evaluation) ──────


def walk_forward_evaluate(df_with_date, min_train_window=None):
    """Rolling-window evaluation. Returns (predictions, actuals, window_start).

    Each iteration trains on data[:end_train] and predicts data[end_train],
    guaranteeing no future information leaks into training.
    """
    cfg = _cfg()
    arima_order = tuple(cfg.get("arima_order", [2, 1, 2]))
    max_iter = cfg.get("arima_max_iter", 500)

    target_col = "prices"

    if min_train_window is not None:
        window_start = min(min_train_window, len(df_with_date) - 1)
    elif len(df_with_date) > 500:
        window_start = int(len(df_with_date) * 0.9)
    elif len(df_with_date) > 200:
        window_start = int(len(df_with_date) * 0.8)
    else:
        window_start = int(len(df_with_date) * 0.7)

    predictions = []
    actuals = []

    for end_train in range(window_start, len(df_with_date)):
        train_slice = df_with_date.iloc[:end_train]
        test_slice = df_with_date.iloc[end_train: end_train + 1]

        arima_model = ARIMA(
            train_slice[target_col],
            exog=train_slice.drop(columns=[target_col]),
            order=arima_order,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
            arima_fit = arima_model.fit(
                method_kwargs={"maxiter": max_iter},
            )
        pred = arima_fit.forecast(
            steps=1, exog=test_slice.drop(columns=target_col),
        )
        predictions.append(pred.values[0])
        actuals.append(test_slice[target_col].values[0])

    return predictions, actuals, window_start


def model_run(
    df_all: pd.DataFrame,
    min_train_window: int | None = None,
    save_checkpoint_flag: bool = False,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Orchestrate prepare -> walk-forward evaluate -> metrics for ARIMA.

    Returns:
        (df_forecast, metrics, result_df) — same contract as rf_model.model_run.
    """
    df_with_date, reframed_lags, df_final, first_day_future = prepare_data(df_all)

    predictions, actuals, window_start = walk_forward_evaluate(
        df_with_date, min_train_window=min_train_window,
    )

    metrics = mu.compute_metrics(actuals, predictions)

    eval_dates = df_with_date.index[window_start: window_start + len(predictions)]
    result_df = pd.DataFrame(
        {"prediction": predictions, "actual": actuals},
        index=eval_dates,
    )
    result_df.index.name = "date"

    prediction_obj = _forecast_from_df(df_all, save_checkpoint_flag=save_checkpoint_flag)
    df_forecast = df_final.copy()
    df_forecast.loc[df_forecast.index[-1], "prices"] = prediction_obj.value

    return df_forecast, metrics, result_df


# ── Multi-horizon walk-forward (for h > 1) ─────────────────────────


def walk_forward_horizon(df_with_date, horizon: int, min_train_window=None):
    """Rolling-window evaluation with h-step-ahead forecasts.

    Same mechanics as walk_forward_evaluate() but uses ARIMA's native
    forecast(steps=horizon). Future exogenous variables are held flat
    (copied from the last known training row) since we don't have true
    future values at prediction time.

    Args:
        df_with_date: Date-indexed DataFrame with `prices` plus ARIMA exog columns.
        horizon: Forecast horizon in bars (1 = next bar, 7 = next week, etc.).
        min_train_window: Optional override for where walk-forward begins.

    Returns:
        (predictions, actuals, window_start) — the prediction at step i is
        the h-step-ahead forecast and `actuals` holds the realized price at
        step i+h-1.
    """
    cfg = _cfg()
    arima_order = tuple(cfg.get("arima_order", [2, 1, 2]))
    max_iter = cfg.get("arima_max_iter", 500)

    target_col = "prices"

    if min_train_window is not None:
        window_start = min(min_train_window, len(df_with_date) - horizon - 1)
    elif len(df_with_date) > 500:
        window_start = int(len(df_with_date) * 0.9)
    elif len(df_with_date) > 200:
        window_start = int(len(df_with_date) * 0.8)
    else:
        window_start = int(len(df_with_date) * 0.7)

    predictions: list[float] = []
    actuals: list[float] = []

    for end_train in range(window_start, len(df_with_date) - horizon):
        train_slice = df_with_date.iloc[:end_train]
        # h-step-ahead actual is at index end_train + horizon - 1
        actual_h = df_with_date.iloc[end_train + horizon - 1][target_col]

        exog_train = train_slice.drop(columns=[target_col])
        # Hold exog flat for future steps (last known row, repeated)
        last_exog = exog_train.iloc[-1:].copy()
        exog_future = pd.concat([last_exog] * horizon, ignore_index=True)
        exog_future.index = df_with_date.index[end_train : end_train + horizon]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
                arima_fit = ARIMA(
                    train_slice[target_col],
                    exog=exog_train,
                    order=arima_order,
                ).fit(method_kwargs={"maxiter": max_iter})
            forecast = arima_fit.forecast(steps=horizon, exog=exog_future)
            predictions.append(float(forecast.iloc[-1]))
            actuals.append(float(actual_h))
        except Exception as e:
            logger.debug(f"ARIMA h={horizon} failed at step {end_train}: {e}")
            continue

    return predictions, actuals, window_start
