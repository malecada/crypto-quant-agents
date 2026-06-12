"""Backtest runners for individual models and full multi-agent system."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.backtesting.engine import BacktestResult, run_backtest
from tradingagents.backtesting.strategies import (
    FiveLevelSignal,
    ModelConsensus,
    ThresholdSignal,
)
from tradingagents.models import model_utils as mu

logger = logging.getLogger(__name__)


@dataclass
class ModelEvalResult:
    """Result container for a single model's walk-forward evaluation."""

    model_name: str
    metrics: dict
    result_df: pd.DataFrame
    forecast_df: Optional[pd.DataFrame] = None


def evaluate_models(
    coin: str,
    lookback_days: int = 730,
    trade_date: Optional[str] = None,
    min_train_window: Optional[int] = None,
    models: list[str] | tuple[str, ...] = ("rf", "arima"),
    output_dir: Path | str = Path("data"),
) -> dict[str, ModelEvalResult]:
    """Run walk-forward evaluation for specified prediction models.

    Args:
        coin: CoinGecko ID (e.g. "bitcoin").
        lookback_days: Historical data window.
        trade_date: Upper date bound (YYYY-MM-DD). None = today.
        min_train_window: Min training rows for walk-forward.
        models: Which models to evaluate ("rf", "arima", "gbr").
        output_dir: Where to save predictions CSV.

    Returns:
        Dict mapping model key to ModelEvalResult.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_model = mu.fetch_ohlcv_for_model(coin, lookback_days, trade_date=trade_date)
    if df_model.empty:
        logger.error(f"No OHLCV data for {coin}")
        return {}

    logger.info(f"Fetched {len(df_model)} rows for {coin} (lookback={lookback_days})")

    results: dict[str, ModelEvalResult] = {}

    if "rf" in models:
        logger.info("Evaluating Random Forest...")
        from tradingagents.models.rf_model import model_run as rf_run

        df_fc, metrics, result_df = rf_run(df_model, min_train_window=min_train_window)
        results["rf"] = ModelEvalResult("RandomForest", metrics, result_df, df_fc)
        logger.info(f"RF: R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}")

    if "arima" in models:
        logger.info("Evaluating ARIMA...")
        from tradingagents.models.arima_model import model_run as arima_run

        df_fc, metrics, result_df = arima_run(df_model, min_train_window=min_train_window)
        results["arima"] = ModelEvalResult("ARIMA", metrics, result_df, df_fc)
        logger.info(f"ARIMA: R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}")

    if "gbr" in models:
        logger.info("Evaluating On-Chain GBR...")
        from tradingagents.models.onchain_model import model_run as gbr_run

        df_fc, metrics, result_df = gbr_run(
            df_model, min_train_window=min_train_window,
            symbol=coin, lookback_days=lookback_days,
        )
        results["gbr"] = ModelEvalResult("OnChainGBR", metrics, result_df, df_fc)
        logger.info(f"GBR: R²={metrics['r2']:.4f}  MAE={metrics['mae']:.2f}")

    # Save merged predictions CSV
    if results:
        frames = []
        for key, res in results.items():
            frame = res.result_df.rename(columns={
                "prediction": f"{key}_prediction",
                "actual": f"{key}_actual",
            })
            if hasattr(frame.index, "tz") and frame.index.tz is not None:
                frame.index = frame.index.tz_localize(None)
            frames.append(frame)

        df_out = frames[0]
        for f in frames[1:]:
            df_out = df_out.join(f, how="outer")

        csv_path = output_dir / "eval_predictions.csv"
        df_out.to_csv(csv_path)
        logger.info(f"Predictions saved -> {csv_path}")

    return results


def generate_system_signals(
    coin: str,
    start_date: str,
    end_date: str,
    config: dict,
    selected_analysts: list[str] | None = None,
    signals_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate agent signals for each date in range via propagate().

    If signals_csv exists and covers the requested range, loads from
    disk (saves $10-50 in LLM costs).

    Returns:
        DataFrame with columns: date, signal.
    """
    # Check for cached signals
    if signals_csv and Path(signals_csv).exists():
        df_cached = pd.read_csv(signals_csv, parse_dates=["date"])
        cached_start = df_cached["date"].min().strftime("%Y-%m-%d")
        cached_end = df_cached["date"].max().strftime("%Y-%m-%d")
        if cached_start <= start_date and cached_end >= end_date:
            logger.info(f"Loaded cached signals from {signals_csv}")
            mask = (df_cached["date"] >= start_date) & (df_cached["date"] <= end_date)
            return df_cached[mask].reset_index(drop=True)
        logger.info(f"Cache {signals_csv} doesn't cover requested range, regenerating")

    # Force replay cache for determinism and cost savings
    config = config.copy()
    config["replay_cache"] = True

    from tradingagents.graph.trading_graph import TradingAgentsGraph

    analysts = selected_analysts or config.get(
        "selected_analysts", ["market", "onchain", "prediction"]
    )

    ta = TradingAgentsGraph(
        selected_analysts=analysts,
        debug=False,
        config=config,
    )

    # Generate daily dates (calendar days — crypto trades 24/7)
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    records = []
    for i, dt in enumerate(dates):
        date_str = dt.strftime("%Y-%m-%d")
        logger.info(f"[{i + 1}/{len(dates)}] Propagating {coin} @ {date_str}")
        try:
            _, signal = ta.propagate(coin, date_str)
            records.append({"date": dt, "signal": signal})
        except Exception as e:
            logger.error(f"propagate() failed for {date_str}: {e}")
            records.append({"date": dt, "signal": "HOLD"})

    df_signals = pd.DataFrame(records)

    # Save for reuse
    if signals_csv is None:
        signals_csv = Path("data") / f"system_signals_{coin}.csv"
    Path(signals_csv).parent.mkdir(parents=True, exist_ok=True)
    df_signals.to_csv(signals_csv, index=False)
    logger.info(f"Signals saved -> {signals_csv}")

    return df_signals


def run_system_backtest(
    coin: str,
    start_date: str,
    end_date: str,
    config: dict,
    selected_analysts: list[str] | None = None,
    signals_csv: Optional[Path] = None,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0005,
    short_cost: float = 0.0003,
    output_dir: Path | str = Path("data"),
) -> list[BacktestResult]:
    """Run full system backtest: generate signals, then evaluate strategies.

    Returns:
        List of BacktestResult (one per strategy).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Get signals
    df_signals = generate_system_signals(
        coin, start_date, end_date, config,
        selected_analysts=selected_analysts,
        signals_csv=signals_csv,
    )

    # Step 2: Get price data for the same range
    lookback = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 30
    df_prices = mu.fetch_ohlcv_for_model(coin, lookback, trade_date=end_date)
    if df_prices.empty:
        logger.error("No price data available")
        return []

    # Build price series aligned to signal dates
    price_df = pd.DataFrame({
        "date": df_prices.index,
        "close": df_prices["prices"].values,
    })
    price_df["date"] = pd.to_datetime(price_df["date"])

    df_signals["date"] = pd.to_datetime(df_signals["date"])
    merged = pd.merge(df_signals, price_df, on="date", how="inner").sort_values("date")

    if len(merged) < 2:
        logger.error(f"Only {len(merged)} dates after merge — need at least 2")
        return []

    dates = merged["date"]
    actuals = merged["close"].values
    signals = merged["signal"].tolist()

    # Step 3: Run each strategy
    strategies = [
        FiveLevelSignal(),
        ThresholdSignal(),
    ]

    results = []
    for strategy in strategies:
        result = run_backtest(
            dates=dates,
            actuals=actuals,
            agent_signals=signals,
            strategy=strategy,
            ticker=coin,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage=slippage,
            short_cost=short_cost,
        )
        results.append(result)
        logger.info(
            f"{strategy.name}: return={result.metrics['total_return']:+.2%}  "
            f"sharpe={result.metrics['sharpe_ratio']:.2f}"
        )

    return results


def generate_system_signals_v2(
    coins: list[str],
    start_date: str,
    end_date: str,
    config: dict,
    selected_analysts: list[str] | None = None,
    output_dir: Path | str = Path("data/agent_signals"),
    force_rerun: bool = False,
) -> dict[str, pd.DataFrame]:
    """Generate per-coin agent signals with confidence over a date range.

    One CSV per coin at {output_dir}/{coin}_{start}_{end}.csv with columns:
    date, signal, confidence, trader_text.

    Loads from cache when the CSV exists and covers the requested range
    (unless force_rerun=True). Cache granularity is per coin, so adding a
    new coin only generates signals for that coin.

    Returns a dict mapping coin -> DataFrame.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Force the LLM replay cache — mandatory for determinism and cost control.
    config = config.copy()
    config["replay_cache"] = True

    from tradingagents.graph.trading_graph import TradingAgentsGraph
    analysts = selected_analysts or config.get(
        "selected_analysts", ["market", "onchain", "prediction"],
    )

    ta = TradingAgentsGraph(
        selected_analysts=analysts, debug=False, config=config,
    )

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    results: dict[str, pd.DataFrame] = {}

    for coin in coins:
        csv_path = output_dir / f"{coin}_{start_date}_{end_date}.csv"

        # Try cache first.
        cached_records: list[dict] = []
        have_dates: set[str] = set()
        if csv_path.exists() and not force_rerun:
            cached = pd.read_csv(csv_path, parse_dates=["date"])
            # Drop ERROR rows so they get refilled on resume.
            mask_err = cached["trader_text"].astype(str).str.startswith("ERROR:")
            n_err = int(mask_err.sum())
            if n_err:
                logger.info(f"{coin}: dropping {n_err} ERROR rows for refill")
                cached = cached[~mask_err].copy()

            cached_records = cached.to_dict(orient="records")
            have_dates = {pd.Timestamp(d).strftime("%Y-%m-%d")
                          for d in cached["date"].tolist()}

            if len(have_dates) >= len(dates):
                logger.info(f"{coin}: loaded {len(cached_records)} cached signals from {csv_path}")
                results[coin] = cached
                continue
            logger.info(
                f"{coin}: resuming from partial cache with {len(cached_records)} good rows "
                f"({len(dates) - len(have_dates)} dates to (re)generate)"
            )

        missing_dates = [dt for dt in dates
                         if dt.strftime("%Y-%m-%d") not in have_dates]
        logger.info(f"{coin}: generating signals for {len(missing_dates)} dates")
        records = list(cached_records)

        # Per-row retry settings — survives transient OpenAI / connection hiccups.
        import time as _time
        import concurrent.futures as _cf
        max_row_attempts = int(config.get("propagate_max_attempts", 4))
        base_backoff = float(config.get("propagate_backoff_seconds", 10.0))
        row_timeout = float(config.get("propagate_row_timeout", 600.0))  # 10 min hard cap

        def _propagate_with_timeout(_coin: str, _date: str):
            """Run ta.propagate_with_confidence with a hard wall-clock timeout.

            A hung HTTP socket inside the LangGraph tool call would otherwise
            leave the row indefinitely stuck. Using a daemon thread + future
            ensures the outer process exits gracefully on timeout.
            """
            with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(ta.propagate_with_confidence, _coin, _date)
                try:
                    return fut.result(timeout=row_timeout)
                except _cf.TimeoutError as te:
                    fut.cancel()
                    raise TimeoutError(f"propagate exceeded {row_timeout:.0f}s") from te

        for i, dt in enumerate(missing_dates):
            date_str = dt.strftime("%Y-%m-%d")
            signal = confidence = trader_text = None
            last_err: Exception | None = None
            for attempt in range(1, max_row_attempts + 1):
                try:
                    _, signal, confidence, trader_text = _propagate_with_timeout(
                        coin, date_str,
                    )
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt >= max_row_attempts:
                        break
                    wait = base_backoff * (2 ** (attempt - 1))
                    logger.warning(
                        f"{coin} @ {date_str}: propagate failed (attempt {attempt}/{max_row_attempts}): {e} — "
                        f"retry in {wait:.1f}s"
                    )
                    _time.sleep(wait)
            if last_err is not None:
                logger.error(f"{coin} @ {date_str}: giving up after {max_row_attempts} attempts: {last_err}")
                signal, confidence, trader_text = "HOLD", "UNKNOWN", f"ERROR: {last_err}"

            records.append({
                "date": dt,
                "signal": signal,
                "confidence": confidence,
                "trader_text": (trader_text or "")[:500],
            })

            # Checkpoint every row via atomic write (tmp file + rename) so a
            # crash or PC shutdown never leaves a half-written CSV.
            tmp_path = csv_path.with_suffix(".csv.tmp")
            pd.DataFrame(records).to_csv(tmp_path, index=False)
            tmp_path.replace(csv_path)
            if (i + 1) % 10 == 0 or (i + 1) == len(missing_dates):
                logger.info(f"{coin}: checkpoint {i + 1}/{len(missing_dates)} -> {csv_path}")

        df = pd.DataFrame(records)
        # Sort by date so the file remains chronological after a resume
        df = df.sort_values("date").reset_index(drop=True)
        tmp_path = csv_path.with_suffix(".csv.tmp")
        df.to_csv(tmp_path, index=False)
        tmp_path.replace(csv_path)
        logger.info(f"{coin}: saved {len(df)} signals to {csv_path}")
        results[coin] = df

    return results
