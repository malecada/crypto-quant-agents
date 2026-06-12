"""Data transformation and metrics utilities for prediction models.

Adapted from Krypto-v0/src/models/model_utils.py to work with the OHLCV
DataFrame format returned by the CoinGecko/Binance data vendor.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# Stockstats indicators to compute as model features
TECHNICAL_INDICATORS = [
    "rsi_14",
    "rsi_30",
    "macd",
    "macds",
    "macdh",
    "boll",
    "boll_ub",
    "boll_lb",
    "atr_14",
    "adx",
    "cci_20",
    "kdjk",
    "kdjd",
    "wr_14",
]


def compute_metrics(y_true, y_pred):
    """Return a dict of regression metrics (R2, MAE, RMSE, MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def compute_technical_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute stockstats indicators on raw OHLCV.

    Args:
        df_ohlcv: DataFrame with at least columns Date, Open, High, Low, Close, Volume.

    Returns:
        DataFrame with the same row order as df_ohlcv (RangeIndex), one column per
        indicator prefixed `ti_`. Indicators that fail to compute are filled with NaN.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    from stockstats import wrap

    # stockstats expects lowercase column names: open, high, low, close, volume, date
    src = df_ohlcv.copy()
    if "Date" in src.columns:
        src = src.rename(columns={"Date": "date"})
    rename_map = {c: c.lower() for c in src.columns if c.lower() != c}
    src = src.rename(columns=rename_map)

    sdf = wrap(src.copy())
    out = pd.DataFrame(index=df_ohlcv.index)
    for ind in TECHNICAL_INDICATORS:
        try:
            out[f"ti_{ind}"] = np.asarray(sdf[ind].values, dtype=float)
        except Exception as e:
            logger.debug(f"Failed to compute {ind}: {e}")
            out[f"ti_{ind}"] = np.nan
    return out


def add_cross_asset_features(
    coin_dfs: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Add BTC-anchored cross-asset features to each coin's model df.

    Adds three columns prefixed `xa_`:
      - xa_btc_return: BTC daily return (reindexed to each coin's dates)
      - xa_eth_btc_ratio: ETH price / BTC price (if ethereum present, else 1.0)
      - xa_btc_dom: BTC volume as a proxy for dominance

    Args:
        coin_dfs: dict mapping coin_id -> DataFrame from ohlcv_to_model_df()
        btc_df: BTC's DataFrame from ohlcv_to_model_df()

    Returns:
        New dict with same keys, each value enriched with xa_* columns.
    """
    btc_returns = btc_df["prices"].pct_change()
    if "ethereum" in coin_dfs and not coin_dfs["ethereum"].empty:
        eth_btc_ratio = coin_dfs["ethereum"]["prices"] / btc_df["prices"].reindex(
            coin_dfs["ethereum"].index
        )
    else:
        eth_btc_ratio = pd.Series(1.0, index=btc_df.index)
    btc_volume = btc_df["total_volumes"]

    enriched = {}
    for coin, df in coin_dfs.items():
        df = df.copy()
        df["xa_btc_return"] = btc_returns.reindex(df.index).fillna(0).values
        df["xa_eth_btc_ratio"] = eth_btc_ratio.reindex(df.index).ffill().fillna(1.0).values
        df["xa_btc_dom"] = btc_volume.reindex(df.index).fillna(0).values
        enriched[coin] = df
    return enriched


def add_onchain_features(
    df: pd.DataFrame,
    coin_id: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """Join funding rate, TVL delta, and stablecoin mcap delta into a model df.

    All on-chain fetches are best-effort: any failure results in the column
    being filled with zeros (model still trains, just without that feature).

    Args:
        df: Date-indexed DataFrame from ohlcv_to_model_df().
        coin_id: CoinGecko ID (used to resolve the Binance symbol for funding rate).
        start_date: Python date object — earliest date to fetch.
        end_date: Python date object — latest date to fetch.

    Returns:
        Same df with three new columns: oc_funding_rate, oc_tvl_delta, oc_stable_delta.
    """
    from tradingagents.dataflows.coingecko_binance import _resolve_binance_symbol
    from tradingagents.dataflows.onchain import (
        _scrape_funding_rates,
        _scrape_stablecoin_mcap_history,
        _scrape_total_tvl,
    )

    df = df.copy()

    # Funding rate (per-coin via Binance)
    binance_symbol = _resolve_binance_symbol(coin_id)
    if binance_symbol is None:
        df["oc_funding_rate"] = 0.0
    else:
        try:
            fr = _scrape_funding_rates(start_date, end_date, binance_symbol)
            if not fr.empty:
                fr.index = pd.to_datetime(fr.index)
                df["oc_funding_rate"] = (
                    fr["funding_rate"].reindex(df.index).ffill().fillna(0).values
                )
            else:
                df["oc_funding_rate"] = 0.0
        except Exception as e:
            logger.debug(f"Funding rate fetch failed for {coin_id}: {e}")
            df["oc_funding_rate"] = 0.0

    # Total TVL delta (global, same for all coins)
    try:
        tvl = _scrape_total_tvl(start_date, end_date)
        if not tvl.empty:
            tvl.index = pd.to_datetime(tvl.index)
            tvl_series = tvl["total_tvl"].reindex(df.index).ffill()
            df["oc_tvl_delta"] = tvl_series.pct_change().fillna(0).values
        else:
            df["oc_tvl_delta"] = 0.0
    except Exception as e:
        logger.debug(f"TVL fetch failed: {e}")
        df["oc_tvl_delta"] = 0.0

    # Stablecoin mcap delta (global)
    try:
        sc = _scrape_stablecoin_mcap_history(start_date, end_date)
        if not sc.empty:
            sc.index = pd.to_datetime(sc.index)
            sc_series = sc["stablecoin_mcap"].reindex(df.index).ffill()
            df["oc_stable_delta"] = sc_series.pct_change().fillna(0).values
        else:
            df["oc_stable_delta"] = 0.0
    except Exception as e:
        logger.debug(f"Stablecoin fetch failed: {e}")
        df["oc_stable_delta"] = 0.0

    return df


def ohlcv_to_model_df(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Convert OHLCV DataFrame (from CoinGecko/Binance vendor) to model-ready format.

    The vendor returns columns: Date, Open, High, Low, Close, Volume.
    This function produces a date-indexed DataFrame with 'prices' as the
    close price column, plus derived features compatible with the original
    Krypto-v0 data_transform pipeline.

    Args:
        df_ohlcv: DataFrame with Date, Open, High, Low, Close, Volume columns.

    Returns:
        DataFrame indexed by date with 'prices' and supplementary features.
    """
    if df_ohlcv.empty:
        return pd.DataFrame()

    df = df_ohlcv.copy()

    # Ensure Date column is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    df = df.sort_index()

    # Rename Close -> prices (the target column expected by all models)
    result = pd.DataFrame(index=df.index)
    result["prices"] = df["Close"].astype(float)

    # Derive supplementary features from OHLCV
    result["open"] = df["Open"].astype(float)
    result["high"] = df["High"].astype(float)
    result["low"] = df["Low"].astype(float)
    result["total_volumes"] = df["Volume"].astype(float)

    # Price-derived features
    result["daily_return"] = result["prices"].pct_change()
    result["high_low_spread"] = result["high"] - result["low"]
    result["open_close_spread"] = result["prices"] - result["open"]

    # Rolling statistics
    for window in [7, 14, 30]:
        result[f"ma_{window}"] = result["prices"].rolling(window).mean()
        result[f"vol_{window}"] = result["prices"].rolling(window).std()

    # Volume moving averages
    result["vol_ma_7"] = result["total_volumes"].rolling(7).mean()
    result["vol_ma_30"] = result["total_volumes"].rolling(30).mean()

    # Add an 'id' column (placeholder, used by data_transform for dummy encoding)
    result["id"] = "crypto"

    return result


def data_transform(
    df_all: pd.DataFrame,
    first_day_future,
    include_future_row: bool = True,
    horizons=(1,),
):
    """Transform model DataFrame into features suitable for model training.

    Adapted from the original Krypto-v0 data_transform. The .shift(1) aligns
    features so that row i contains feature values from day i-1 -- no look-ahead.

    Args:
        df_all: Date-indexed DataFrame (output of ohlcv_to_model_df or similar).
        first_day_future: Date for the forecast horizon.
        include_future_row: If True, append a placeholder row for the forecast date.
        horizons: Iterable of integer horizons to create target columns for.
            For each h in horizons, a `prices_h{h}` column is added that holds
            the price h days *after* the row's features. The default (1,) is
            backward-compatible with the original single-horizon behavior; the
            existing `prices` target column remains unchanged.

    Returns:
        (reframed_lags, df_final) — same as before, but reframed_lags has a
        `date` column with the row date so callers can re-index if needed.
    """
    cfg = get_config().get("prediction_models", {})
    n_lags = cfg.get("lag_features", 7)

    df_all = df_all.copy()

    # Drop 'market_caps' if present (not used in model training)
    if "market_caps" in df_all.columns:
        df_all = df_all.drop(columns="market_caps")

    if "index" in df_all.columns:
        df_all = df_all.drop(columns="index")

    # Add multi-horizon targets BEFORE the shift. These hold the price h days
    # ahead of the row, so after the shift below they become "h days ahead of
    # the day from which features were drawn".
    horizons = tuple(int(h) for h in horizons)
    for h in horizons:
        df_all[f"prices_h{h}"] = df_all["prices"].shift(-h)

    if include_future_row:
        # Build a placeholder row for the forecast date, copying the last
        # known feature values (they will be shifted down by one position).
        future_idx = pd.to_datetime(first_day_future)
        future_row = df_all.iloc[[-1]].copy()
        future_row.index = [future_idx]
        # Cast to match dtypes before concat
        future_row = future_row.astype(
            {col: df_all[col].dtype for col in future_row.columns if col in df_all.columns},
            errors="ignore",
        )
        df_with_future = pd.concat([df_all, future_row], axis=0)
    else:
        df_with_future = df_all.copy()

    df_with_future.index.names = ["date"]
    df_with_future.index = pd.to_datetime(df_with_future.index).strftime("%Y-%m-%d")

    # Shift all columns down by 1 so features at position i originate
    # from day i-1 (prevents using same-day information for prediction).
    df_with_future = df_with_future.shift()

    # After shift(), row 0 is all-NaN -- drop it.
    df_with_future = df_with_future.iloc[1:]

    # Forward-fill, then fill remaining NaN with 0. Target columns
    # (`prices_h{h}`) are intentionally excluded — the last h rows have no
    # future price label, and downstream `walk_forward_pooled` /
    # `fit_pooled_full` rely on `dropna(subset=[target_col])` to skip them.
    # Filling targets here masks the NaN, the dropna no-ops, and the model
    # produces predictions for dates it shouldn't (e.g. SOL 2026-05-25
    # walk-forward extrapolated to -10.04 in the V5 parity check).
    target_cols = [c for c in df_with_future.columns if c.startswith("prices_h")]
    non_target_cols = [c for c in df_with_future.columns if c not in target_cols]
    df_final = df_with_future.copy()
    df_final[non_target_cols] = (
        df_final[non_target_cols].infer_objects(copy=False).ffill().fillna(0)
    )

    # Name/dummy encoding
    if "id" in df_final.columns:
        df_final["name"] = np.repeat(df_final["id"].iloc[0], len(df_final))
        df_final = df_final.drop(columns="id")
    else:
        df_final["name"] = "crypto"

    df_final["name_no"] = pd.get_dummies(df_final["name"], dtype="int")
    df_final.index = pd.to_datetime(df_final.index, utc=True)
    df_final["Day"] = df_final.index.day
    df_final["Month"] = df_final.index.month
    df_final["Year"] = df_final.index.year

    seasonal_dummy = pd.get_dummies(df_final.index.day, dtype="int")
    seasonal_dummy.index = df_final.index
    seasonal_dummy.columns = [f"day_{v}" for v in seasonal_dummy.columns]

    reframed = pd.concat([df_final, seasonal_dummy], axis=1).drop(columns="name")
    cols_to_drop = [c for c in reframed.columns if c == "date"]
    if cols_to_drop:
        reframed = reframed.drop(columns=cols_to_drop)
    # Preserve dates for pooled callers BEFORE the positional reset.
    reframed["date"] = df_final.index
    reframed = reframed.reset_index(drop=True)

    # Lag features (backward-looking only)
    reframed_lags = reframed.copy()
    prices = reframed_lags["prices"].values
    for k in range(1, n_lags + 1):
        reframed_lags[f"lag{k}"] = pd.Series(prices).shift(k).values

    return reframed_lags, df_final


def build_pooled_dataset(
    coin_universe: list,
    lookback_days: int,
    horizons: list,
    trade_date: str | None = None,
    add_technical: bool = True,
    add_cross_asset: bool = True,
    add_onchain: bool = True,
    add_onchain_pit: bool = False,
) -> pd.DataFrame:
    """Build a pooled multi-coin dataset enriched with optional features.

    For each coin in the universe:
      1. Fetch OHLCV via the cached vendor layer
      2. Optionally compute stockstats technical indicators (ti_*)
      3. Convert to model df via ohlcv_to_model_df()
      4. Optionally enrich with cross-asset features (xa_*)
      5. Optionally enrich with on-chain features (oc_*)
      6. Tag with coin_id and concat into one wide DataFrame

    The returned DataFrame is the *pre-transform* input — call data_transform()
    per coin (so .shift() respects coin boundaries) to create the final
    training dataset with `prices_h{h}` target columns.

    Args:
        coin_universe: List of CoinGecko coin IDs (e.g. "bitcoin", "ethereum").
        lookback_days: How many days of history to fetch per coin.
        horizons: List of integer horizons (e.g. [1, 3, 7, 14]).
        trade_date: Upper date boundary (YYYY-mm-dd). None = today.
        add_technical: If True, compute and merge stockstats indicators.
        add_cross_asset: If True, add BTC-anchored cross-asset features.
        add_onchain: If True, add funding rate / TVL / stablecoin features.

    Returns:
        Concatenated date-indexed DataFrame with one `coin_id` column. Empty
        DataFrame if no coins yielded data.
    """
    from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv

    if trade_date is not None:
        end_date = datetime.strptime(trade_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    coin_dfs_model: dict[str, pd.DataFrame] = {}

    for coin in coin_universe:
        end_str = end_date.strftime("%Y-%m-%d")
        try:
            df_ohlcv = _load_crypto_ohlcv(coin, end_str)
        except Exception as e:
            logger.warning(f"Failed to fetch OHLCV for {coin}: {e}")
            continue

        if df_ohlcv.empty:
            logger.warning(f"No OHLCV data for {coin}, skipping")
            continue

        # Filter to lookback window
        if "Date" in df_ohlcv.columns:
            df_ohlcv = df_ohlcv[df_ohlcv["Date"] >= pd.to_datetime(start_date)].copy()
        df_ohlcv = df_ohlcv.reset_index(drop=True)

        if df_ohlcv.empty:
            logger.warning(f"No OHLCV in lookback window for {coin}, skipping")
            continue

        # Compute technical indicators on raw OHLCV
        if add_technical:
            ti = compute_technical_indicators(df_ohlcv)
            # ti has the same RangeIndex as df_ohlcv after reset_index
            df_ohlcv = pd.concat([df_ohlcv, ti], axis=1)

        df_model = ohlcv_to_model_df(df_ohlcv)

        # Carry the technical indicator columns through (ohlcv_to_model_df drops them)
        if add_technical and not df_model.empty:
            ti_for_model = ti.copy()
            ti_for_model.index = df_model.index
            for col in ti_for_model.columns:
                df_model[col] = ti_for_model[col].values

        coin_dfs_model[coin] = df_model

    if not coin_dfs_model:
        logger.error("No coins yielded data; returning empty DataFrame")
        return pd.DataFrame()

    # Cross-asset features (uses BTC as anchor)
    if add_cross_asset and "bitcoin" in coin_dfs_model:
        coin_dfs_model = add_cross_asset_features(coin_dfs_model, coin_dfs_model["bitcoin"])

    # On-chain features per coin (best-effort)
    if add_onchain:
        for coin, df in list(coin_dfs_model.items()):
            coin_dfs_model[coin] = add_onchain_features(
                df, coin, start_date.date(), end_date.date(),
            )

    # PIT on-chain features from the bitemporal store (preferred; opt-in).
    # When enabled this SUPPLEMENTS add_onchain with PIT-safe metrics
    # (MVRV, exchange flows, active addresses, Puell, TVL). Leakage-safe.
    if add_onchain_pit:
        from tradingagents.dataflows.onchain_features import (
            build_pit_onchain_features,
        )
        per_coin_feats: dict[str, pd.DataFrame] = {}
        for coin, df in list(coin_dfs_model.items()):
            try:
                feats = build_pit_onchain_features(coin, df.index)
            except Exception as e:  # pragma: no cover
                logger.warning(f"PIT on-chain fetch failed for {coin}: {e}")
                feats = pd.DataFrame(index=df.index)
            if feats is None:
                feats = pd.DataFrame(index=df.index)
            if not feats.empty:
                # Align tz: model df indices are tz-naive (pandas Timestamp),
                # PIT feature indices are tz-aware UTC. Strip tz on feats so
                # reindex can join on the calendar day.
                feats = feats.copy()
                feats.index = feats.index.tz_convert("UTC").tz_localize(None)
                feats = feats.reindex(df.index)
            per_coin_feats[coin] = feats

        # Pool-wide oc_* column union. For coins with thin coverage (e.g.
        # BNB returning only tvl_bsc + stablecoin), fill the missing oc_*
        # columns with 0 so pooled LGB sees a consistent schema across coins.
        # Keeping NaN here pollutes the pool — LGB treats NaN as signal,
        # which on a thin-coverage coin lets the model latch onto coin
        # identity instead of feature semantics. 0 = "feature unobserved
        # for this coin" is a cleaner null encoding for a tree model.
        all_oc_cols: set[str] = set()
        for f in per_coin_feats.values():
            all_oc_cols.update(c for c in f.columns if c.startswith("oc_"))
        for coin, df in list(coin_dfs_model.items()):
            feats = per_coin_feats.get(coin, pd.DataFrame(index=df.index))
            for col in sorted(all_oc_cols):
                if col in feats.columns:
                    df[col] = feats[col].values
                else:
                    df[col] = 0.0  # thin-coverage mask
            coin_dfs_model[coin] = df

    # Tag with coin_id and concat
    pooled_rows = []
    for coin, df in coin_dfs_model.items():
        df = df.copy()
        df["coin_id"] = coin
        pooled_rows.append(df)

    pooled = pd.concat(pooled_rows, axis=0)
    pooled = pooled.sort_index()
    return pooled


def fetch_ohlcv_for_model(
    coingecko_id: str, lookback_days: int, trade_date: str | None = None
) -> pd.DataFrame:
    """Fetch OHLCV data via the CoinGecko/Binance vendor and convert for model use.

    This is the bridge between TradingAgents' data vendor layer and the
    prediction model pipeline.

    Args:
        coingecko_id: CoinGecko coin ID (e.g. "bitcoin", "ethereum").
        lookback_days: Number of days of historical data to fetch.
        trade_date: Upper date boundary (YYYY-mm-dd). When backtesting, this
            must be the simulation date to prevent look-ahead bias.  Defaults
            to today for live usage.

    Returns:
        Date-indexed DataFrame ready for data_transform().
    """
    # Import here to avoid circular imports at module load time
    from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv

    if trade_date is not None:
        end_date = datetime.strptime(trade_date, "%Y-%m-%d")
    else:
        end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    # _load_crypto_ohlcv expects a date string for filtering
    end_str = end_date.strftime("%Y-%m-%d")

    df_ohlcv = _load_crypto_ohlcv(coingecko_id, end_str)

    if df_ohlcv.empty:
        logger.warning(f"No OHLCV data returned for {coingecko_id}")
        return pd.DataFrame()

    # Filter to lookback window
    start_dt = pd.to_datetime(start_date)
    if "Date" in df_ohlcv.columns:
        df_ohlcv = df_ohlcv[df_ohlcv["Date"] >= start_dt]

    # Convert to model format
    return ohlcv_to_model_df(df_ohlcv)
