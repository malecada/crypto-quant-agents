"""Load V5 composite checkpoint, route each coin to its bundle, predict.

Inference companion to ``retrain.py``. Loads the composite dict that
``run_retrain`` wrote to disk — keyed by ``f"{coin}_{feature_set}"``, each
value a per-horizon bundle dict — materializes a PIT-aware feature frame
per route (one ``build_features_asof`` call per coin, since each coin has
its own routed pool + feature set), and emits one row per (coin, horizon)
in a tidy DataFrame.

Output schema (DataFrame):
    coin           — coin id (e.g. ``"bitcoin"``)
    horizon        — int day-horizon (e.g. ``7``)
    prediction     — float price prediction at that horizon
    ref_price      — float reference (asof) price
    bundle_route   — str route id (e.g. ``"bitcoin_78f"``) showing which
                     composite bundle produced this row

Per-coin failure isolation: if predict fails for one coin (data fetch,
feature build, predict_pooled raising), that coin is skipped and the
remaining coins continue. If a majority of coins fail (≥ 3 of 4 in a
4-coin universe; configurable via the ``max(3, n-1)`` threshold), a
``PredictMajorityFail`` is raised so the live runner can abort the cycle.

Design note — feature parity with retrain:
    ``build_features_asof`` lazily imports ``_transform_pooled`` from
    ``retrain.py``. The lazy import avoids any cycle at module load
    time, and reusing the exact transform helper guarantees that live
    and backtest paths see identical feature schemas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from tradingagents.models.lgb_model import predict_pooled

logger = logging.getLogger(__name__)


class PredictMajorityFail(RuntimeError):
    """≥ 3 of 4 coins failed predict — strategy cannot run."""


def build_features_asof(
    coin_pool: list[str],
    asof: str,
    store_root: Path | None = None,
    ohlcv_cache: Path | None = None,
    add_onchain_pit: bool = True,
    horizons: list[int] = (7, 14),
    lookback_days: int = 730,
) -> pd.DataFrame:
    """Build a feature frame for the asof date — one row per coin.

    Reuses the same pipeline as retrain (``build_pooled_dataset`` +
    ``_transform_pooled``) so live and backtest produce identical
    features. The latest row per coin is selected — that row's features
    describe the state of the world at asof, and the model predicts the
    next-period target from them.

    Args:
        coin_pool: CoinGecko coin IDs to fetch as the training pool
            (e.g. ``["bitcoin", "ethereum"]``). For V5 routing, every
            route's full pool is materialized so cross-asset features
            line up with the bundle's training distribution.
        asof: Upper-bound trade date (YYYY-mm-dd).
        store_root: PIT on-chain feature store root (forwarded to the
            data-fetch layer when used by callers; kept as a parameter
            for V5 composite compatibility).
        ohlcv_cache: OHLCV cache directory (same rationale).
        add_onchain_pit: Whether to include the PIT on-chain feature
            set. ``True`` for 193f routes, ``False`` for 78f routes.
        horizons: Forecast horizons to materialize as ``prices_h{h}``
            columns (forwarded to ``_transform_pooled``).
        lookback_days: How many days of history to load per coin.

    Returns:
        DataFrame with ``coin_id``, ``ref_price``, and all feature
        columns — one row per coin (the latest available date). Empty
        if no coin produced data.
    """
    from tradingagents.models.model_utils import build_pooled_dataset
    # Lazy import keeps `predict` independent of `retrain` at module load.
    # Both modules must use the same transform to preserve live↔backtest
    # feature equivalence; the equivalence test guards this.
    from tradingagents.execution.live.retrain import _transform_pooled

    pooled = build_pooled_dataset(
        coin_universe=list(coin_pool),
        lookback_days=lookback_days,
        horizons=list(horizons),
        trade_date=asof,
        add_onchain_pit=bool(add_onchain_pit),
    )
    transformed = _transform_pooled(pooled, list(horizons))
    if transformed is None or len(transformed) == 0:
        return pd.DataFrame()

    # `_transform_pooled` returns a date-indexed frame; surface `date` as a
    # regular column so we can sort/groupby uniformly.
    if "date" not in transformed.columns:
        transformed = transformed.reset_index()
    latest = (
        transformed.sort_values("date")
        .groupby("coin_id", as_index=False)
        .tail(1)
    )
    # Normalize column names: live API uses `ref_price`, but the trained
    # checkpoint's feature_names includes `prices` (since fit_pooled_full
    # treats it as a feature). Add `ref_price` as an alias and KEEP `prices`
    # so predict_pooled can still find it.
    if "prices" in latest.columns and "ref_price" not in latest.columns:
        latest = latest.copy()
        latest["ref_price"] = latest["prices"]
    return latest


def run_predict(
    coin_universe: list[str],
    routing: dict[str, dict[str, object]],
    ckpt_path: Path,
    asof: str,
    store_root: Path,
    ohlcv_cache: Path,
    horizons: list[int],
) -> pd.DataFrame:
    """V5 composite predict — route each coin to its bundle.

    For each coin in ``coin_universe``:
      1. Resolve its route from ``routing[coin]`` — pool + feature_set.
      2. ``build_features_asof`` with that pool and PIT toggle.
      3. Select the row for ``coin`` in the resulting frame.
      4. Look up the per-horizon bundles in the composite checkpoint
         under ``f"{coin}_{feature_set}"`` and call ``predict_pooled``
         once per horizon.

    Per-coin failures (data missing, predict raising, etc.) are caught
    and skipped — the remaining coins continue. If a majority of coins
    fail (``≥ max(3, n-1)``) we raise ``PredictMajorityFail`` so the
    live runner aborts the cycle cleanly. The ``max(3, n-1)`` threshold
    means: in a 4-coin universe 3 failures trigger; in a 2-coin test
    universe a single failure does NOT trigger (the threshold is 3).

    Args:
        coin_universe: Coin IDs to predict for, in priority order.
        routing: ``{coin: {"feature_set": "78f"|"193f", "pool": [...]}}``
            from V5 ROUTING config. Must contain every coin in
            ``coin_universe``.
        ckpt_path: Path to the composite ``.pkl`` written by
            ``retrain.run_retrain``. The composite is a dict keyed by
            ``f"{coin}_{feature_set}"``, each value a ``{horizon: bundle}``
            dict produced by ``fit_pooled_full``.
        asof: Upper-bound trade date (YYYY-mm-dd).
        store_root: PIT on-chain feature store root (forwarded into
            ``build_features_asof``).
        ohlcv_cache: OHLCV cache directory (forwarded into
            ``build_features_asof``).
        horizons: Forecast horizons in days. Each must have a bundle
            for every route or that coin will fail and be skipped.

    Returns:
        Long-format DataFrame with one row per (coin, horizon) and
        columns ``coin``, ``horizon``, ``prediction``, ``ref_price``,
        ``bundle_route``.

    Raises:
        PredictMajorityFail: when ``≥ max(3, len(coin_universe) - 1)``
            coins fail — strategy cannot run with that few signals.
    """
    composite = joblib.load(ckpt_path)

    out_rows: list[dict] = []
    failures: list[tuple[str, Exception]] = []

    for coin in coin_universe:
        route = routing[coin]
        feature_set = str(route["feature_set"])
        pool = list(route["pool"])
        route_id = f"{coin}_{feature_set}"
        use_pit = feature_set == "193f"

        try:
            feats = build_features_asof(
                coin_pool=pool,
                asof=asof,
                store_root=store_root,
                ohlcv_cache=ohlcv_cache,
                add_onchain_pit=use_pit,
                horizons=horizons,
            )
            row_df = feats[feats["coin_id"] == coin]
            if row_df.empty:
                raise ValueError(
                    f"no feature row for {coin} in pool {pool}"
                )
            row = row_df.iloc[[0]]

            if route_id not in composite:
                raise KeyError(
                    f"composite missing route {route_id}; "
                    f"have {sorted(composite.keys())}"
                )
            pool_bundles = composite[route_id]

            for h in horizons:
                if h not in pool_bundles:
                    raise KeyError(
                        f"route {route_id} missing horizon {h}"
                    )
                bundle = pool_bundles[h]
                pred = predict_pooled(bundle, row)
                out_rows.append({
                    "coin": coin,
                    "horizon": h,
                    "prediction": float(pred),
                    "ref_price": float(row["ref_price"].iloc[0]),
                    "bundle_route": route_id,
                })
        except Exception as exc:  # noqa: BLE001 — per-coin isolation
            logger.warning(
                "predict failed for %s asof %s (route=%s): %s",
                coin, asof, route_id, exc,
            )
            failures.append((coin, exc))

    threshold = max(3, len(coin_universe) - 1)
    if len(failures) >= threshold:
        raise PredictMajorityFail(
            f"{len(failures)}/{len(coin_universe)} coins failed predict "
            f"(threshold={threshold}): {failures}"
        )

    return pd.DataFrame(out_rows)
