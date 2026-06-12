"""Daily fit-once-on-full-history retrain of pooled LGB models (V5 composite).

Wraps `tradingagents.models.model_utils.build_pooled_dataset` +
`tradingagents.models.lgb_model.fit_pooled_full` so the live runner can
invoke them with the live module's vocabulary (`routing`, `asof`) and get
back a versioned, joblib-persistable composite checkpoint.

Architecture note — why not `model_run_pooled`?
   The upstream `model_run_pooled` is walk-forward EVAL: for each iteration
   it fits a per-iteration model and throws it away, returning predictions +
   metrics. That contract has no persistable booster usable by live
   inference. The live cycle needs a "fit once on the full history → save →
   .predict()" path, which is what `lgb_model.fit_pooled_full` provides.
   This module is the integration glue.

Composite layout on disk (joblib.dump of):
    {
        "bitcoin_78f":  {7: bundle, 14: bundle},
        "ethereum_193f": {7: bundle, 14: bundle},
        "binancecoin_78f": {...},
        "solana_193f":  {...},
    }
where each bundle is {booster, feature_names, horizon, target_col,
n_train_rows, scaler, coin_to_int}.

Tests patch `build_pooled_dataset`, `_transform_pooled`, and
`fit_pooled_full` on this module; the real upstream signatures are only
exercised in live cycles.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd

from tradingagents.models.lgb_model import fit_pooled_full
from tradingagents.models.model_utils import build_pooled_dataset, data_transform

logger = logging.getLogger(__name__)


@dataclass
class CheckpointArtifact:
    """V5 composite checkpoint metadata.

    Wraps a single `.pkl` containing 4 per-route ``fit_pooled_full`` bundles
    keyed by ``f"{coin}_{feature_set}"``. See ``run_retrain`` docstring for
    the on-disk shape.
    """

    path: Path
    sha: str
    retrain_id: str
    routes: list[str] = field(default_factory=list)
    n_train_rows: int = 0
    train_window_start: str = ""
    train_dir_acc: float = 0.0


def _transform_pooled(pooled_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Apply `data_transform` per coin so .shift() respects coin boundaries.

    Mirrors the canonical wiring in `scripts/evaluate_models_multi.py`
    (`build_pooled_transformed`): split by coin, run the per-coin transform,
    re-tag with `coin_id`, concat, and set the `date` column as index.

    Args:
        pooled_df: Raw output of `build_pooled_dataset` (date-indexed,
            with a `coin_id` column).
        horizons: Forecast horizons to materialize as `prices_h{h}` columns.

    Returns:
        Date-indexed pooled DataFrame with one `coin_id` column and
        `prices_h{h}` target columns. Empty if no coin produced data.
    """
    if pooled_df is None or len(pooled_df) == 0 or "coin_id" not in pooled_df.columns:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    for coin in pooled_df["coin_id"].unique():
        sub = pooled_df[pooled_df["coin_id"] == coin].drop(columns=["coin_id"])
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
        return pd.DataFrame()

    pooled = pd.concat(pieces, ignore_index=True)
    pooled["date"] = pd.to_datetime(pooled["date"])
    pooled = pooled.set_index("date").sort_index()
    return pooled


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def run_retrain(
    routing: dict[str, dict[str, object]],
    horizons: list[int],
    asof: str,
    checkpoint_dir: Path,
    retrain_id: str = "",
    lookback_days: int = 730,
) -> CheckpointArtifact:
    """V5 composite retrain — 4 fit_pooled_full bundles in one .pkl.

    For each (coin, route) in ``routing``:
      1. ``build_pooled_dataset`` with route['pool'] and add_onchain_pit per
         route['feature_set'].
      2. ``_transform_pooled`` to add prices_h{h} target columns.
      3. ``fit_pooled_full`` per horizon — bundle stored under
         ``f"{coin}_{route['feature_set']}"``.

    Final composite ``{route_id: {h: bundle}}`` is joblib.dump'd to
    ``{checkpoint_dir}/lgb_v5_mix_{asof}.pkl``. Atomic: written via tmp
    + rename to prevent half-files.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    composite: dict[str, dict[int, dict]] = {}
    for coin, route in routing.items():
        pool = list(route["pool"])
        use_pit = route["feature_set"] == "193f"
        route_id = f"{coin}_{route['feature_set']}"

        raw = build_pooled_dataset(
            coin_universe=pool,
            lookback_days=lookback_days,
            horizons=horizons,
            trade_date=asof,
            add_technical=True,
            add_cross_asset=True,
            add_onchain=True,
            add_onchain_pit=use_pit,
        )
        transformed = _transform_pooled(raw, horizons)

        composite[route_id] = {}
        for h in horizons:
            composite[route_id][h] = fit_pooled_full(transformed, horizon=h)

    out_tmp = checkpoint_dir / f"lgb_v5_mix_{asof}.pkl.tmp"
    out_final = checkpoint_dir / f"lgb_v5_mix_{asof}.pkl"
    try:
        joblib.dump(composite, out_tmp)
        out_tmp.rename(out_final)
    except Exception:
        if out_tmp.exists():
            out_tmp.unlink()
        raise

    return CheckpointArtifact(
        path=out_final,
        sha=_sha256_of(out_final),
        retrain_id=retrain_id,
        routes=sorted(composite.keys()),
        n_train_rows=sum(b[horizons[0]]["n_train_rows"]
                          for b in composite.values()),
        train_window_start=asof,
        train_dir_acc=0.0,
    )


def run_retrain_with_fallback(
    routing: dict[str, dict[str, object]],
    horizons: list[int],
    asof: str,
    checkpoint_dir: Path,
    retrain_id: str = "",
    lookback_days: int = 730,
) -> CheckpointArtifact:
    """Try `run_retrain`; on any failure return the most recent existing
    composite. Composite atomicity = all 4 routes fresh or all 4 fall back —
    never mixed-vintage.

    Used by the live runner so a single bad data fetch (CoinMetrics outage,
    DefiLlama 5xx, ...) doesn't break the daily cycle — we keep trading with
    yesterday's composite checkpoint. If no prior composite exists, raise.
    """
    try:
        return run_retrain(
            routing=routing, horizons=horizons, asof=asof,
            checkpoint_dir=Path(checkpoint_dir), retrain_id=retrain_id,
            lookback_days=lookback_days,
        )
    except Exception as exc:
        logger.warning(
            "V5 retrain failed: %s — falling back to previous composite", exc
        )
        previous = sorted(Path(checkpoint_dir).glob("lgb_v5_mix_*.pkl"))
        if not previous:
            raise RuntimeError(
                "V5 retrain failed and no previous composite to fall back to"
            ) from exc
        prior_path = previous[-1]
        # Recover route list from the loaded composite
        composite = joblib.load(prior_path)
        return CheckpointArtifact(
            path=prior_path,
            sha=_sha256_of(prior_path),
            retrain_id=retrain_id,
            routes=sorted(composite.keys()),
            train_window_start=prior_path.stem.split("_")[-1],
            train_dir_acc=0.0,
        )
