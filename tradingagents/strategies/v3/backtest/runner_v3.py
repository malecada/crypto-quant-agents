"""V3 runner orchestrator: features → regime → models → sizing → trades.

Composes Phase 1-6 components into a single backtest entry point. Look-ahead-
safe: every per-bar computation slices inputs to ``index <= as_of`` before
running rolling/aggregation operations.

Calls into existing ``tradingagents.backtesting.engine.run_backtest`` for
trade execution (fees, slippage, hold logic, stop-loss, circuit breaker).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.backtesting.engine import BacktestResult, run_backtest
from tradingagents.backtesting.strategies import FiveLevelSignal, SignalLevel
from tradingagents.strategies.v3.config import V3Config
from tradingagents.strategies.v3.models.multi_horizon import (
    MultiHorizonEnsemble,
    consensus_signal,
)
from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3
from tradingagents.strategies.v3.regime.hmm_v2 import NHHmmBundle
from tradingagents.strategies.v2_sizing import apply_trend_filter
from tradingagents.strategies.v3.sizing.vol_target import (
    cdap_adjust,
    vol_target_position,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Walk-forward feature builder (vectorised; avoids O(n²) per-bar cost)
# ---------------------------------------------------------------------------


def build_global_features(
    prices: pd.Series,
    microstructure_features: pd.DataFrame,
    derivatives_features: pd.DataFrame,
) -> pd.DataFrame:
    """Build a full-history feature DataFrame aligned to ``prices.index``.

    Produces the same columns that ``_build_v3_features_at`` generates per bar,
    but computed once for the whole series (vectorised) so that walk-forward
    retraining is O(n) rather than O(n²).

    Rows with NaN (first 21 bars) are retained so integer-index slicing stays
    aligned with ``prices.index``; callers must dropna() or handle NaN before
    fitting.
    """
    idx = prices.index

    def _tz_norm(other_idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
        if idx.tz is not None and other_idx.tz is None:
            return other_idx.tz_localize("UTC")
        if idx.tz is None and other_idx.tz is not None:
            return other_idx.tz_localize(None)
        return other_idx

    ret_series = prices.pct_change()
    df = pd.DataFrame(index=idx)
    df["ret_1d"] = ret_series
    df["ret_5d"] = prices.pct_change(5)
    df["vol_5d"] = ret_series.rolling(5).std()
    df["vol_21d"] = ret_series.rolling(21).std()

    micro_cols = ["ofi_proxy", "ofi_proxy_w", "vol_dispersion"]
    if not microstructure_features.empty:
        m = microstructure_features.copy()
        m.index = _tz_norm(m.index)
        for col in micro_cols:
            if col in m.columns:
                df[col] = m[col].reindex(idx, method="ffill")
            else:
                df[col] = 0.0
    else:
        for col in micro_cols:
            df[col] = 0.0

    deriv_cols = ["funding_rate", "funding_rate_ma7"]
    if not derivatives_features.empty:
        d = derivatives_features.copy()
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


def train_walk_forward_mhe(
    global_features: pd.DataFrame,
    returns_series: pd.Series,
    as_of: pd.Timestamp,
    horizons: tuple[int, ...] = (3, 7, 14, 21),
    members: tuple[str, ...] = ("lgb",),
    use_calibration: bool = False,
    purge_horizon: int = 21,
    min_train_rows: int = 252,
) -> MultiHorizonEnsemble:
    """Train a fresh MultiHorizonEnsemble on data strictly before ``as_of``.

    Purges the last ``purge_horizon`` rows from the training tail so that
    h-step-ahead labels computed inside ``MultiHorizonEnsemble.fit`` cannot
    peek past ``as_of`` (label leakage guard).

    Args:
        global_features: Full-history feature DataFrame (price-index aligned).
            Produced by ``build_global_features``.
        returns_series: Simple-return Series aligned to ``global_features``.
        as_of: Bar date.  Training data is bounded to
            ``index <= as_of - purge_horizon days``.
        horizons: Forecast horizons (days).
        members: Ensemble members, e.g. ``("lgb",)``.
        use_calibration: Whether to fit isotonic calibrator on holdout 20%.
            Defaults to False — raw probs confirmed to yield wider spread and
            better signal coverage post root-cause analysis.
        purge_horizon: Number of days to purge from the train tail (should
            equal ``max(horizons)`` = 21 to prevent label leakage).
        min_train_rows: Minimum usable rows required; raises ValueError if
            insufficient data exists.

    Returns:
        Fitted ``MultiHorizonEnsemble``.

    Raises:
        ValueError: If training data before the purge cutoff is shorter than
            ``min_train_rows``.
    """
    cutoff = as_of - pd.Timedelta(days=purge_horizon)
    train_mask = global_features.index <= cutoff
    X_train = global_features.loc[train_mask].dropna()
    y_train = returns_series.loc[X_train.index]

    if len(X_train) < min_train_rows:
        raise ValueError(
            f"Insufficient train data ({len(X_train)} rows) at {as_of} "
            f"(cutoff={cutoff}); need >= {min_train_rows}"
        )

    mhe = MultiHorizonEnsemble(horizons=horizons, holdout_fraction=0.20)
    mhe.fit(X_train, y_train, members=members, use_calibration=use_calibration)
    return mhe


def _position_to_signal(position: float, low_vol_scale: float = 10.0) -> str:
    """Map a continuous position value to a 5-level signal string.

    ``low_vol_scale`` amplifies positions before thresholding when the underlying
    vol-targeted position is small due to high realized vol / low confidence.
    This preserves directional intent at the signal level without changing the
    actual position size used by the backtest engine.
    Threshold boundaries (post-scale): BUY>1.0, OVERWEIGHT>0.3, HOLD±0.3,
    UNDERWEIGHT>-1.0, SELL≤-1.0.
    """
    scaled = position * low_vol_scale
    if scaled > 1.0:
        return SignalLevel.BUY.value
    if scaled > 0.3:
        return SignalLevel.OVERWEIGHT.value
    if scaled >= -0.3:
        return SignalLevel.HOLD.value
    if scaled >= -1.0:
        return SignalLevel.UNDERWEIGHT.value
    return SignalLevel.SELL.value


def _build_v3_features_at(
    prices: pd.Series,
    microstructure_features: pd.DataFrame,
    derivatives_features: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Build a single-row feature vector usable by MultiHorizonEnsemble.

    Returns an empty DataFrame if not enough history is available.
    All slices are bounded to ``index <= as_of`` (look-ahead guard).
    """
    sub_prices = prices[prices.index <= as_of]
    if len(sub_prices) < 21:
        return pd.DataFrame()

    ret_1d = sub_prices.pct_change().iloc[-1]
    ret_5d = sub_prices.pct_change(5).iloc[-1]
    vol_5d = sub_prices.pct_change().rolling(5).std().iloc[-1]
    vol_21d = sub_prices.pct_change().rolling(21).std().iloc[-1]

    feats: dict[str, float] = {
        "ret_1d": float(ret_1d) if pd.notna(ret_1d) else 0.0,
        "ret_5d": float(ret_5d) if pd.notna(ret_5d) else 0.0,
        "vol_5d": float(vol_5d) if pd.notna(vol_5d) else 0.0,
        "vol_21d": float(vol_21d) if pd.notna(vol_21d) else 0.0,
    }

    # Helper: normalize a DatetimeIndex to match the tz of a reference Timestamp
    def _tz_normalize_index(idx: pd.DatetimeIndex, ref: pd.Timestamp) -> pd.DatetimeIndex:
        if ref.tz is not None and idx.tz is None:
            return idx.tz_localize("UTC")
        if ref.tz is None and idx.tz is not None:
            return idx.tz_localize(None)
        return idx

    # Append last microstructure row if available
    if not microstructure_features.empty:
        m = microstructure_features.copy()
        m.index = _tz_normalize_index(m.index, as_of)
        sub_m = m[m.index <= as_of]
        if not sub_m.empty:
            for col in sub_m.columns:
                val = sub_m.iloc[-1][col]
                feats[col] = float(val) if pd.notna(val) else 0.0

    if not derivatives_features.empty:
        d = derivatives_features.copy()
        d.index = _tz_normalize_index(d.index, as_of)
        sub_d = d[d.index <= as_of]
        if not sub_d.empty:
            for col in sub_d.columns:
                val = sub_d.iloc[-1][col]
                feats[col] = float(val) if pd.notna(val) else 0.0

    return pd.DataFrame([feats], index=[as_of])


def _extract_expected_features(mhe: MultiHorizonEnsemble) -> list[str]:
    """Extract the feature name list from a fitted MultiHorizonEnsemble.

    Tries ``feature_name_`` first (LightGBM native), then
    ``feature_names_in_`` (scikit-learn convention). Falls back to an empty
    list (runner will use whatever columns ``_build_v3_features_at`` produces).

    If the stored names are generic (``Column_N`` format), returns an empty
    list so the runner passes features as-is without column reordering.
    """
    import re
    _GENERIC_COL = re.compile(r"^Column_\d+$")

    for _h, ph in mhe._models.items():
        members = getattr(ph.ensemble, "_fitted_members", None)
        if not members:
            break
        first = next(iter(members.values()))
        # LightGBM: feature_name_ is "auto" when trained with plain arrays
        fn = getattr(first, "feature_name_", None)
        if fn is not None and fn != "auto" and len(fn) > 0:
            # Skip generic column names — they indicate training with plain arrays
            if not all(_GENERIC_COL.match(str(f)) for f in fn):
                return list(fn)
        # scikit-learn convention
        fn2 = getattr(first, "feature_names_in_", None)
        if fn2 is not None and len(fn2) > 0:
            if not all(_GENERIC_COL.match(str(f)) for f in fn2):
                return list(fn2)
        break
    return []


def run_v3_backtest(
    coin: str,
    prices: pd.Series,
    returns: pd.Series,
    microstructure_features: pd.DataFrame,
    derivatives_features: pd.DataFrame,
    regime_bundle: NHHmmBundle,
    multi_horizon_bundle: MultiHorizonEnsemble,
    config: V3Config,
    start: pd.Timestamp,
    end: pd.Timestamp,
    ticker: str = "",
    initial_capital: float = 10_000.0,
    signal_deadband: float = 0.02,
    # Walk-forward retraining knobs
    retrain_per_bar: bool = False,
    retrain_cadence: int = 1,
    retrain_members: tuple[str, ...] = ("lgb",),
    retrain_use_calibration: bool = False,
    # SMA30 trend filter (V2 bolt-on)
    sma30_filter: bool = False,
    sma30_multiplier: float = 1.5,
    # Pluggable feature builders (override default 9-feature pipeline).
    # ``global_features_override`` is a pre-built DataFrame aligned to
    # prices.index, used in place of build_global_features(). When provided,
    # ``features_at_builder`` (optional) extracts the single-bar feature row;
    # default selects the row whose index == as_of (or the last row <= as_of).
    global_features_override: pd.DataFrame | None = None,
    features_at_builder=None,
) -> BacktestResult:
    """End-to-end V3 backtest.

    Per-bar loop (look-ahead-safe):

    1. For each ``as_of`` in ``[start, end]``:
       - (Optional) Retrain ``MultiHorizonEnsemble`` on all data through
         ``as_of - purge_horizon`` when ``retrain_per_bar=True``.
       - Slice all inputs to ``index <= as_of``.
       - Build price + microstructure + derivatives features (single row).
       - Update ``RegimeState`` via ``detect_regime_v3``.
       - Per-horizon predictions via ``MultiHorizonEnsemble.predict_proba``.
       - ``consensus_signal`` → ``(direction, confidence)``.
       - ``vol_target_position`` + ``cdap_adjust`` → final position.
       - Convert position to 5-level agent signal string.

    2. Pass signal list + price arrays to
       ``tradingagents.backtesting.engine.run_backtest``.

    3. Return ``BacktestResult``.

    Args:
        coin: Coin name (e.g. ``"bitcoin"``).
        prices: Close-price series with DatetimeIndex.
        returns: Simple-return series aligned to ``prices``.
        microstructure_features: Optional microstructure feature DataFrame
            (empty DataFrame accepted — runner falls back to price-only feats).
        derivatives_features: Optional derivatives feature DataFrame.
        regime_bundle: Fitted ``NHHmmBundle`` from ``train_nh_hmm``.
        multi_horizon_bundle: Fitted ``MultiHorizonEnsemble`` (used as the
            initial model; may be replaced each bar when ``retrain_per_bar``
            is True).
        config: ``V3Config`` instance.
        start: First bar (inclusive) in the backtest window.
        end: Last bar (inclusive) in the backtest window.
        ticker: Ticker label stored in the result (defaults to ``coin.upper()``).
        initial_capital: Starting equity.
        retrain_per_bar: When True, retrain a fresh ``MultiHorizonEnsemble``
            at each bar (or every ``retrain_cadence`` bars) on all available
            data through ``as_of - 21 days`` (label-leakage guard).  Matches
            the V2 baseline's walk-forward retraining protocol.
        retrain_cadence: Retrain interval in bars (default 1 = every bar).
            Set to e.g. 7 to retrain weekly and reduce compute cost.
        retrain_members: Ensemble members to train during walk-forward, e.g.
            ``("lgb",)`` for LGB-only (fastest).
        retrain_use_calibration: Whether to fit isotonic calibrator during
            walk-forward retraining.  Defaults to False (raw probs better per
            root-cause analysis).
        sma30_filter: When True, apply V2-style SMA30 trend filter as a final
            position multiplier after vol-target + CDAP sizing.  1.5× when
            position aligns with trend (price > SMA30 → long boost, short
            damp); 0.5× when against.  Uses ``apply_trend_filter`` from
            ``tradingagents.strategies.v2_sizing``.
        sma30_multiplier: Aligned-direction multiplier for SMA30 filter
            (default 1.5 matches V2 default).

    Returns:
        ``BacktestResult`` from the V2 engine.
    """
    if start > end:
        raise ValueError("start must be <= end")
    bars = prices.loc[start:end].index
    if len(bars) == 0:
        raise ValueError(f"No bars between {start} and {end}")

    # Probe model feature names once — used to align runtime feature columns to
    # training schema. LGB trained on arrays uses feature_name_ == "auto"; in
    # that case fall through and use whatever columns the builder produces.
    expected_features = _extract_expected_features(multi_horizon_bundle)

    # Walk-forward setup: pre-compute vectorised feature matrix once (O(n))
    # so per-bar retraining is cheap (only slice + fit, no re-building).
    global_feats: pd.DataFrame | None = None
    if retrain_per_bar:
        logger.info(
            "Walk-forward retraining enabled: cadence=%d bar(s), members=%s, calibration=%s",
            retrain_cadence,
            retrain_members,
            retrain_use_calibration,
        )
        if global_features_override is not None:
            global_feats = global_features_override
            logger.info(
                "Using global_features_override: shape=%s",
                tuple(global_features_override.shape),
            )
        else:
            global_feats = build_global_features(
                prices, microstructure_features, derivatives_features
            )

    # Track the last successfully retrained model so we can fall back to it
    # if a bar lacks sufficient history.
    current_mhe: MultiHorizonEnsemble = multi_horizon_bundle

    agent_signals: list[str] = []
    raw_positions: list[float] = []  # tracked for SMA30 post-processing
    portfolio_dd_running = 0.0
    equity_high = float(initial_capital)
    equity_curr = float(initial_capital)

    for bar_i, as_of in enumerate(bars):
        # -------------------------------------------------------------------
        # Per-bar walk-forward retraining (when enabled)
        # -------------------------------------------------------------------
        if retrain_per_bar and global_feats is not None:
            if bar_i % retrain_cadence == 0:
                try:
                    current_mhe = train_walk_forward_mhe(
                        global_features=global_feats,
                        returns_series=returns,
                        as_of=as_of,
                        horizons=(3, 7, 14, 21),
                        members=retrain_members,
                        use_calibration=retrain_use_calibration,
                        purge_horizon=21,
                        min_train_rows=252,
                    )
                    if bar_i == 0 or bar_i % max(1, len(bars) // 5) == 0:
                        train_size = (global_feats.index <= as_of - pd.Timedelta(days=21)).sum()
                        logger.info(
                            "Retrained MHE at bar %d/%d (%s); train_size=%d",
                            bar_i + 1,
                            len(bars),
                            as_of.date(),
                            train_size,
                        )
                except ValueError as exc:
                    logger.warning(
                        "Walk-forward retrain failed at bar %d (%s): %s — using previous model",
                        bar_i,
                        as_of.date(),
                        exc,
                    )
        # Use the live (possibly just-retrained) model
        active_mhe = current_mhe
        if features_at_builder is not None and global_feats is not None:
            feat_df = features_at_builder(global_feats, as_of)
        elif global_features_override is not None:
            # Default: take the row at as_of from the override matrix.
            sub = global_features_override[global_features_override.index <= as_of]
            feat_df = sub.iloc[[-1]] if not sub.empty else pd.DataFrame()
        else:
            feat_df = _build_v3_features_at(
                prices, microstructure_features, derivatives_features, as_of
            )
        if feat_df.empty:
            agent_signals.append(SignalLevel.HOLD.value)
            raw_positions.append(0.0)
            continue

        # When walk-forward retraining is active, the freshly trained model
        # uses vectorised global features (DataFrame columns) rather than
        # plain numpy arrays, so expected_features alignment applies.
        # When using the frozen bundle, fall back to the pre-computed list.
        active_expected = expected_features
        if retrain_per_bar and active_mhe is not multi_horizon_bundle:
            # Walk-forward model was trained on global_feats — use its columns
            active_expected = list(global_feats.columns) if global_feats is not None else expected_features

        # Align columns to training schema when we have explicit names.
        if active_expected:
            for col in active_expected:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[active_expected]

        try:
            probas_dict = active_mhe.predict_proba(feat_df)
        except Exception:
            logger.exception("predict_proba failed at %s; falling back to HOLD", as_of)
            agent_signals.append(SignalLevel.HOLD.value)
            raw_positions.append(0.0)
            continue

        # predict_proba returns dict[int, np.ndarray] — extract scalar per horizon
        scalar_probas: dict[int, float] = {
            h: float(arr[0]) for h, arr in probas_dict.items()
        }

        try:
            regime = detect_regime_v3(
                prices=prices, bundle=regime_bundle, as_of=as_of
            )
        except Exception:
            logger.exception("detect_regime_v3 failed at %s; falling back to HOLD", as_of)
            agent_signals.append(SignalLevel.HOLD.value)
            raw_positions.append(0.0)
            continue

        direction, confidence = consensus_signal(scalar_probas, regime, config, deadband=signal_deadband)

        # Realized annualised vol from log returns (21-bar rolling)
        sub_rets = returns.loc[returns.index <= as_of].iloc[-21:]
        rv = float(sub_rets.std() * np.sqrt(252)) if len(sub_rets) > 1 else 0.15

        position = vol_target_position(
            direction=direction,
            confidence=confidence,
            realized_vol_annual=rv,
            target_vol_annual=config.target_annual_vol,
            max_leverage=config.max_leverage,
        )
        position = cdap_adjust(
            position=position,
            portfolio_dd_pct=portfolio_dd_running,
            regime=regime,
            config=config,
        )

        # Update running equity simulation for CDAP drawdown tracking
        # (approximate — does not replicate the engine's exact cost model)
        daily_ret = returns.loc[as_of] if as_of in returns.index else 0.0
        gross = position * float(daily_ret)
        equity_curr = equity_curr * (1.0 + gross)
        if equity_curr > equity_high:
            equity_high = equity_curr
        portfolio_dd_running = (
            (equity_high - equity_curr) / equity_high if equity_high > 0 else 0.0
        )

        agent_signals.append(_position_to_signal(position))
        raw_positions.append(position)

    # Safety: pad / truncate to exactly len(bars)
    while len(agent_signals) < len(bars):
        agent_signals.append(SignalLevel.HOLD.value)
        raw_positions.append(0.0)
    agent_signals = agent_signals[: len(bars)]
    raw_positions = raw_positions[: len(bars)]

    # Optional SMA30 trend filter (V2 bolt-on):
    # Apply V2's apply_trend_filter on the accumulated position array, then
    # re-convert to 5-level signal strings.  The full price series through the
    # backtest window is used so SMA lookback is correct for early bars.
    if sma30_filter:
        pos_arr = np.array(raw_positions, dtype=float)
        bar_prices = prices.loc[bars].values.astype(float)
        filtered_pos = apply_trend_filter(
            positions=pos_arr,
            prices=bar_prices,
            sma_period=30,
            multiplier=sma30_multiplier,
        )
        agent_signals = [_position_to_signal(float(p)) for p in filtered_pos]
        logger.info(
            "SMA30 filter applied (%d bars); pre-filter non-HOLD=%d, post-filter non-HOLD=%d",
            len(bars),
            sum(1 for s in agent_signals if s != SignalLevel.HOLD.value),
            sum(1 for s in [_position_to_signal(float(p)) for p in filtered_pos] if s != SignalLevel.HOLD.value),
        )

    actuals = prices.loc[bars].values
    dates = pd.Series(bars)

    return run_backtest(
        dates=dates,
        actuals=actuals,
        agent_signals=agent_signals,
        strategy=FiveLevelSignal(),
        ticker=ticker or coin.upper(),
        initial_capital=initial_capital,
    )
