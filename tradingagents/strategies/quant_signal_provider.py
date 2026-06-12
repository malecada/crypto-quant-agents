"""V2 / V3 quant signal provider abstraction.

Hybrid scripts pick a quant version via ``build_provider("v2")`` or
``build_provider("v3", ...)``. Both providers emit the existing
``tradingagents.strategies.contracts.QuantSignal`` so downstream modulator
code is unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import numpy as np
import pandas as pd

from tradingagents.strategies.contracts import (
    DirectionLabel,
    QuantSignal,
    RegimeLabel,
)

logger = logging.getLogger(__name__)


# Indirection so tests can monkeypatch
def _v2_get_quant_signal(coin, date, base_dir=None):
    from tradingagents.strategies.quant_engine import get_quant_signal as _impl
    return _impl(coin, date, base_dir)


class QuantSignalProvider(Protocol):
    def signal(self, coin: str, as_of: pd.Timestamp) -> QuantSignal: ...


class V2QuantSignalProvider:
    """V2 provider: wraps existing get_quant_signal."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        self.base_dir = base_dir

    def signal(self, coin: str, as_of: pd.Timestamp) -> QuantSignal:
        date_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        return _v2_get_quant_signal(coin=coin, date=date_str, base_dir=self.base_dir)


class V3QuantSignalProvider:
    """V3 provider: runs single-bar V3 prediction and builds a V2-schema QuantSignal."""

    def __init__(
        self,
        prices: pd.Series,
        regime_bundle,
        multi_horizon_bundle,
        microstructure_features: pd.DataFrame,
        derivatives_features: pd.DataFrame,
        config,
    ) -> None:
        self.prices = prices
        self.regime_bundle = regime_bundle
        self.multi_horizon_bundle = multi_horizon_bundle
        self.microstructure_features = microstructure_features
        self.derivatives_features = derivatives_features
        self.config = config

    def signal(self, coin: str, as_of: pd.Timestamp) -> QuantSignal:
        # Lazy imports to avoid circular dependencies
        from tradingagents.strategies.v3.backtest.runner_v3 import (
            _build_v3_features_at,
        )
        from tradingagents.strategies.v3.models.multi_horizon import consensus_signal
        from tradingagents.strategies.v3.regime.ensemble import detect_regime_v3

        regime = detect_regime_v3(
            prices=self.prices,
            bundle=self.regime_bundle,
            as_of=as_of,
        )

        feat_df = _build_v3_features_at(
            self.prices,
            self.microstructure_features,
            self.derivatives_features,
            as_of,
        )

        if feat_df.empty:
            return QuantSignal(
                coin=coin,
                direction="flat",
                magnitude=0.0,
                regime=regime.label,
                regime_confidence=regime.confidence,
                hurst=regime.hurst,
                deterministic_signals={
                    "v3_quant": True,
                    "v3_changepoint_alert": regime.changepoint_alert,
                },
                as_of_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            )

        # Align feature columns to bundle's expected names
        try:
            probas = self.multi_horizon_bundle.predict_proba(feat_df)
        except Exception:
            logger.exception("V3 predict_proba failed at %s; emitting flat", as_of)
            return QuantSignal(
                coin=coin,
                direction="flat",
                magnitude=0.0,
                regime=regime.label,
                regime_confidence=regime.confidence,
                hurst=regime.hurst,
                deterministic_signals={"v3_quant": True, "v3_predict_failed": True},
                as_of_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
            )

        scalar_probas = {h: float(arr[0]) for h, arr in probas.items()}
        direction_int, confidence = consensus_signal(scalar_probas, regime, self.config)

        if direction_int > 0:
            direction_label: DirectionLabel = "long"
        elif direction_int < 0:
            direction_label = "short"
        else:
            direction_label = "flat"

        magnitude = float(direction_int) * float(confidence)
        magnitude = max(-1.0, min(1.0, magnitude))

        return QuantSignal(
            coin=coin,
            direction=direction_label,
            magnitude=magnitude,
            regime=regime.label,
            regime_confidence=regime.confidence,
            hurst=regime.hurst,
            deterministic_signals={
                "v3_quant": True,
                "v3_horizon_probas": scalar_probas,
                "v3_changepoint_alert": regime.changepoint_alert,
            },
            as_of_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
        )


def build_provider(version: str, **kwargs) -> QuantSignalProvider:
    """Factory: returns a V2 or V3 provider.

    For V2: pass ``base_dir`` (str | None).
    For V3: pass ``prices``, ``regime_bundle``, ``multi_horizon_bundle``,
            ``microstructure_features``, ``derivatives_features``, ``config``.
    """
    version = version.lower()
    if version == "v2":
        return V2QuantSignalProvider(base_dir=kwargs.get("base_dir"))
    if version == "v3":
        required = (
            "prices",
            "regime_bundle",
            "multi_horizon_bundle",
            "microstructure_features",
            "derivatives_features",
            "config",
        )
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"V3QuantSignalProvider missing kwargs: {missing}")
        return V3QuantSignalProvider(**{k: kwargs[k] for k in required})
    raise ValueError(f"Unknown quant version: {version!r} (expected 'v2' or 'v3')")


# ---------------------------------------------------------------------------
# Module-level active version state
# ---------------------------------------------------------------------------

_ACTIVE_QUANT_VERSION: str = "v2"

# Per-coin V3 provider state keyed by coin name.
# Special key "__default__" is used when set_v3_provider_state is called
# without a coin argument (single-coin convenience form).
_V3_PROVIDER_STATES: dict[str, dict] = {}


def set_active_quant_version(version: str) -> None:
    """Set the active quant version. Called at startup by hybrid scripts."""
    global _ACTIVE_QUANT_VERSION
    version = version.lower()
    if version not in ("v2", "v3"):
        raise ValueError(f"Unknown quant version: {version!r}")
    _ACTIVE_QUANT_VERSION = version
    logger.info("Active quant version set to %s", version)


def get_active_quant_version() -> str:
    return _ACTIVE_QUANT_VERSION


def set_v3_provider_state(
    *,
    prices: pd.Series,
    regime_bundle,
    multi_horizon_bundle,
    microstructure_features: pd.DataFrame,
    derivatives_features: pd.DataFrame,
    config,
    coin: Optional[str] = None,
) -> None:
    """Inject per-coin V3 provider state.

    Args:
        coin: Coin identifier to key the state under.  When omitted (or
              ``None``) the state is stored under the ``"__default__"`` key,
              which ``get_active_quant_signal`` falls back to when no
              coin-specific entry exists.  Pass ``coin`` explicitly when
              registering state for multiple coins so each lookup dispatches to
              the correct data.
    """
    key = coin if coin is not None else "__default__"
    _V3_PROVIDER_STATES[key] = {
        "prices": prices,
        "regime_bundle": regime_bundle,
        "multi_horizon_bundle": multi_horizon_bundle,
        "microstructure_features": microstructure_features,
        "derivatives_features": derivatives_features,
        "config": config,
    }
    logger.debug("V3 provider state registered for coin=%r (key=%r)", coin, key)


def clear_v3_provider_state(coin: Optional[str] = None) -> None:
    """Clear V3 provider state.

    Args:
        coin: If provided, remove only the entry for that coin.  If ``None``
              (default), clear all registered states including ``"__default__"``.
    """
    global _V3_PROVIDER_STATES
    if coin is None:
        _V3_PROVIDER_STATES.clear()
    else:
        _V3_PROVIDER_STATES.pop(coin, None)
        # Also remove __default__ when the caller clears without specifying coin=…
        # No — only clear the specific key so partial clear is safe.


def get_active_quant_signal(coin: str, as_of) -> QuantSignal:
    """Dispatch to V2 or V3 based on the active version.

    For V2 the coin argument is passed straight through to the V2 engine.

    For V3 the per-coin state registered via ``set_v3_provider_state`` is used.
    Lookup order: ``coin`` key first, then ``"__default__"``.  Raises
    ``RuntimeError`` if neither is found.

    Args:
        coin: Coin identifier (e.g. ``"bitcoin"``).
        as_of: Trade date as a ``pd.Timestamp`` or ISO date string.
    """
    version = _ACTIVE_QUANT_VERSION
    as_of_ts = pd.Timestamp(as_of)
    if version == "v2":
        return V2QuantSignalProvider(base_dir=None).signal(coin=coin, as_of=as_of_ts)
    if version == "v3":
        state = _V3_PROVIDER_STATES.get(coin) or _V3_PROVIDER_STATES.get("__default__")
        if state is None:
            raise RuntimeError(
                f"V3 state not set for coin={coin!r}; call "
                "set_v3_provider_state(coin=...) at startup"
            )
        provider = V3QuantSignalProvider(**state)
        return provider.signal(coin=coin, as_of=as_of_ts)
    raise ValueError(f"Unknown active quant version: {version!r}")
