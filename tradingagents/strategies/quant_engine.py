"""Layer 1 quant engine — single-day inference wrapping precomputed LGB CSVs.

Reads ``data/multi_2coins_v2/preds_lgb_h{7,14}.csv`` (or ``multi_3coins_*``
for altcoins via the "2+1" pooling pattern), looks up the row for the
target ``(coin, date)``, runs ``generate_term_structure_signals`` from
``v2_sizing`` to produce direction + magnitude, then composes with
``detect_regime`` into a ``QuantSignal``.

We deliberately do NOT train LGB on the fly per call — the precomputed
walk-forward CSVs already have correct PIT boundaries from
``evaluate_models_multi.py`` and per-call training would re-introduce
look-ahead risk.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.strategies.contracts import DirectionLabel, QuantSignal
from tradingagents.strategies.deterministic_signals import compute_deterministic_pack
from tradingagents.strategies.regime import detect_regime
from tradingagents.strategies.v2_sizing import generate_term_structure_signals

logger = logging.getLogger(__name__)

_HORIZONS = [7, 14]
_CONFIDENCE_REF = 0.05  # 5% expected return → confidence=1.0 (matches baseline V2 default)


def _candidate_pred_dirs(coin: str, base_dir: Optional[str] = None) -> list[str]:
    """Search order for precomputed LGB pools.

    Major coins (BTC/ETH) are in the 2-coin pool. Altcoins live in their
    "2+1" 3-coin pools. ``base_dir`` overrides the config default.
    """
    cfg = get_config() if base_dir is None else {"quant_pred_dir": base_dir}
    primary = cfg.get("quant_pred_dir", "data/multi_2coins_v2")
    candidates = [primary]
    # Common altcoin-specific 3-coin pools
    altcoin_pools = {
        "binancecoin": "data/multi_3coins_bnb",
        "solana": "data/multi_3coins_sol",
        "ripple": "data/multi_3coins_xrp",
        "cardano": "data/multi_3coins_ada",
    }
    if coin in altcoin_pools:
        candidates.insert(0, altcoin_pools[coin])
    return candidates


def _load_pred_row(coin: str, date: str, base_dir: Optional[str] = None) -> Optional[dict]:
    """Find the row for (coin, date) in the precomputed LGB CSVs.

    Returns ``{"ref_price": float, "pred_h7": float, "pred_h14": float}``
    or ``None`` if not found.
    """
    target_date = pd.to_datetime(date).normalize()
    for pred_dir in _candidate_pred_dirs(coin, base_dir):
        path7 = os.path.join(pred_dir, "preds_lgb_h7.csv")
        path14 = os.path.join(pred_dir, "preds_lgb_h14.csv")
        if not (os.path.exists(path7) and os.path.exists(path14)):
            continue
        df7 = pd.read_csv(path7, parse_dates=["date"])
        df14 = pd.read_csv(path14, parse_dates=["date"])
        df7["date"] = pd.to_datetime(df7["date"]).dt.tz_localize(None).dt.normalize()
        df14["date"] = pd.to_datetime(df14["date"]).dt.tz_localize(None).dt.normalize()
        df7 = df7[df7["coin_id"] == coin]
        df14 = df14[df14["coin_id"] == coin]
        row7 = df7[df7["date"] == target_date]
        row14 = df14[df14["date"] == target_date]
        if row7.empty or row14.empty:
            continue
        return {
            "ref_price": float(row7["ref_price"].iloc[0]),
            "pred_h7": float(row7["prediction"].iloc[0]),
            "pred_h14": float(row14["prediction"].iloc[0]),
        }
    return None


def _direction_label(s: float) -> DirectionLabel:
    if s > 0:
        return "long"
    if s < 0:
        return "short"
    return "flat"


def get_quant_signal(
    coin: str,
    date: str,
    base_dir: Optional[str] = None,
) -> QuantSignal:
    """Return a Layer 1 ``QuantSignal`` for ``(coin, date)``.

    Reads precomputed LGB predictions, applies
    ``generate_term_structure_signals``, then composes with the regime
    detector. ``deterministic_signals`` is left empty here — Phase 2
    populates it.
    """
    pred = _load_pred_row(coin, date, base_dir)
    pack = compute_deterministic_pack(coin, date)
    if pred is None:
        logger.warning(
            f"no LGB prediction for {coin} @ {date}; emitting flat QuantSignal"
        )
        regime, regime_conf, hurst = detect_regime(coin, date)
        return QuantSignal(
            coin=coin,
            direction="flat",
            magnitude=0.0,
            regime=regime,
            regime_confidence=regime_conf,
            hurst=hurst,
            deterministic_signals=pack,
            as_of_date=date,
        )

    df_row = pd.DataFrame(
        [{
            "ref_price": pred["ref_price"],
            "pred_h7": pred["pred_h7"],
            "pred_h14": pred["pred_h14"],
        }]
    )
    signals, confidence = generate_term_structure_signals(
        df_row, _HORIZONS, _CONFIDENCE_REF, asymmetric=True
    )
    s = float(signals[0])
    c = float(confidence[0])

    # Magnitude: signed confidence ∈ [-1, 1]
    magnitude = s * c

    regime, regime_conf, hurst = detect_regime(coin, date)
    pack.update({
        "lgb_h7": pred["pred_h7"],
        "lgb_h14": pred["pred_h14"],
        "ref_price": pred["ref_price"],
        "lgb_confidence": c,
    })
    return QuantSignal(
        coin=coin,
        direction=_direction_label(s),
        magnitude=magnitude,
        regime=regime,
        regime_confidence=regime_conf,
        hurst=hurst,
        deterministic_signals=pack,
        as_of_date=date,
    )
