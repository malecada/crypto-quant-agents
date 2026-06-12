"""Single-step V2 sizing for the live cycle.

Wraps tradingagents.strategies.v2_sizing primitives and applies them to one
coin's most recent prediction + the rolling price history. Returns a SizingResult
with all components needed for journal logging.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tradingagents.strategies.v2_sizing import (
    apply_leverage,
    apply_trend_filter,
    compute_realized_vol,
    generate_term_structure_signals,
    vol_regime_mask,
    vol_targeted_size,
)


@dataclass
class SizingResult:
    coin: str
    signal: int
    confidence: float
    realized_vol: float
    base_size: float
    leverage: float
    sma_multiplier: float
    final_size_notional: float
    vol_ok: bool
    dirs_per_horizon: dict[int, int] | None = None


def bars_through(price_history, asof):
    """Return only bars dated on or before ``asof`` (drops the in-progress bar).

    The cycle fires shortly after 00:00 UTC, so the OHLCV cache may carry
    today's daily bar with only minutes of data. Computing realized vol / SMA
    on that partial bar corrupts the vol denominator and the trend multiplier
    (P4). Slicing to ``asof`` (yesterday's complete close — the same vintage the
    prediction uses) keeps the sizing inputs self-consistent and complete.

    ``asof`` is an ISO date string (``"YYYY-MM-DD"``) or anything pandas can
    coerce to a Timestamp. A history without a ``date`` column is returned
    unchanged.
    """
    if price_history is None or len(price_history) == 0 or "date" not in getattr(price_history, "columns", []):
        return price_history
    d = pd.to_datetime(price_history["date"]).dt.normalize()
    cutoff = pd.Timestamp(asof).normalize()
    return price_history[d <= cutoff]


def target_position_qty(
    *, size_fraction: float, portfolio_value: float,
    weight: float, ref_price: float,
) -> float:
    """Convert a per-coin size fraction to a signed target quantity.

    ``size_fraction`` is ``SizingResult.final_size_notional`` — a fraction of
    equity on a full-portfolio basis (matching one of the backtest's per-coin
    sleeves, bounded by ``max_leverage``). ``weight`` is the coin's
    renormalized portfolio weight (see ``config.compute_portfolio_weights``).
    Folding the weight in here reproduces ``baseline_v5_mix.portfolio_return``:
    the live book allocates ``weight * equity`` to each coin's sleeve rather
    than the full equity, so an N-coin shared-margin account no longer runs
    ~N x the validated gross exposure.
    """
    if ref_price <= 0:
        return 0.0
    return size_fraction * portfolio_value * weight / ref_price


def compute_size(
    *, coin, prediction, price_history,
    horizons, symmetric,
    target_vol, kelly_fraction, max_leverage,
    vol_lookback, vol_cap_pct, confidence_ref,
    trend_sma, trend_multiplier,
) -> SizingResult:
    df_coin = pd.DataFrame({
        "ref_price": [prediction["ref_price"]],
        **{f"pred_h{h}": [prediction[f"pred_h{h}"]] for h in horizons},
    })
    signals, conf = generate_term_structure_signals(
        df_coin, horizons=horizons, confidence_ref=confidence_ref,
        asymmetric=not symmetric,
    )
    signal = int(signals[0])
    confidence = float(conf[0])

    # Per-horizon directions (decoupled from the consensus): each prediction
    # is +1 if pred>ref, -1 if pred<ref, 0 if equal/missing. Lets the journal
    # record signal_h7 / signal_h14 distinctly from the consensus_signal.
    dirs_per_horizon: dict[int, int] = {}
    for h in horizons:
        p = prediction[f"pred_h{h}"]
        ref = prediction["ref_price"]
        if pd.isna(p) or pd.isna(ref):
            dirs_per_horizon[h] = 0
        else:
            dirs_per_horizon[h] = 1 if p > ref else (-1 if p < ref else 0)

    prices = price_history.sort_values("date")["close"].values
    vol_series = compute_realized_vol(prices, lookback=vol_lookback)
    realized_vol = float(vol_series[-1]) if len(vol_series) and not np.isnan(vol_series[-1]) else float("nan")
    mask = vol_regime_mask(vol_series, percentile_cap=vol_cap_pct)
    vol_ok = bool(mask[-1]) if len(mask) else False

    if not vol_ok or signal == 0:
        return SizingResult(coin=coin, signal=signal, confidence=confidence,
                             realized_vol=realized_vol, base_size=0.0, leverage=0.0,
                             sma_multiplier=1.0, final_size_notional=0.0, vol_ok=vol_ok,
                             dirs_per_horizon=dirs_per_horizon)

    base = vol_targeted_size(signal, confidence, realized_vol, target_vol, kelly_fraction)
    sized = apply_leverage(base, confidence, max_leverage)

    # apply_trend_filter requires the full price history to compute SMA over the
    # final `trend_sma` bars; we build a positions array where only the last
    # element holds our sized position and read the filtered last element back.
    pos_arr = np.zeros(len(prices))
    pos_arr[-1] = sized
    filtered = apply_trend_filter(
        pos_arr, np.asarray(prices), sma_period=trend_sma,
        multiplier=trend_multiplier,
    )
    final_size = float(filtered[-1])
    sma_mult = final_size / sized if abs(sized) > 1e-9 else 1.0
    return SizingResult(
        coin=coin, signal=signal, confidence=confidence, realized_vol=realized_vol,
        base_size=base, leverage=sized, sma_multiplier=sma_mult,
        final_size_notional=final_size, vol_ok=True,
        dirs_per_horizon=dirs_per_horizon,
    )
