"""Extended V3 feature builders combining V2 OHLC mechanics + TI + cross-asset +
V3 microstructure + Coinglass derivatives + PIT on-chain.

V3-base trains on only 9 features (returns + realized vol + 3-col klines-proxy
microstructure + 2-col Binance funding). BT8 4.5-yr WF (§14) showed V3 SR
-2.71 / -1.10 vs V2 +1.57 / +0.88 — a ΔSR of -4.28 / -1.98. The data layer
extension (§13) added 100+ feature columns to the PIT on-chain builder and
Coinglass-augmented derivatives parquet but none of them flow into V3 training.

This module provides extended builders that compose:
  - V2 OHLC + derived prices (8 cols)
  - V2 rolling MA / vol (5 cols)
  - V2 stockstats technical indicators (14 cols, ``ti_*``)
  - V2 cross-asset (3 cols, ``xa_*``)
  - V2-style price lags (7 cols)
  - V2 calendar dummies (35 cols)
  - V3 microstructure (3 cols)
  - Coinglass-augmented derivatives (27 cols)
  - PIT on-chain features (77+ cols)

All are causal: per-bar slicing enforces ``index <= as_of``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


_LAG_N = 7
_MA_WINDOWS = (7, 14, 30)


def _tz_norm(other_idx: pd.DatetimeIndex, ref_tz) -> pd.DatetimeIndex:
    if ref_tz is not None and other_idx.tz is None:
        return other_idx.tz_localize("UTC")
    if ref_tz is None and other_idx.tz is not None:
        return other_idx.tz_localize(None)
    return other_idx


def _compute_ti(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Stockstats indicators on raw OHLCV (matches model_utils.compute_technical_indicators)."""
    from tradingagents.models.model_utils import compute_technical_indicators

    return compute_technical_indicators(df_ohlcv)


def build_extended_global_features(
    coin: str,
    prices: pd.Series,
    ohlcv: pd.DataFrame,
    microstructure_features: pd.DataFrame,
    derivatives_features: pd.DataFrame,
    btc_prices: pd.Series | None = None,
    eth_prices: pd.Series | None = None,
    btc_volume: pd.Series | None = None,
    include_pit_onchain: bool = True,
    pit_root: Path | None = None,
) -> pd.DataFrame:
    """Build a comprehensive feature DataFrame aligned to ``prices.index``.

    Vectorised over the full series — O(n) compute. Per-bar callers reindex
    the last row out of the result. PIT-safe by construction since rolling
    windows + cross-asset reindex are causal.

    Args:
        coin: CoinGecko id (e.g. ``"bitcoin"``). Used for PIT on-chain lookup.
        prices: Close-price series, DatetimeIndex matching ``ohlcv["Date"]``.
        ohlcv: Raw OHLCV from ``_load_crypto_ohlcv`` (columns Date/Open/High/
            Low/Close/Volume).
        microstructure_features: V3 microstructure parquet
            (``ofi_proxy, ofi_proxy_w, vol_dispersion``). Empty df → zeros.
        derivatives_features: Coinglass-augmented derivatives parquet (27 cols
            including OI OHLC + liquidations + L/S ratios + taker volume +
            funding + basis). Empty df → zeros.
        btc_prices, eth_prices, btc_volume: For cross-asset features. None →
            constant 0/1 fallback.
        include_pit_onchain: Whether to join PIT on-chain feature frame.
        pit_root: Override on-chain store root (defaults to ``data/onchain``).

    Returns:
        DataFrame indexed by ``prices.index`` with ~180 feature columns.
    """
    idx = prices.index
    out = pd.DataFrame(index=idx)

    # ── 1. V2 OHLC + derived prices ────────────────────────────────────
    if "Date" in ohlcv.columns:
        o = ohlcv.set_index("Date").sort_index()
    else:
        o = ohlcv.sort_index()
    o.index = _tz_norm(o.index, idx.tz)
    o = o.reindex(idx)
    out["prices"] = prices.values
    out["open"] = o["Open"].astype(float).values
    out["high"] = o["High"].astype(float).values
    out["low"] = o["Low"].astype(float).values
    out["total_volumes"] = o["Volume"].astype(float).values
    out["daily_return"] = prices.pct_change().values
    out["high_low_spread"] = (out["high"] - out["low"]).values
    out["open_close_spread"] = (out["prices"] - out["open"]).values

    # ── 2. Rolling MAs + stdev ─────────────────────────────────────────
    for w in _MA_WINDOWS:
        out[f"ma_{w}"] = prices.rolling(w).mean().values
        out[f"vol_{w}"] = prices.rolling(w).std().values
    out["vol_ma_7"] = pd.Series(out["total_volumes"].values).rolling(7).mean().values
    out["vol_ma_30"] = pd.Series(out["total_volumes"].values).rolling(30).mean().values

    # ── 3. Stockstats technical indicators (ti_*) ──────────────────────
    ohlcv_for_ti = o.reset_index()
    if "Date" not in ohlcv_for_ti.columns:
        ohlcv_for_ti = ohlcv_for_ti.rename(columns={ohlcv_for_ti.columns[0]: "Date"})
    ti = _compute_ti(ohlcv_for_ti)
    if not ti.empty:
        ti.index = idx
        for c in ti.columns:
            out[c] = ti[c].astype(float).values

    # ── 4. Cross-asset (xa_*) ──────────────────────────────────────────
    if btc_prices is not None and not btc_prices.empty:
        btc_p = btc_prices.copy()
        btc_p.index = _tz_norm(btc_p.index, idx.tz)
        out["xa_btc_return"] = btc_p.reindex(idx).pct_change().fillna(0.0).values
    else:
        out["xa_btc_return"] = 0.0
    if eth_prices is not None and btc_prices is not None and not eth_prices.empty:
        eth_p = eth_prices.copy(); eth_p.index = _tz_norm(eth_p.index, idx.tz)
        btc_p = btc_prices.copy(); btc_p.index = _tz_norm(btc_p.index, idx.tz)
        ratio = eth_p.reindex(idx) / btc_p.reindex(idx).replace(0.0, np.nan)
        out["xa_eth_btc_ratio"] = ratio.ffill().fillna(1.0).values
    else:
        out["xa_eth_btc_ratio"] = 1.0
    if btc_volume is not None and not btc_volume.empty:
        bv = btc_volume.copy(); bv.index = _tz_norm(bv.index, idx.tz)
        out["xa_btc_dom"] = bv.reindex(idx).fillna(0.0).values
    else:
        out["xa_btc_dom"] = 0.0

    # ── 5. V3 microstructure (ofi_proxy*, vol_dispersion) ──────────────
    micro_cols = ["ofi_proxy", "ofi_proxy_w", "vol_dispersion"]
    if not microstructure_features.empty:
        m = microstructure_features.copy()
        m.index = _tz_norm(m.index, idx.tz)
        for c in micro_cols:
            if c in m.columns:
                out[c] = m[c].reindex(idx, method="ffill").astype(float).values
            else:
                out[c] = 0.0
    else:
        for c in micro_cols:
            out[c] = 0.0

    # ── 6. Coinglass-augmented derivatives (all 27 cols) ───────────────
    if not derivatives_features.empty:
        d = derivatives_features.copy()
        d.index = _tz_norm(d.index, idx.tz)
        for c in d.columns:
            if c in ("perp_price", "spot_price"):
                continue  # raw prices already captured via OHLCV
            out[f"deriv_{c}"] = d[c].reindex(idx, method="ffill").astype(float).values

    # ── 7. PIT on-chain features (~77 cols, oc_* and derived) ──────────
    if include_pit_onchain:
        try:
            from tradingagents.dataflows.onchain_features import (
                build_pit_onchain_features,
                onchain_store,
            )
            root = pit_root if pit_root is not None else onchain_store.DEFAULT_ROOT
            # Get tz-aware dates for the builder
            pit_dates = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
            pit = build_pit_onchain_features(
                coin=coin,
                dates=pit_dates,
                include_global=True,
                include_derived=True,
                include_stablecoin_context=True,
                include_options=True,
                include_derivatives=False,  # already done above to avoid double-prefix
                root=root,
            )
            if not pit.empty:
                # Reindex pit to caller's idx (tz-strip if needed)
                pit_idx = pit.index
                if pit_idx.tz is not None and idx.tz is None:
                    pit.index = pit_idx.tz_convert("UTC").tz_localize(None)
                pit = pit.reindex(idx)
                for c in pit.columns:
                    if c in out.columns:
                        continue  # avoid duplicates (e.g. oc_funding_rate)
                    out[c] = pit[c].astype(float).values
        except Exception as e:
            logger.warning("PIT on-chain features unavailable: %s", e)

    # ── 8. Price lags (lag1..lag7) ─────────────────────────────────────
    p_arr = prices.values
    for k in range(1, _LAG_N + 1):
        out[f"lag{k}"] = pd.Series(p_arr).shift(k).values

    # ── 9. Calendar features ───────────────────────────────────────────
    cal_idx = idx.tz_convert("UTC") if idx.tz is not None else idx
    out["Day"] = cal_idx.day.values.astype(int)
    out["Month"] = cal_idx.month.values.astype(int)
    out["Year"] = cal_idx.year.values.astype(int)
    # Day-of-month dummies (day_1 .. day_31) — matches V2 data_transform
    day_arr = cal_idx.day.values.astype(int)
    for d in range(1, 32):
        out[f"day_{d}"] = (day_arr == d).astype(int)

    # ── 10. Shift everything by 1 to prevent same-day leakage ─────────
    # V2's data_transform does .shift(1) on the full frame: row at t holds
    # values observed strictly before t. Mirror that here so V3 sees the same
    # causal structure V2 LGB sees in production.
    shifted = out.shift(1)
    shifted = shifted.ffill().fillna(0.0)

    # Replace inf with 0 (from ratios with zero denominators)
    shifted = shifted.replace([np.inf, -np.inf], 0.0)
    return shifted


def build_extended_features_at(
    global_features: pd.DataFrame,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Return the single-row feature vector at ``as_of`` from a pre-built frame."""
    if as_of not in global_features.index:
        # Fall back to the most recent row <= as_of
        sub = global_features[global_features.index <= as_of]
        if sub.empty:
            return pd.DataFrame()
        return sub.iloc[[-1]]
    return global_features.loc[[as_of]]
