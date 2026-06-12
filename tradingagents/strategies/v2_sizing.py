"""V2 sizing primitives — single source of truth for backtest + live.

Extracted from `scripts/baseline_strategy_v2.py` so the same signal-generation,
volatility, sizing, leverage, position-building, and trend-filter logic can be
shared between the offline backtest script and the live trading cycle.

Functions are reproduced verbatim from the original script (preserving
signatures and behavior). Golden-value tests in
`tests/strategies/test_v2_sizing_golden.py` pin behavior so any future change
is detected.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Signal Generation ────────────────────────────────────────────────


def generate_term_structure_signals(
    df_coin: pd.DataFrame,
    horizons: list[int],
    confidence_ref: float,
    asymmetric: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate consensus signals and confidence for a single coin.

    Symmetric mode: Signal = +1/-1 if ALL horizons agree, else 0.
    Asymmetric mode (default): LONG if longest horizon says UP (easier to
    go long since crypto has structural positive drift). SHORT only if ALL
    horizons agree on DOWN (harder to short — requires full consensus).
    When longest horizon says UP but shorter disagrees, signal is LONG at
    half confidence (fallback).

    Returns (signals, confidence) arrays aligned with df_coin rows.
    """
    n = len(df_coin)
    signals = np.zeros(n)
    confidence = np.zeros(n)
    ref = df_coin["ref_price"].values
    longest_h = max(horizons)

    for i in range(n):
        if ref[i] <= 0 or np.isnan(ref[i]):
            continue

        dirs = []
        ret_magnitudes = []
        for h in horizons:
            pred = df_coin[f"pred_h{h}"].values[i]
            if np.isnan(pred):
                break
            d = 1 if pred > ref[i] else -1
            dirs.append(d)
            ret_magnitudes.append(abs(pred - ref[i]) / ref[i])

        if len(dirs) != len(horizons):
            continue

        avg_ret = np.mean(ret_magnitudes)

        if all(d == dirs[0] for d in dirs):
            # Full agreement
            signals[i] = dirs[0]
            confidence[i] = min(1.0, avg_ret / confidence_ref)
        elif asymmetric:
            # Longest horizon gets final say for longs
            longest_dir = dirs[horizons.index(longest_h)]
            if longest_dir == 1:
                signals[i] = 1
                confidence[i] = min(1.0, avg_ret / confidence_ref) * 0.5
            # For shorts, require full agreement (already handled above)

    return signals, confidence


# ── Volatility, Sizing, Leverage ─────────────────────────────────────


def compute_realized_vol(prices: np.ndarray, lookback: int) -> np.ndarray:
    """Rolling annualized realized volatility from log returns."""
    log_ret = np.full(len(prices), np.nan)
    log_ret[1:] = np.log(prices[1:] / prices[:-1])

    vol = np.full(len(prices), np.nan)
    for i in range(lookback, len(prices)):
        window = log_ret[i - lookback + 1 : i + 1]
        window = window[~np.isnan(window)]
        if len(window) >= 2:
            vol[i] = np.std(window, ddof=1) * np.sqrt(252)
    return vol


def vol_regime_mask(vol: np.ndarray, percentile_cap: float) -> np.ndarray:
    """Return boolean mask: True = OK to trade, False = vol too high."""
    mask = np.ones(len(vol), dtype=bool)
    for i in range(len(vol)):
        if np.isnan(vol[i]):
            mask[i] = False
            continue
        history = vol[:i]
        history = history[~np.isnan(history)]
        if len(history) < 20:
            continue
        threshold = np.quantile(history, percentile_cap)
        if vol[i] > threshold:
            mask[i] = False
    return mask


def vol_targeted_size(
    signal: int, confidence: float, realized_vol: float,
    target_vol: float, kelly_fraction: float,
) -> float:
    """Compute position size using vol targeting + Kelly + confidence."""
    if signal == 0 or np.isnan(realized_vol) or realized_vol <= 0:
        return 0.0
    base = target_vol / realized_vol
    return float(signal) * kelly_fraction * base * confidence


def apply_leverage(base_size: float, confidence: float, max_leverage: float) -> float:
    """Scale position by conditional leverage based on confidence."""
    if base_size == 0:
        return 0.0
    lev = 1 + (max_leverage - 1) * confidence
    sized = base_size * lev
    if abs(sized) > max_leverage:
        sized = np.sign(sized) * max_leverage
    return float(sized)


def build_positions_with_hold(
    signals: np.ndarray,
    vol_ok: np.ndarray,
    confidence: np.ndarray,
    realized_vol: np.ndarray,
    prices: np.ndarray,
    target_vol: float,
    kelly_fraction: float,
    max_leverage: float,
    min_hold: int,
    early_exit_loss: float = 0.015,
) -> np.ndarray:
    """Build position series with exit-only-on-flip + adaptive hold.

    Min hold applies to winning positions. Losing positions can exit early
    (after 3 bars) if cumulative loss exceeds early_exit_loss AND signal
    has flipped or gone flat.
    """
    positions = np.zeros(len(signals))
    current_pos = 0.0
    current_dir = 0
    bars_held = 0
    entry_price = 0.0

    for i in range(len(signals)):
        sig = int(signals[i])

        if current_dir != 0:
            bars_held += 1

        # Check early exit for losers
        if current_dir != 0 and bars_held >= 3 and bars_held < min_hold:
            if entry_price > 0 and prices[i] > 0:
                pnl = current_dir * (prices[i] - entry_price) / entry_price
                signal_changed = (sig != current_dir)
                if pnl < -early_exit_loss and signal_changed:
                    current_pos = 0.0
                    current_dir = 0
                    bars_held = 0

        # Entry from flat
        if current_dir == 0 and sig != 0 and vol_ok[i]:
            base = vol_targeted_size(
                sig, confidence[i], realized_vol[i], target_vol, kelly_fraction,
            )
            current_pos = apply_leverage(base, confidence[i], max_leverage)
            current_dir = sig
            bars_held = 0
            entry_price = prices[i]

        # Flip: only if hold period expired AND signal reversed
        elif (current_dir != 0 and sig != 0 and sig != current_dir
              and bars_held >= min_hold and vol_ok[i]):
            base = vol_targeted_size(
                sig, confidence[i], realized_vol[i], target_vol, kelly_fraction,
            )
            current_pos = apply_leverage(base, confidence[i], max_leverage)
            current_dir = sig
            bars_held = 0
            entry_price = prices[i]

        positions[i] = current_pos

    return positions


def apply_trend_filter(
    positions: np.ndarray,
    prices: np.ndarray,
    sma_period: int,
    multiplier: float,
) -> np.ndarray:
    """Scale positions based on trend alignment with SMA.

    When price > SMA: longs scaled by multiplier, shorts by 1/multiplier.
    When price < SMA: shorts scaled by multiplier, longs by 1/multiplier.
    """
    if sma_period <= 0:
        return positions.copy()

    filtered = positions.copy()
    sma = np.full(len(prices), np.nan)
    for i in range(sma_period - 1, len(prices)):
        sma[i] = np.mean(prices[i - sma_period + 1 : i + 1])

    for i in range(len(positions)):
        if np.isnan(sma[i]) or abs(positions[i]) < 1e-9:
            continue
        if prices[i] > sma[i]:  # uptrend
            if positions[i] > 0:
                filtered[i] = positions[i] * multiplier
            else:
                filtered[i] = positions[i] / multiplier
        else:  # downtrend
            if positions[i] < 0:
                filtered[i] = positions[i] * multiplier
            else:
                filtered[i] = positions[i] / multiplier

    return filtered
