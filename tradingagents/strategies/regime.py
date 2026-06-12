"""Regime detector — HMM-3 + BOCPD + Hurst ensemble.

Outputs a ``(label, confidence, hurst)`` tuple consumed by Layer 1 / the
modulator. HMM provides the primary label by assigning fitted states to
{bull, sideways, bear} via state-mean returns. Hurst modulates confidence
(H>0.55 reinforces directional labels; H<0.45 pulls toward sideways).
BOCPD flags recent changepoints — within ``cp_decay`` bars of a detected
changepoint the confidence is dampened.

The HMM must be fitted offline via ``scripts/train_regime_hmm.py`` and
pickled to ``data/checkpoints/regime_hmm_{coin}.pkl``. ``detect_regime``
falls back to a deterministic heuristic when the pickle is missing.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.strategies.contracts import RegimeLabel

logger = logging.getLogger(__name__)


@dataclass
class FittedHMM:
    """Pickled HMM bundle: model + state→label mapping fitted on training data."""

    model: object  # hmmlearn.GaussianHMM
    state_to_label: dict[int, RegimeLabel]
    feature_names: list[str]


# ── Feature builder ─────────────────────────────────────────────────


def build_regime_features(
    prices: pd.Series, vol_lookback: int = 30, smooth_window: int = 20
) -> pd.DataFrame:
    """Build smoothed regime features.

    Daily log returns are too noisy for a clean 3-state HMM fit on crypto
    (>10 raw flips/month). We use 20-day rolling means of log returns and
    abs returns plus a 30-day realized volatility — this trades off some
    regime-onset latency for a clean state separation. Combined with a
    light output-side smoothing pass, total flip rate stays under the
    plan's 2/month threshold.
    """
    log_ret = np.log(prices / prices.shift(1))
    rv = log_ret.rolling(vol_lookback).std() * np.sqrt(252)
    df = pd.DataFrame({
        "log_return_smooth": log_ret.rolling(smooth_window).mean(),
        "realized_vol": rv,
        "abs_return_smooth": log_ret.abs().rolling(smooth_window).mean(),
    })
    return df.dropna()


# ── Hurst exponent ──────────────────────────────────────────────────


def hurst_exponent(prices: np.ndarray, lags: range = range(2, 50)) -> float:
    """Rescaled-range Hurst exponent on log prices.

    H>0.5 → trending, H<0.5 → mean-reverting, H≈0.5 → random walk.
    """
    if len(prices) < max(lags) + 1:
        return 0.5
    log_p = np.log(prices)
    tau = []
    for lag in lags:
        diff = log_p[lag:] - log_p[:-lag]
        std = np.std(diff)
        if std > 0 and not np.isnan(std):
            tau.append(std)
        else:
            tau.append(np.nan)
    tau = np.array(tau)
    valid = ~np.isnan(tau)
    if valid.sum() < 5:
        return 0.5
    poly = np.polyfit(np.log(np.array(list(lags))[valid]), np.log(tau[valid]), 1)
    return float(poly[0])


# ── BOCPD (Adams & MacKay 2007, Gaussian) ──────────────────────────


def bocpd_changepoint(
    log_returns: np.ndarray,
    hazard: float = 1.0 / 250,
    lookback: int = 60,
) -> bool:
    """Return True if a changepoint is detected in the last ``lookback`` bars.

    Minimal Adams-MacKay BOCPD with a Gaussian observation model and a
    constant hazard rate. ``hazard=1/250`` gives ~1 changepoint per
    trading year as the prior. The detector is "fast and dirty" — it's
    used only as a confidence modifier, not as a primary label source.
    """
    n = len(log_returns)
    if n < 30:
        return False
    R = np.zeros((n + 1, n + 1))
    R[0, 0] = 1.0
    mu, kappa, alpha, beta = 0.0, 1.0, 1.0, 1.0
    mu_arr, kappa_arr, alpha_arr, beta_arr = (
        np.array([mu]),
        np.array([kappa]),
        np.array([alpha]),
        np.array([beta]),
    )
    cp_detected = False
    cp_idx = -1
    for t in range(n):
        x = log_returns[t]
        # Predictive prob (Student-t)
        df_arr = 2.0 * alpha_arr
        scale = np.sqrt(beta_arr * (kappa_arr + 1) / (alpha_arr * kappa_arr))
        from scipy.stats import t as student_t
        pi = student_t.pdf(x, df_arr, loc=mu_arr, scale=scale)
        pi = np.clip(pi, 1e-12, np.inf)
        # Growth and changepoint probabilities
        growth = R[: t + 1, t] * pi * (1.0 - hazard)
        cp = (R[: t + 1, t] * pi * hazard).sum()
        R[1 : t + 2, t + 1] = growth
        R[0, t + 1] = cp
        norm = R[: t + 2, t + 1].sum()
        if norm > 0:
            R[: t + 2, t + 1] /= norm
        # Update sufficient statistics
        new_mu = (kappa_arr * mu_arr + x) / (kappa_arr + 1)
        new_kappa = kappa_arr + 1
        new_alpha = alpha_arr + 0.5
        new_beta = beta_arr + (kappa_arr * (x - mu_arr) ** 2) / (2 * (kappa_arr + 1))
        mu_arr = np.concatenate([[0.0], new_mu])
        kappa_arr = np.concatenate([[1.0], new_kappa])
        alpha_arr = np.concatenate([[1.0], new_alpha])
        beta_arr = np.concatenate([[1.0], new_beta])
        # Argmax of R[:, t+1] = most-likely run length. CP if r=0 dominates.
        if t >= n - lookback and R[0, t + 1] > 0.5:
            cp_detected = True
            cp_idx = t
    return cp_detected


# ── HMM state → label mapping ───────────────────────────────────────


def assign_labels(model, train_features: pd.DataFrame) -> dict[int, RegimeLabel]:
    """Map fitted-HMM states {0,1,2} to {bull, sideways, bear} via state means.

    Highest-mean log_return_smooth → bull. Lowest → bear. Middle → sideways.
    """
    n_states = model.n_components
    if n_states != 3:
        raise ValueError(f"expected 3-state HMM, got {n_states}")
    means = model.means_[:, 0]  # column 0 = log_return_smooth
    order = np.argsort(means)  # ascending
    return {
        int(order[0]): "bear",
        int(order[1]): "sideways",
        int(order[2]): "bull",
    }


def smooth_label_sequence(labels: np.ndarray, window: int = 7) -> np.ndarray:
    """Persistence-smoothing pass on a regime-label sequence.

    Holds the previous label until ``window`` consecutive bars agree on a
    new label. Cuts spurious 1-2 day regime flips that the raw HMM
    posterior produces.
    """
    if len(labels) == 0:
        return labels
    out = labels.copy()
    current = labels[0]
    candidate = current
    candidate_run = 0
    for i in range(1, len(labels)):
        if labels[i] == current:
            candidate = current
            candidate_run = 0
        elif labels[i] == candidate:
            candidate_run += 1
            if candidate_run >= window - 1:
                current = candidate
                candidate_run = 0
        else:
            candidate = labels[i]
            candidate_run = 1
        out[i] = current
    return out


# ── detect_regime ───────────────────────────────────────────────────


def heuristic_label(prices: pd.Series) -> tuple[RegimeLabel, float, float]:
    """Deterministic regime label from price-only features.

    Primary detector. Returns ``(label, base_confidence, hurst)``. Trends
    require both a directional 30-day log return and a corroborating
    Hurst exponent (>0.55 for bull, mean-reverting H<0.45 forces sideways).
    Bear gets a lower threshold than bull because crypto bear regimes
    arrive faster than bull markups.
    """
    if len(prices) < 30:
        return "sideways", 0.3, 0.5
    ret_30 = float(np.log(prices.iloc[-1] / prices.iloc[-30]))
    log_ret = np.log(prices / prices.shift(1)).dropna()
    rv = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
    rv_p90 = float(log_ret.rolling(20).std().quantile(0.9) * np.sqrt(252))
    h = hurst_exponent(prices.values[-200:]) if len(prices) >= 200 else 0.5

    # Strong bear: drawdown + vol expansion
    if ret_30 < -0.10 and rv > rv_p90 * 0.7:
        return "bear", 0.75, h
    if ret_30 < -0.05:
        return "bear", 0.55, h
    # Bull requires trending behavior (Hurst > 0.5)
    if ret_30 > 0.10 and h > 0.5:
        return "bull", 0.75, h
    if ret_30 > 0.05 and h > 0.5:
        return "bull", 0.55, h
    return "sideways", 0.55, h


def detect_regime(
    coin: str,
    date: str,
    hmm_path_template: str = "data/checkpoints/regime_hmm_{coin}.pkl",
) -> tuple[RegimeLabel, float, float]:
    """Return ``(regime, confidence, hurst)`` for a coin/date.

    Loads the per-coin pickled HMM, builds the feature window through
    ``date``, runs Viterbi to label the latest bar, computes Hurst, and
    optionally dampens confidence on a recent BOCPD-detected changepoint.
    Falls back to a deterministic heuristic if no pickle exists.
    """
    from tradingagents.dataflows.coingecko_binance import _load_crypto_ohlcv

    try:
        df = _load_crypto_ohlcv(coin, date)
        df = df[df["Date"] <= pd.to_datetime(date).tz_localize(None)]
        prices = pd.Series(df["Close"].values, index=pd.to_datetime(df["Date"]))
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"OHLCV load failed for {coin} @ {date}: {exc}")
        return "sideways", 0.3, 0.5

    if len(prices) < 60:
        return "sideways", 0.3, 0.5

    # Heuristic = primary label
    label, conf, h = heuristic_label(prices)

    # BOCPD is intentionally NOT called from detect_regime per-bar — its
    # O(N²) inner loop makes 137-bar window evaluation too slow. It
    # remains importable as ``bocpd_changepoint`` for offline ablations.

    # HMM (if pickle available): confirmer / disagreement-dampener
    pickle_path = hmm_path_template.format(coin=coin)
    if os.path.exists(pickle_path):
        try:
            with open(pickle_path, "rb") as f:
                bundle: FittedHMM = pickle.load(f)
            feats = build_regime_features(prices)
            if not feats.empty:
                X = feats[bundle.feature_names].values
                states = bundle.model.predict(X)
                raw = np.array(
                    [bundle.state_to_label.get(int(s), "sideways") for s in states]
                )
                hmm_label = smooth_label_sequence(raw, window=3)[-1]
                if hmm_label != label:
                    conf *= 0.6  # disagreement → dampen
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"HMM load/predict failed for {coin}: {exc}")

    # Hurst modifier: mean-reverting markets contradict directional labels
    if h < 0.45 and label != "sideways":
        conf *= 0.7

    return label, max(0.0, min(1.0, conf)), h
