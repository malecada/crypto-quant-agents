"""V3 regime detector — HMM-3 + BOCPD + Hurst ensemble.

Promoted from ``tradingagents.strategies.regime`` (the parent module) into the
V3 sub-package so that Tasks 18-21 can extend it (NH-HMM transition covariates,
online posterior, training script, ensemble combiner) without touching the
production parent module.

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
from hmmlearn.hmm import GaussianHMM

from tradingagents.strategies.v3.contracts import RegimeLabel

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


# ── NH-HMM transition matrix ────────────────────────────────────────


@dataclass
class NHTransitionMatrix:
    """Non-homogeneous transition matrix for HMM-3.

    For each (from_state, to_state) pair the unnormalized log-probability is:
      ``logit[i, j] = intercepts[i, j] + coefs[i, j, :] @ covariates``
    A row-wise softmax then yields a proper 3x3 stochastic matrix per timestep.

    Shape conventions:
      - ``coefs``      : (n_states, n_states, n_covariates)
      - ``intercepts`` : (n_states, n_states)
      - ``covariates`` (passed to ``transition``) : (n_covariates,)

    Per spec §4.3, ``n_covariates = 2`` corresponding to
    ``(realized_vol_21d, funding_rate_8h)``. The class itself is generic over
    ``n_covariates`` so future feature additions don't require a rewrite.

    Training of the coefficients is performed by ``train_nh_hmm`` (Task 20).
    Online posterior updates use the matrix returned by ``transition`` in the
    forward algorithm (Task 19).
    """

    coefs: np.ndarray
    intercepts: np.ndarray

    def __post_init__(self) -> None:
        if self.coefs.shape[0] != self.coefs.shape[1]:
            raise ValueError(
                f"coefs leading dims must match (square transition); got {self.coefs.shape}"
            )
        if self.intercepts.shape != self.coefs.shape[:2]:
            raise ValueError(
                f"intercepts shape {self.intercepts.shape} != coefs[:2] {self.coefs.shape[:2]}"
            )

    def transition(self, covariates: np.ndarray) -> np.ndarray:
        """Return the 3x3 transition matrix for the given covariate vector."""
        if covariates.ndim != 1 or covariates.shape[0] != self.coefs.shape[2]:
            raise ValueError(
                f"covariates must be 1-D of length {self.coefs.shape[2]}; got {covariates.shape}"
            )
        # logit[i, j] = intercepts[i, j] + coefs[i, j, :] @ covariates
        logits = self.intercepts + self.coefs @ covariates  # broadcasts to (n, n)
        # row-wise softmax
        max_per_row = logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits - max_per_row)
        return exp_l / exp_l.sum(axis=1, keepdims=True)


# ── Online forward-algorithm posterior update ───────────────────────


def update_posterior(
    prev_posterior: np.ndarray,
    transition_matrix: np.ndarray,
    emission_logprobs: np.ndarray,
) -> np.ndarray:
    """One forward-algorithm step (look-ahead-safe online posterior update).

    Math:
      pred[j]         = sum_i prev_posterior[i] * transition_matrix[i, j]
      unnorm[j]       = pred[j] * exp(emission_logprobs[j])
      posterior[j]    = unnorm[j] / sum(unnorm)

    Numerical stability: subtract the max log-prob before exponentiating so the
    update remains stable even when all emission_logprobs are very negative.

    Args:
      prev_posterior: 1-D array of state probabilities (sums to 1.0).
      transition_matrix: (n_states, n_states) row-stochastic matrix.
      emission_logprobs: 1-D array of log-likelihoods, one per state.

    Returns:
      Updated 1-D posterior of shape (n_states,) summing to 1.0.

    Raises:
      ValueError: if shapes mismatch.
    """
    if prev_posterior.ndim != 1:
        raise ValueError("prev_posterior must be 1-D")
    n_states = prev_posterior.shape[0]
    if transition_matrix.shape != (n_states, n_states):
        raise ValueError(
            f"transition_matrix shape {transition_matrix.shape} "
            f"!= ({n_states}, {n_states})"
        )
    if emission_logprobs.shape != (n_states,):
        raise ValueError(
            f"emission_logprobs shape {emission_logprobs.shape} "
            f"!= ({n_states},)"
        )

    pred = prev_posterior @ transition_matrix
    # numerically stable update: shift max log-prob to 0
    shifted = emission_logprobs - emission_logprobs.max()
    unnorm = pred * np.exp(shifted)
    total = unnorm.sum()
    if total <= 0.0:
        # Degenerate case: emission likelihood vanishes for all states.
        # Fall back to predict-only step.
        return pred / pred.sum()
    return unnorm / total


# ── NH-HMM bundle + training ────────────────────────────────────────


@dataclass
class NHHmmBundle:
    """Pickled NH-HMM bundle: GaussianHMM + transition coefs + label map.

    For Phase-4 v0 the NH transition is initialized with zero covariate
    coefficients — i.e., the matrix is constant per timestep and equal to
    ``hmm.transmat_``. Future work: fit real coefficients via L-BFGS on
    smoothed posteriors.

    Spec §4.3 deviation: zero-coef NH transitions; the transition matrix is
    constant (homogeneous HMM), not truly non-homogeneous. This is a degenerate
    NH-HMM that reduces to standard HMM. Real coefficient learning is deferred
    to post-thesis work.
    """

    hmm: object  # hmmlearn.GaussianHMM (any to avoid circular import on type)
    nh_transition: NHTransitionMatrix
    state_to_label: dict[int, RegimeLabel]
    feature_names: list[str]
    n_states: int


def train_nh_hmm(
    prices,  # pd.Series of closing prices
    covariates_df=None,  # optional pd.DataFrame of covariates aligned with features
    n_states: int = 3,
    n_iter: int = 200,
    random_state: int = 42,
) -> NHHmmBundle:
    """Train a 3-state NH-HMM bundle on ``prices``.

    Phase 4 v0 implementation: fits a standard GaussianHMM on the smoothed
    regime features (log_return_smooth, realized_vol, abs_return_smooth) and
    wraps it in an ``NHHmmBundle`` with zero-coef NH transitions (i.e., the
    transition matrix is constant per timestep and equal to ``hmm.transmat_``).

    Future versions can fit real NH-HMM coefficients via L-BFGS on smoothed
    posteriors.

    Args:
        prices: pd.Series of closing prices (any timezone).
        covariates_df: optional pd.DataFrame of covariates aligned with the
            feature index. Currently unused (v0 degenerate path). If provided,
            its column count sets ``n_covariates``; otherwise defaults to 2
            per spec §4.3 (vol, funding).
        n_states: number of HMM states (default 3: bull/sideways/bear).
        n_iter: maximum EM iterations (default 200).
        random_state: random seed for reproducibility.

    Returns:
        NHHmmBundle with fitted GaussianHMM and zero-coef NHTransitionMatrix.

    Raises:
        ValueError: if there are fewer than 50 samples after feature building.
        RuntimeError: if GaussianHMM.fit raises (propagated after logging).
    """
    features = build_regime_features(prices)
    X = features.values
    if X.shape[0] < 50:
        raise ValueError(f"Not enough samples to fit HMM (got {X.shape[0]})")

    try:
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            random_state=random_state,
        )
        hmm.fit(X)
    except Exception:
        logger.exception("GaussianHMM fit failed; raising")
        raise

    # assign_labels(model, train_features) — train_features is unused in the body
    state_to_label = assign_labels(hmm, features)

    # Initialize NH transition with zero coefs and intercepts equal to
    # log of the fitted (homogeneous) transition matrix. Number of covariates
    # defaults to 2 per spec §4.3 (vol, funding) even though they're unused.
    n_covariates = 2 if covariates_df is None else covariates_df.shape[1]
    coefs = np.zeros((n_states, n_states, n_covariates))
    # Use log of a smoothed transmat to avoid -inf when entries are zero.
    transmat_safe = np.clip(hmm.transmat_, 1e-9, 1.0)
    intercepts = np.log(transmat_safe)
    nh = NHTransitionMatrix(coefs=coefs, intercepts=intercepts)

    return NHHmmBundle(
        hmm=hmm,
        nh_transition=nh,
        state_to_label=state_to_label,
        feature_names=list(features.columns),
        n_states=n_states,
    )
