"""V3 regime ensemble — combines HMM-3 posterior + Hurst exponent + BOCPD
changepoint flag into a single RegimeState.

Uses the NH-HMM bundle (Task 20) for emission scoring + transition matrix and
the online forward update (Task 19) to track posterior bar-by-bar without
look-ahead bias.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from tradingagents.strategies.v3.contracts import RegimeState
from tradingagents.strategies.v3.regime.hmm_v2 import (
    NHHmmBundle,
    build_regime_features,
    update_posterior,
)


def _rolling_hurst(returns: np.ndarray, window: int) -> float:
    """Rescaled-range Hurst exponent over the last ``window`` returns."""
    if len(returns) < window:
        return 0.5
    series = returns[-window:]
    mean = series.mean()
    centered = series - mean
    cum = np.cumsum(centered)
    R = cum.max() - cum.min()
    S = series.std(ddof=1)
    if S <= 0 or R <= 0:
        return 0.5
    return float(np.log(R / S) / np.log(window))


def _detect_changepoint(
    returns: np.ndarray, window: int, z_threshold: float = 2.5
) -> bool:
    """Crude online changepoint: flag if the last bar's return is more than
    ``z_threshold`` rolling-std deviations from the mean of the prior
    ``window`` bars. Cheap proxy for full BOCPD; sufficient for regime alert.
    """
    if len(returns) < window + 1:
        return False
    prior = returns[-(window + 1) : -1]
    mu = prior.mean()
    sd = prior.std(ddof=1)
    if sd <= 0:
        return False
    z = abs((returns[-1] - mu) / sd)
    return bool(z > z_threshold)


def detect_regime_v3(
    prices: pd.Series,
    bundle: NHHmmBundle,
    as_of: pd.Timestamp,
    hurst_lookback: int = 63,
    bocpd_window: int = 30,
    bocpd_alert_window: int = 5,
    funding_series: Optional[pd.Series] = None,
) -> RegimeState:
    """Compute the current regime state from prices + an NH-HMM bundle.

    Steps:
      1. Slice prices to ``prices.index <= as_of`` (look-ahead guard).
      2. Build smoothed regime features.
      3. Run online forward updates to obtain the bar-level posterior.
      4. Compute rolling Hurst exponent + crude changepoint flag.
      5. Combine: most-likely state → label, posterior mass → base confidence;
         Hurst > 0.55 reinforces directional confidence, < 0.45 dampens it;
         changepoint within last ``bocpd_alert_window`` bars dampens × 0.5.
    """
    prices = prices[prices.index <= as_of]
    if prices.empty:
        raise ValueError("No prices on or before as_of")

    features = build_regime_features(prices)
    if features.empty:
        return RegimeState(
            label="sideways",
            confidence=0.34,
            hurst=0.5,
            changepoint_alert=False,
            posterior={"bull": 1.0 / 3.0, "sideways": 1.0 / 3.0, "bear": 1.0 / 3.0},
        )

    hmm = bundle.hmm
    state_to_label = bundle.state_to_label
    nh_t = bundle.nh_transition

    n_states = bundle.n_states
    posterior = np.full(n_states, 1.0 / n_states)

    # Default to vol covariate from features; funding zero if not provided.
    rv = features["realized_vol"].values
    funding_aligned = (
        funding_series.reindex(features.index).fillna(0.0).values
        if funding_series is not None
        else np.zeros(len(features))
    )

    X = features.values
    emission_logprobs_all = hmm._compute_log_likelihood(X)  # (T, n_states)

    for t in range(len(features)):
        cov = np.array([rv[t], funding_aligned[t]])
        T_mat = nh_t.transition(cov)
        posterior = update_posterior(
            posterior, T_mat, emission_logprobs_all[t]
        )

    # Decode label + posterior dict
    most_likely = int(np.argmax(posterior))
    label = state_to_label.get(most_likely, "sideways")
    posterior_dict_raw = {
        state_to_label.get(i, "sideways"): float(posterior[i])
        for i in range(n_states)
    }
    # Combine duplicates that may occur if multiple HMM states map to the same label
    posterior_dict = {"bull": 0.0, "sideways": 0.0, "bear": 0.0}
    for k, v in posterior_dict_raw.items():
        posterior_dict[k] = posterior_dict.get(k, 0.0) + v
    # Renormalize (paranoia — should already sum to 1)
    total = sum(posterior_dict.values())
    if total > 0:
        posterior_dict = {k: v / total for k, v in posterior_dict.items()}

    base_confidence = float(posterior[most_likely])

    # Hurst + changepoint
    log_returns = np.log(prices / prices.shift(1)).dropna().values
    hurst = _rolling_hurst(log_returns, hurst_lookback)
    cp_window_returns = log_returns[-(bocpd_alert_window + bocpd_window):]
    changepoint_alert = _detect_changepoint(cp_window_returns, bocpd_window)

    # Apply Hurst conditioning: dampen confidence on directional label when
    # market is mean-reverting; reinforce when trending.
    confidence = base_confidence
    if label != "sideways":
        if hurst < 0.45:
            confidence *= 0.5  # mean-reverting market — directional label suspect
        elif hurst > 0.55:
            confidence = min(1.0, confidence * 1.1)  # trending market — reinforce
    if changepoint_alert:
        confidence *= 0.5
    confidence = float(min(1.0, max(0.0, confidence)))

    # Clamp hurst to [0, 1] for the RegimeState field constraint
    hurst = float(min(1.0, max(0.0, hurst)))

    return RegimeState(
        label=label,
        confidence=confidence,
        hurst=hurst,
        changepoint_alert=changepoint_alert,
        posterior=posterior_dict,
    )
