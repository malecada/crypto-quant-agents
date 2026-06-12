"""Deflated Sharpe Ratio (Bailey & López de Prado 2014).

Adjusts an observed Sharpe ratio for selection bias from running multiple
backtests, using extreme-value theory to compute the expected maximum SR under
the null hypothesis of zero true skill.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


_EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max SR | null] across ``n_trials`` independent strategies (zero true SR).

    Uses the closed-form approximation from Bailey & López de Prado 2014:
    ``E[max SR] ≈ sqrt(var_sr) * ((1 - γ) Φ⁻¹(1 - 1/N) + γ Φ⁻¹(1 - 1/(N·e)))``
    where ``γ`` is the Euler–Mascheroni constant.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if var_sr < 0.0:
        raise ValueError("var_sr must be >= 0")
    if n_trials == 1:
        return 0.0
    sigma = math.sqrt(var_sr)
    term_a = (1.0 - _EULER_MASCHERONI) * norm.ppf(1.0 - 1.0 / n_trials)
    term_b = _EULER_MASCHERONI * norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return sigma * (term_a + term_b)


def variance_of_sr(returns: np.ndarray) -> float:
    """Variance of the SR estimator including skew/kurtosis adjustment.

    var(SR) ≈ (1 - γ₃·SR + ((γ₄ - 1)/4)·SR²) / (T - 1)
    where γ₃ = skew, γ₄ = kurtosis (raw, not excess) of returns.
    """
    if len(returns) < 2:
        raise ValueError("returns must have length >= 2")
    mu = float(np.mean(returns))
    sd = float(np.std(returns, ddof=1))
    if sd == 0.0:
        return 0.0
    sr = mu / sd
    skew = float(np.mean((returns - mu) ** 3) / sd**3)
    kurt = float(np.mean((returns - mu) ** 4) / sd**4)
    t = len(returns)
    return (1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr**2) / (t - 1)


def deflated_sharpe_ratio(
    sr_observed: float,
    sr_expected_under_null: float,
    se_sr: float,
) -> float:
    """DSR = Φ((SR_obs − E[max SR | null]) / SE(SR))."""
    if se_sr <= 0.0:
        raise ValueError("se_sr must be > 0")
    z = (sr_observed - sr_expected_under_null) / se_sr
    return float(norm.cdf(z))
