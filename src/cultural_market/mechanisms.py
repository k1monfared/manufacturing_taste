"""Core mechanism functions for cultural market simulation.

Implements:
- Mere Exposure Effect (MEE) - inverted-U relationship
- Social Influence (SI) - log-based function
- Exposure allocation from capital and success
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def exposure_from_capital(K: ArrayLike, alpha: float = 0.5) -> np.ndarray:
    """Compute exposure from capital using concave function.

    Concave function captures diminishing returns to capital investment.
    f(K) = K^alpha where alpha < 1

    Args:
        K: Capital values (scalar or array)
        alpha: Concavity parameter (0 < alpha < 1)

    Returns:
        Exposure values derived from capital

    Raises:
        ValueError: If alpha not in (0, 1) or K contains negatives
    """
    K = np.asarray(K, dtype=float)
    if alpha <= 0 or alpha >= 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if np.any(K < 0):
        raise ValueError("Capital values must be non-negative")
    return np.power(K, alpha)


def exposure_from_success(
    S: ArrayLike,
    S_median: float,
    beta: float = 1.5,
) -> np.ndarray:
    """Compute exposure from prior success using convex function.

    Convex function captures increasing returns (cumulative advantage).
    g(S) = (S / S_median)^beta where beta > 1

    Args:
        S: Prior success values
        S_median: Median success for normalization
        beta: Convexity parameter (beta > 1)

    Returns:
        Exposure values derived from success
    """
    S = np.asarray(S, dtype=float)
    if beta <= 1:
        raise ValueError(f"beta must be > 1 for convexity, got {beta}")
    if S_median <= 0:
        raise ValueError(f"S_median must be positive, got {S_median}")
    return np.power(np.maximum(S, 0) / S_median, beta)


def mere_exposure_effect(
    E: ArrayLike,
    lambda_: float = 0.3,
    tau: float = 15.0,
) -> np.ndarray:
    """Compute mere exposure effect contribution to perceived quality.

    Implements inverted-U relationship: liking increases with exposure
    up to a peak, then declines (habituation/boredom).

    MEE(E) = lambda * E * exp(-E / tau)

    Peak occurs at E = tau with value lambda * tau / e

    Args:
        E: Exposure counts (number of encounters)
        lambda_: Maximum effect strength (default: 0.3)
        tau: Saturation parameter - exposure at peak (default: 15)

    Returns:
        MEE contribution to perceived quality

    Example:
        >>> mere_exposure_effect(15, lambda_=0.3, tau=15)  # At peak
        1.655...  # Approximately lambda * tau / e
    """
    E = np.asarray(E, dtype=float)
    if lambda_ < 0:
        raise ValueError(f"lambda_ must be non-negative, got {lambda_}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return lambda_ * E * np.exp(-E / tau)


def mee_peak_exposure(tau: float = 15.0) -> float:
    """Return the exposure level at which MEE is maximized."""
    return tau


def mee_peak_value(lambda_: float = 0.3, tau: float = 15.0) -> float:
    """Return the maximum MEE value (at E = tau)."""
    return lambda_ * tau / np.e


def social_influence(
    S_mean: ArrayLike,
    S_median: float,
    gamma: float = 0.5,
) -> np.ndarray:
    """Compute social influence contribution to perceived quality.

    Log transformation prevents runaway effects while still
    capturing that observed popularity affects perception.

    SI(S_mean) = gamma * log(1 + S_mean / S_median)

    Args:
        S_mean: Average observed success across consumers
        S_median: Median success for normalization
        gamma: Social influence strength (default: 0.5)

    Returns:
        SI contribution to perceived quality
    """
    S_mean = np.asarray(S_mean, dtype=float)
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if S_median <= 0:
        raise ValueError(f"S_median must be positive, got {S_median}")
    # Use np.log1p for numerical stability when S_mean/S_median is small
    return gamma * np.log1p(np.maximum(S_mean, 0) / S_median)


def compute_total_exposure(
    capital: ArrayLike,
    prior_success: ArrayLike,
    S_median: float,
    alpha: float = 0.5,
    beta: float = 1.5,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute total exposure from capital and prior success.

    E_it = f(K_it) + g(S_{i,t-1}) + epsilon_it

    Args:
        capital: Current capital values for each producer
        prior_success: Success from previous period
        S_median: Median success for normalization
        alpha: Capital exposure concavity
        beta: Success exposure convexity
        noise_std: Standard deviation of exposure noise
        rng: Random number generator

    Returns:
        Total exposure for each producer
    """
    if rng is None:
        rng = np.random.default_rng()

    capital = np.asarray(capital, dtype=float)
    prior_success = np.asarray(prior_success, dtype=float)

    capital_exposure = exposure_from_capital(capital, alpha)
    success_exposure = exposure_from_success(prior_success, S_median, beta)
    noise = rng.normal(0, noise_std, size=capital.shape)

    # Ensure non-negative exposure
    return np.maximum(capital_exposure + success_exposure + noise, 0)
