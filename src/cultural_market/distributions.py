"""Distribution generators for quality and capital.

Quality: Truncated normal distribution
Capital: Log-normal distribution (to capture inequality)
Consumer susceptibilities: Normal distributions
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats


def generate_quality_distribution(
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float = -3.0,
    upper: float = 3.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate quality values from truncated normal distribution.

    Args:
        n: Number of quality values to generate
        mean: Mean of underlying normal
        std: Standard deviation of underlying normal
        lower: Lower truncation bound (in original scale)
        upper: Upper truncation bound (in original scale)
        seed: Random seed for reproducibility

    Returns:
        Array of n quality values
    """
    rng = np.random.default_rng(seed)

    # Convert bounds to standard normal scale
    a = (lower - mean) / std
    b = (upper - mean) / std

    # Use scipy's truncnorm
    truncnorm = stats.truncnorm(a, b, loc=mean, scale=std)
    return truncnorm.rvs(n, random_state=rng)


def generate_capital_distribution(
    n: int,
    mean: float = 0.0,
    std: float = 1.5,
    seed: int | None = None,
) -> np.ndarray:
    """Generate capital values from log-normal distribution.

    Log-normal captures wealth inequality with long right tail.
    Parameters are for the underlying normal distribution.

    Args:
        n: Number of capital values to generate
        mean: Mean of log (not of the distribution itself)
        std: Standard deviation of log
        seed: Random seed

    Returns:
        Array of n capital values (all positive)
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean, std, n)


def generate_consumer_susceptibilities(
    n: int,
    mee_mean: float = 1.0,
    mee_std: float = 0.2,
    si_mean: float = 1.0,
    si_std: float = 0.2,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate consumer susceptibility parameters.

    Args:
        n: Number of consumers
        mee_mean: Mean susceptibility to mere exposure effect
        mee_std: Standard deviation of MEE susceptibility
        si_mean: Mean susceptibility to social influence
        si_std: Standard deviation of SI susceptibility
        seed: Random seed

    Returns:
        Tuple of (alpha_j array, beta_j array) for MEE and SI susceptibilities
    """
    rng = np.random.default_rng(seed)

    # Ensure non-negative susceptibilities by clipping
    alpha = np.maximum(rng.normal(mee_mean, mee_std, n), 0)
    beta = np.maximum(rng.normal(si_mean, si_std, n), 0)

    return alpha, beta


def compute_quality_percentiles(qualities: ArrayLike) -> np.ndarray:
    """Compute percentile rank for each quality value.

    Useful for identifying top/bottom deciles for analysis.
    """
    qualities = np.asarray(qualities)
    return stats.rankdata(qualities, method="average") / len(qualities) * 100
