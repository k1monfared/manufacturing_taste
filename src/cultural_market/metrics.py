"""Metrics for evaluating simulation outcomes.

Includes:
- Gini coefficient for inequality
- Quality-success correlations
- Jaccard similarity between canonical sets
- Counterfactual distance measures
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Set

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .agents import Producer


def gini_coefficient(values: ArrayLike) -> float:
    """Compute Gini coefficient of inequality.

    Gini = 0 means perfect equality, Gini = 1 means maximum inequality.

    Args:
        values: Array of values (e.g., success, wealth)

    Returns:
        Gini coefficient between 0 and 1
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]  # Remove NaN

    if len(values) == 0:
        return 0.0

    # Handle all-zero case
    if np.all(values == 0):
        return 0.0

    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumsum = np.cumsum(sorted_values)

    # Standard Gini formula
    return (2 * np.sum((np.arange(1, n + 1) * sorted_values)) / (n * cumsum[-1])) - (
        n + 1
    ) / n


def quality_success_correlation(
    qualities: ArrayLike,
    successes: ArrayLike,
) -> float:
    """Compute Pearson correlation between quality and success.

    Args:
        qualities: Array of intrinsic quality values
        successes: Array of cumulative success values

    Returns:
        Pearson correlation coefficient
    """
    qualities = np.asarray(qualities)
    successes = np.asarray(successes)

    if len(qualities) != len(successes):
        raise ValueError("Arrays must have same length")

    # Handle edge cases
    if np.std(qualities) == 0 or np.std(successes) == 0:
        return 0.0

    return float(np.corrcoef(qualities, successes)[0, 1])


def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Compute Jaccard similarity between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity between 0 and 1
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def jaccard_distance(set1: Set, set2: Set) -> float:
    """Compute Jaccard distance (1 - similarity)."""
    return 1.0 - jaccard_similarity(set1, set2)


def counterfactual_distance(canonical_sets: List[Set]) -> float:
    """Compute average Jaccard distance across all pairs of canonical sets.

    Higher values indicate more path-dependent outcomes.

    Args:
        canonical_sets: List of canonical sets from different runs

    Returns:
        Mean pairwise Jaccard distance
    """
    if len(canonical_sets) < 2:
        return 0.0

    distances = []
    for i, s1 in enumerate(canonical_sets):
        for s2 in canonical_sets[i + 1 :]:
            distances.append(jaccard_distance(s1, s2))

    return float(np.mean(distances))


def canonical_probability_by_decile(
    quality_percentiles: ArrayLike,
    is_canonical: ArrayLike,
) -> dict[int, float]:
    """Compute canonical probability for each quality decile.

    Args:
        quality_percentiles: Percentile rank (0-100) for each producer
        is_canonical: Boolean array indicating canonical status

    Returns:
        Dict mapping decile (1-10) to probability of canonical status
    """
    quality_percentiles = np.asarray(quality_percentiles)
    is_canonical = np.asarray(is_canonical, dtype=bool)

    results = {}
    for decile in range(1, 11):
        lower = (decile - 1) * 10
        upper = decile * 10
        mask = (quality_percentiles >= lower) & (quality_percentiles < upper)
        if np.sum(mask) > 0:
            results[decile] = float(np.mean(is_canonical[mask]))
        else:
            results[decile] = 0.0

    return results


def rank_variance_by_quality_band(
    qualities: ArrayLike,
    ranks_across_runs: np.ndarray,
) -> dict[str, float]:
    """Compute rank variance for top, middle, and bottom quality bands.

    Args:
        qualities: Quality values (determines band assignment)
        ranks_across_runs: 2D array (n_runs x n_producers) of ranks

    Returns:
        Dict with variance for 'top_decile', 'middle_80', 'bottom_decile'
    """
    qualities = np.asarray(qualities)
    percentiles = np.argsort(np.argsort(qualities)) / len(qualities) * 100

    top_mask = percentiles >= 90
    bottom_mask = percentiles < 10
    middle_mask = ~top_mask & ~bottom_mask

    # Variance across runs for each producer, then mean within band
    rank_variances = np.var(ranks_across_runs, axis=0)

    return {
        "top_decile": float(np.mean(rank_variances[top_mask]))
        if np.any(top_mask)
        else 0,
        "middle_80": float(np.mean(rank_variances[middle_mask]))
        if np.any(middle_mask)
        else 0,
        "bottom_decile": float(np.mean(rank_variances[bottom_mask]))
        if np.any(bottom_mask)
        else 0,
    }
