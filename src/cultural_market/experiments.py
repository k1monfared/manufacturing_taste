"""Experiment runners for the five main experiments.

1. Salganik replication
2. Counterfactual canon formation
3. Variance decomposition
4. Historical scenario (18th-century Vienna)
5. Sensitivity analysis
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np
from tqdm import tqdm

from .distributions import generate_quality_distribution
from .market import DEFAULT_PARAMS, CulturalMarket
from .metrics import (
    canonical_probability_by_decile,
    counterfactual_distance,
    gini_coefficient,
    jaccard_similarity,
    quality_success_correlation,
)


@dataclass
class ExperimentResult:
    """Container for experiment results."""

    name: str
    params: dict
    metrics: dict
    raw_data: dict | None = None


def experiment_salganik_replication(
    n_runs: int = 100,
    params: dict | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Replicate Salganik MusicLab conditions.

    Compares independent (gamma=0) vs social influence conditions.

    Args:
        n_runs: Number of simulation runs per condition
        params: Base parameters (uses calibrated defaults if None)
        seed: Random seed
        progress: Show progress bar

    Returns:
        ExperimentResult with comparison metrics
    """
    rng = np.random.default_rng(seed)
    base_params = {**DEFAULT_PARAMS, **(params or {})}

    # Independent condition
    ind_params = {**base_params, "si_gamma": 0.0}
    ind_correlations = []
    ind_ginis = []

    iterator = range(n_runs)
    if progress:
        iterator = tqdm(iterator, desc="Independent condition")

    for _ in iterator:
        market = CulturalMarket(ind_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        ind_correlations.append(quality_success_correlation(qualities, successes))
        ind_ginis.append(gini_coefficient(successes))

    # Social influence condition
    soc_correlations = []
    soc_ginis = []
    soc_canonical_sets: List[Set[int]] = []

    iterator = range(n_runs)
    if progress:
        iterator = tqdm(iterator, desc="Social condition")

    for _ in iterator:
        market = CulturalMarket(base_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        soc_correlations.append(quality_success_correlation(qualities, successes))
        soc_ginis.append(gini_coefficient(successes))
        soc_canonical_sets.append(market.get_canonical_set())

    # Compute canonical overlap
    if len(soc_canonical_sets) > 1:
        overlaps = []
        for i, s1 in enumerate(soc_canonical_sets):
            for s2 in soc_canonical_sets[i + 1 :]:
                overlaps.append(jaccard_similarity(s1, s2))
        canonical_overlap = float(np.mean(overlaps))
    else:
        canonical_overlap = 1.0

    return ExperimentResult(
        name="salganik_replication",
        params=base_params,
        metrics={
            "independent": {
                "quality_success_corr_mean": float(np.mean(ind_correlations)),
                "quality_success_corr_std": float(np.std(ind_correlations)),
                "gini_mean": float(np.mean(ind_ginis)),
                "gini_std": float(np.std(ind_ginis)),
            },
            "social": {
                "quality_success_corr_mean": float(np.mean(soc_correlations)),
                "quality_success_corr_std": float(np.std(soc_correlations)),
                "gini_mean": float(np.mean(soc_ginis)),
                "gini_std": float(np.std(soc_ginis)),
            },
            "comparison": {
                "gini_ratio": float(np.mean(soc_ginis)) / float(np.mean(ind_ginis)),
                "corr_difference": float(np.mean(ind_correlations))
                - float(np.mean(soc_correlations)),
                "canonical_overlap": canonical_overlap,
            },
        },
    )


def experiment_counterfactual(
    n_runs: int = 1000,
    quality_seed: int = 42,
    params: dict | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Counterfactual canon formation experiment.

    Fix quality distribution, vary only capital allocation.
    Measures path dependence in canonical outcomes.

    Args:
        n_runs: Number of runs with different capital seeds
        quality_seed: Fixed seed for quality distribution
        params: Base parameters
        progress: Show progress bar

    Returns:
        ExperimentResult with counterfactual metrics
    """
    base_params = {**DEFAULT_PARAMS, **(params or {})}
    n_prod = base_params["n_producers"]

    # Generate fixed quality distribution
    fixed_qualities = generate_quality_distribution(
        n_prod,
        mean=base_params["quality_mean"],
        std=base_params["quality_std"],
        seed=quality_seed,
    )

    rng = np.random.default_rng()
    canonical_sets: List[Set[int]] = []
    canonical_counts = np.zeros(n_prod)

    iterator = range(n_runs)
    if progress:
        iterator = tqdm(iterator, desc="Counterfactual runs")

    for _ in iterator:
        market = CulturalMarket(base_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))

        # Override qualities with fixed distribution
        for i, producer in enumerate(market.producers):
            producer.quality = float(fixed_qualities[i])

        market.run()

        canonical_set = market.get_canonical_set()
        canonical_sets.append(canonical_set)

        for pid in canonical_set:
            canonical_counts[pid] += 1

    # Compute metrics
    canonical_probs = canonical_counts / n_runs

    # Quality percentiles for analysis
    quality_percentiles = np.argsort(np.argsort(fixed_qualities)) / n_prod * 100

    return ExperimentResult(
        name="counterfactual",
        params=base_params,
        metrics={
            "counterfactual_distance": counterfactual_distance(canonical_sets),
            "canonical_probability_variance": float(np.var(canonical_probs)),
            "canonical_by_decile": canonical_probability_by_decile(
                quality_percentiles, canonical_probs > 0.5
            ),
            "mean_canonical_prob": float(np.mean(canonical_probs)),
            "max_canonical_prob": float(np.max(canonical_probs)),
            "min_canonical_prob": float(np.min(canonical_probs)),
        },
        raw_data={
            "qualities": fixed_qualities.tolist(),
            "canonical_probs": canonical_probs.tolist(),
            "n_runs": n_runs,
        },
    )


def experiment_variance_decomposition(
    n_runs: int = 100,
    params: dict | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Decompose canonical variance into components.

    Ablation study:
    1. Full model
    2. No social influence (gamma=0)
    3. Homogeneous capital
    4. Both ablations

    Returns:
        ExperimentResult with variance decomposition
    """
    rng = np.random.default_rng(seed)
    base_params = {**DEFAULT_PARAMS, **(params or {})}

    conditions = {
        "full": base_params,
        "no_social": {**base_params, "si_gamma": 0.0},
        "homogeneous_capital": {**base_params, "capital_std": 0.01},
        "both_ablations": {**base_params, "si_gamma": 0.0, "capital_std": 0.01},
    }

    results = {}

    for cond_name, cond_params in conditions.items():
        if progress:
            print(f"Running {cond_name}...")

        canonical_variances = []
        quality_correlations = []

        for _ in range(n_runs):
            market = CulturalMarket(cond_params)
            market.initialize(seed=int(rng.integers(0, 2**31)))
            market.run()

            qualities = [p.quality for p in market.producers]
            canonical = [float(p.canonical) for p in market.producers]

            quality_correlations.append(
                quality_success_correlation(qualities, canonical)
            )
            canonical_variances.append(np.var(canonical))

        results[cond_name] = {
            "canonical_variance_mean": float(np.mean(canonical_variances)),
            "quality_canonical_corr_mean": float(np.mean(quality_correlations)),
        }

    # Compute variance attributions
    full_var = results["full"]["canonical_variance_mean"]

    if full_var > 0:
        # Social influence contribution
        social_contrib = (
            full_var - results["no_social"]["canonical_variance_mean"]
        ) / full_var

        # Capital contribution
        capital_contrib = (
            full_var - results["homogeneous_capital"]["canonical_variance_mean"]
        ) / full_var

        # Quality contribution (from double ablation)
        quality_contrib = (
            results["both_ablations"]["canonical_variance_mean"]
        ) / full_var

        # Interaction/residual
        residual = 1 - social_contrib - capital_contrib - quality_contrib
    else:
        social_contrib = capital_contrib = quality_contrib = residual = 0.0

    return ExperimentResult(
        name="variance_decomposition",
        params=base_params,
        metrics={
            "conditions": results,
            "decomposition": {
                "quality": max(0.0, quality_contrib),
                "capital": max(0.0, capital_contrib),
                "social_influence": max(0.0, social_contrib),
                "residual": max(0.0, residual),
            },
        },
    )


def experiment_historical_scenario(
    n_runs: int = 500,
    params: dict | None = None,
    seed: int | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Stylized 18th-century Viennese musical culture.

    Parameters adjusted for historical context:
    - Fewer producers (~300 active composers)
    - Higher capital inequality (patronage concentrated)
    - Constrained exposure (no recording)
    """
    rng = np.random.default_rng(seed)

    # Historical scenario parameters
    historical_params = {
        **DEFAULT_PARAMS,
        **(params or {}),
        "n_producers": 300,
        "n_consumers": 5000,  # Smaller audience
        "capital_std": 2.0,  # Higher inequality
        "si_gamma": 0.7,  # Stronger social influence (word of mouth)
        "t_active": 30,  # Shorter active career
        "t_canon": 200,  # Longer canonization process
    }

    canonical_sets: List[Set[int]] = []
    quality_corrs = []
    capital_corrs = []

    iterator = range(n_runs)
    if progress:
        iterator = tqdm(iterator, desc="Historical scenario")

    for _ in iterator:
        market = CulturalMarket(historical_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        capitals = [p.initial_capital for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        quality_corrs.append(quality_success_correlation(qualities, successes))
        capital_corrs.append(quality_success_correlation(capitals, successes))
        canonical_sets.append(market.get_canonical_set())

    # Compute canonical overlap (sample for efficiency)
    if len(canonical_sets) > 1:
        sample_size = min(100, len(canonical_sets))
        overlaps = []
        for i in range(sample_size):
            for j in range(i + 1, min(i + 10, sample_size)):
                overlaps.append(
                    jaccard_similarity(canonical_sets[i], canonical_sets[j])
                )
        canonical_overlap = float(np.mean(overlaps)) if overlaps else 1.0
    else:
        canonical_overlap = 1.0

    return ExperimentResult(
        name="historical_scenario",
        params=historical_params,
        metrics={
            "counterfactual_distance": counterfactual_distance(canonical_sets[:100]),
            "quality_success_corr_mean": float(np.mean(quality_corrs)),
            "quality_success_corr_std": float(np.std(quality_corrs)),
            "capital_success_corr_mean": float(np.mean(capital_corrs)),
            "capital_success_corr_std": float(np.std(capital_corrs)),
            "canonical_set_size_mean": float(np.mean([len(s) for s in canonical_sets])),
            "canonical_overlap_mean": canonical_overlap,
        },
    )


def experiment_sensitivity(
    base_params: dict | None = None,
    vary_by: float = 0.5,
    n_runs: int = 20,
    seed: int | None = None,
    progress: bool = True,
) -> ExperimentResult:
    """Sensitivity analysis varying each parameter +/- 50%.

    Args:
        base_params: Baseline parameters
        vary_by: Fraction to vary (0.5 = +/- 50%)
        n_runs: Runs per parameter setting
        seed: Random seed

    Returns:
        ExperimentResult with sensitivity metrics
    """
    rng = np.random.default_rng(seed)
    base_params = {**DEFAULT_PARAMS, **(base_params or {})}

    # Parameters to vary
    param_names = [
        "si_gamma",
        "mee_lambda",
        "mee_tau",
        "capital_alpha",
        "success_beta",
        "capital_std",
    ]

    sensitivities = {}

    for param_name in param_names:
        if progress:
            print(f"Varying {param_name}...")

        base_value = base_params[param_name]
        low_value = base_value * (1 - vary_by)
        high_value = base_value * (1 + vary_by)

        results_by_level = {}

        for level, value in [
            ("low", low_value),
            ("base", base_value),
            ("high", high_value),
        ]:
            test_params = {**base_params, param_name: value}

            corrs = []
            ginis = []

            for _ in range(n_runs):
                market = CulturalMarket(test_params)
                market.initialize(seed=int(rng.integers(0, 2**31)))
                market.run()

                qualities = [p.quality for p in market.producers]
                successes = [p.cumulative_success for p in market.producers]

                corrs.append(quality_success_correlation(qualities, successes))
                ginis.append(gini_coefficient(successes))

            results_by_level[level] = {
                "value": float(value),
                "quality_success_corr": float(np.mean(corrs)),
                "gini": float(np.mean(ginis)),
            }

        # Compute sensitivity as change per unit parameter change
        if base_value != 0:
            corr_sensitivity = (
                results_by_level["high"]["quality_success_corr"]
                - results_by_level["low"]["quality_success_corr"]
            ) / (2 * vary_by * base_value)

            gini_sensitivity = (
                results_by_level["high"]["gini"] - results_by_level["low"]["gini"]
            ) / (2 * vary_by * base_value)
        else:
            corr_sensitivity = gini_sensitivity = 0.0

        sensitivities[param_name] = {
            "levels": results_by_level,
            "corr_sensitivity": float(corr_sensitivity),
            "gini_sensitivity": float(gini_sensitivity),
        }

    return ExperimentResult(
        name="sensitivity",
        params=base_params,
        metrics={"sensitivities": sensitivities},
    )
