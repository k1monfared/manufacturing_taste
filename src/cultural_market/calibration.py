"""Calibration against Salganik et al. (2006) experimental data.

Uses method of simulated moments to find parameters that match
empirical targets from the MusicLab experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.optimize import differential_evolution, minimize

from .market import DEFAULT_PARAMS, CulturalMarket
from .metrics import gini_coefficient, quality_success_correlation


@dataclass
class CalibrationTarget:
    """A single empirical target for calibration."""

    name: str
    value: float
    range_low: float
    range_high: float
    weight: float = 1.0


def load_calibration_targets(
    path: str | Path | None = None,
) -> List[CalibrationTarget]:
    """Load calibration targets from JSON file.

    Args:
        path: Path to salganik_targets.json (or None for default)

    Returns:
        List of CalibrationTarget objects
    """
    if path is None:
        # Default path relative to package
        path = Path(__file__).parent.parent.parent / "data" / "salganik_targets.json"

    with open(path) as f:
        data = json.load(f)

    targets = []
    for name, info in data["targets"].items():
        targets.append(
            CalibrationTarget(
                name=name,
                value=info["value"],
                range_low=info["range"][0],
                range_high=info["range"][1],
            )
        )

    return targets


def simulate_independent_condition(
    params: dict,
    n_runs: int = 10,
    seed: int | None = None,
) -> dict:
    """Run simulations with gamma=0 (no social influence).

    Returns:
        Dict with mean metrics across runs
    """
    rng = np.random.default_rng(seed)

    # Override social influence to zero
    ind_params = {**params, "si_gamma": 0.0}

    correlations = []
    ginis = []

    for _ in range(n_runs):
        market = CulturalMarket(ind_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        correlations.append(quality_success_correlation(qualities, successes))
        ginis.append(gini_coefficient(successes))

    return {
        "quality_success_correlation": float(np.mean(correlations)),
        "gini_coefficient": float(np.mean(ginis)),
    }


def simulate_social_condition(
    params: dict,
    n_runs: int = 10,
    seed: int | None = None,
) -> dict:
    """Run simulations with social influence active.

    Returns:
        Dict with mean metrics across runs
    """
    rng = np.random.default_rng(seed)

    correlations = []
    ginis = []

    qualities_ref = None
    all_ranks = []

    for run_idx in range(n_runs):
        market = CulturalMarket(params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = np.array([p.quality for p in market.producers])
        successes = np.array([p.cumulative_success for p in market.producers])

        correlations.append(quality_success_correlation(qualities, successes))
        ginis.append(gini_coefficient(successes))

        # Compute ranks for variance calculation
        ranks = np.argsort(np.argsort(-successes))  # Higher success = lower rank
        all_ranks.append(ranks)

        if run_idx == 0:
            qualities_ref = qualities

    # Compute rank variance for middle-quality producers
    all_ranks_arr = np.array(all_ranks)
    quality_percentiles = (
        np.argsort(np.argsort(qualities_ref)) / len(qualities_ref) * 100
    )
    middle_mask = (quality_percentiles >= 10) & (quality_percentiles < 90)

    if np.any(middle_mask):
        middle_rank_variance = float(
            np.mean(np.var(all_ranks_arr[:, middle_mask], axis=0))
        )
    else:
        middle_rank_variance = 0.0

    return {
        "quality_success_correlation": float(np.mean(correlations)),
        "gini_coefficient": float(np.mean(ginis)),
        "rank_variance_middle": middle_rank_variance,
    }


def calibration_loss(
    param_values: np.ndarray,
    param_names: List[str],
    base_params: dict,
    targets: List[CalibrationTarget],
    n_runs: int = 5,
) -> float:
    """Compute loss for calibration optimization.

    Args:
        param_values: Current parameter values being optimized
        param_names: Names of parameters being optimized
        base_params: Base parameters (non-optimized)
        targets: Calibration targets
        n_runs: Runs per condition

    Returns:
        Weighted sum of squared errors from targets
    """
    # Update params with current values
    params = base_params.copy()
    for name, value in zip(param_names, param_values):
        params[name] = value

    # Run both conditions
    ind_results = simulate_independent_condition(params, n_runs)
    soc_results = simulate_social_condition(params, n_runs)

    # Compute loss
    loss = 0.0

    for target in targets:
        if target.name == "quality_success_correlation_independent":
            simulated = ind_results["quality_success_correlation"]
        elif target.name == "quality_success_correlation_social":
            simulated = soc_results["quality_success_correlation"]
        elif target.name == "gini_ratio_social_independent":
            ind_gini = ind_results["gini_coefficient"]
            soc_gini = soc_results["gini_coefficient"]
            simulated = soc_gini / ind_gini if ind_gini > 0 else 0
        elif target.name == "rank_variance_middle_quality":
            simulated = soc_results["rank_variance_middle"]
        else:
            continue

        # Normalized squared error
        error = (simulated - target.value) / target.value if target.value != 0 else 0
        loss += target.weight * error**2

    return float(loss)


def calibrate(
    param_names: List[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_runs: int = 5,
    method: str = "differential_evolution",
    seed: int | None = None,
) -> dict:
    """Run calibration optimization.

    Args:
        param_names: Parameters to calibrate (default: key mechanism params)
        bounds: Bounds for each parameter
        n_runs: Simulation runs per evaluation
        method: Optimization method
        seed: Random seed

    Returns:
        Dict with calibrated parameters and diagnostics
    """
    if param_names is None:
        param_names = [
            "si_gamma",
            "mee_lambda",
            "mee_tau",
            "capital_alpha",
            "success_beta",
        ]

    if bounds is None:
        bounds = {
            "si_gamma": (0.1, 2.0),
            "mee_lambda": (0.1, 1.0),
            "mee_tau": (5, 30),
            "capital_alpha": (0.3, 0.8),
            "success_beta": (1.1, 3.0),
        }

    targets = load_calibration_targets()
    base_params = DEFAULT_PARAMS.copy()

    param_bounds = [bounds.get(name, (0.1, 2.0)) for name in param_names]

    if method == "differential_evolution":
        result = differential_evolution(
            calibration_loss,
            bounds=param_bounds,
            args=(param_names, base_params, targets, n_runs),
            seed=seed,
            maxiter=50,
            tol=0.01,
            workers=1,
        )
    else:
        # Initial guess from defaults
        x0 = np.array([base_params.get(name, 1.0) for name in param_names])
        result = minimize(
            calibration_loss,
            x0,
            args=(param_names, base_params, targets, n_runs),
            method="L-BFGS-B",
            bounds=param_bounds,
        )

    # Build calibrated params dict
    calibrated = base_params.copy()
    for name, value in zip(param_names, result.x):
        calibrated[name] = float(value)

    return {
        "calibrated_params": calibrated,
        "optimized_values": dict(zip(param_names, [float(v) for v in result.x])),
        "final_loss": float(result.fun),
        "success": result.success if hasattr(result, "success") else True,
        "message": result.message if hasattr(result, "message") else "",
    }
