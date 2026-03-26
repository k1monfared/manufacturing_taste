#!/usr/bin/env python
"""
Run full cultural market analysis with default parameters.

This script runs all five experiments with full-scale parameters
and 1000+ runs per experiment for robust statistics.

Expected runtime: Several hours depending on hardware.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.experiments import (
    experiment_counterfactual,
    experiment_historical_scenario,
    experiment_salganik_replication,
    experiment_sensitivity,
    experiment_variance_decomposition,
)
from cultural_market.market import DEFAULT_PARAMS

# Use default parameters from the spec
SIMULATION_PARAMS = {
    "n_producers": DEFAULT_PARAMS["n_producers"],   # 1000 producers
    "n_consumers": DEFAULT_PARAMS["n_consumers"],   # 10000 consumers
    "t_canon": DEFAULT_PARAMS["t_canon"],           # 100 periods
    "t_active": DEFAULT_PARAMS["t_active"],         # 50 active periods
}

# High run counts for robust statistics
EXPERIMENT_RUNS = {
    "salganik": 1000,        # 1000 runs per condition
    "counterfactual": 2000,  # More runs for path dependence measurement
    "variance": 1000,        # Ablation study
    "historical": 1000,      # Historical scenario
    "sensitivity": 100,      # Per-parameter-level (x3 levels x6 params = 1800 total)
}


def run_all_experiments(output_dir: Path, seed: int = 42) -> dict:
    """Run all experiments and return results."""
    results = {}

    # 1. Salganik Replication
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Salganik Replication")
    print("Comparing independent vs social influence conditions")
    print("=" * 60)

    result = experiment_salganik_replication(
        n_runs=EXPERIMENT_RUNS["salganik"],
        params=SIMULATION_PARAMS,
        seed=seed,
        progress=True,
    )
    results["salganik"] = result

    print("\nResults:")
    print(f"  Independent condition:")
    print(f"    Quality-Success Correlation: {result.metrics['independent']['quality_success_corr_mean']:.3f} ± {result.metrics['independent']['quality_success_corr_std']:.3f}")
    print(f"    Gini Coefficient: {result.metrics['independent']['gini_mean']:.3f}")
    print(f"  Social condition:")
    print(f"    Quality-Success Correlation: {result.metrics['social']['quality_success_corr_mean']:.3f} ± {result.metrics['social']['quality_success_corr_std']:.3f}")
    print(f"    Gini Coefficient: {result.metrics['social']['gini_mean']:.3f}")
    print(f"  Gini Ratio (social/independent): {result.metrics['comparison']['gini_ratio']:.3f}")
    print(f"  Correlation Reduction: {result.metrics['comparison']['corr_difference']:.3f}")

    # 2. Counterfactual Canon Formation
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Counterfactual Canon Formation")
    print("Fixed quality, varying capital - measuring path dependence")
    print("=" * 60)

    result = experiment_counterfactual(
        n_runs=EXPERIMENT_RUNS["counterfactual"],
        quality_seed=42,
        params=SIMULATION_PARAMS,
        progress=True,
    )
    results["counterfactual"] = result

    print("\nResults:")
    print(f"  Counterfactual Distance: {result.metrics['counterfactual_distance']:.3f}")
    print(f"    (0 = same canon every time, 1 = completely different canons)")
    print(f"  Canonical Probability Variance: {result.metrics['canonical_probability_variance']:.4f}")
    print(f"  Mean Canonical Probability: {result.metrics['mean_canonical_prob']:.3f}")
    print(f"  Canonical Probability by Quality Decile:")
    for decile, prob in result.metrics['canonical_by_decile'].items():
        bar = "█" * int(prob * 20)
        print(f"    Decile {decile:2d}: {prob:.2f} {bar}")

    # 3. Variance Decomposition
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Variance Decomposition")
    print("Attributing canonical status to quality, capital, social influence")
    print("=" * 60)

    result = experiment_variance_decomposition(
        n_runs=EXPERIMENT_RUNS["variance"],
        params=SIMULATION_PARAMS,
        seed=seed,
        progress=True,
    )
    results["variance"] = result

    print("\nVariance Decomposition:")
    decomp = result.metrics['decomposition']
    total = sum(decomp.values())
    for source, value in decomp.items():
        pct = (value / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 5)
        print(f"  {source:20s}: {pct:5.1f}% {bar}")

    # 4. Historical Scenario (18th-century Vienna)
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Historical Scenario (18th-century Vienna)")
    print("Stylized parameters: fewer composers, higher inequality")
    print("=" * 60)

    result = experiment_historical_scenario(
        n_runs=EXPERIMENT_RUNS["historical"],
        seed=seed,
        progress=True,
    )
    results["historical"] = result

    print("\nResults:")
    print(f"  Quality-Success Correlation: {result.metrics['quality_success_corr_mean']:.3f} ± {result.metrics['quality_success_corr_std']:.3f}")
    print(f"  Capital-Success Correlation: {result.metrics['capital_success_corr_mean']:.3f} ± {result.metrics['capital_success_corr_std']:.3f}")
    print(f"  Counterfactual Distance: {result.metrics['counterfactual_distance']:.3f}")
    print(f"  Canonical Overlap: {result.metrics['canonical_overlap_mean']:.3f}")

    # 5. Sensitivity Analysis
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Sensitivity Analysis")
    print("Varying parameters ±50% to assess robustness")
    print("=" * 60)

    result = experiment_sensitivity(
        base_params=SIMULATION_PARAMS,
        vary_by=0.5,
        n_runs=EXPERIMENT_RUNS["sensitivity"],
        seed=seed,
        progress=True,
    )
    results["sensitivity"] = result

    print("\nParameter Sensitivity (impact on quality-success correlation):")
    sensitivities = result.metrics['sensitivities']
    # Sort by absolute sensitivity
    sorted_params = sorted(
        sensitivities.keys(),
        key=lambda p: abs(sensitivities[p]['corr_sensitivity']),
        reverse=True
    )
    for param in sorted_params:
        sens = sensitivities[param]
        direction = "↑" if sens['corr_sensitivity'] > 0 else "↓"
        print(f"  {param:15s}: {direction} {abs(sens['corr_sensitivity']):.4f}")
        print(f"    Low ({sens['levels']['low']['value']:.2f}): corr={sens['levels']['low']['quality_success_corr']:.3f}")
        print(f"    High ({sens['levels']['high']['value']:.2f}): corr={sens['levels']['high']['quality_success_corr']:.3f}")

    return results


def save_results(results: dict, output_dir: Path) -> None:
    """Save all results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, result in results.items():
        output_file = output_dir / f"{name}_{timestamp}.json"
        result_dict = {
            "name": result.name,
            "params": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in result.params.items()
            },
            "metrics": result.metrics,
            "timestamp": timestamp,
        }
        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)
        print(f"Saved: {output_file}")


def main() -> None:
    print("=" * 60)
    print("CULTURAL MARKET SIMULATION - FULL ANALYSIS")
    print("=" * 60)
    print(f"\nParameters:")
    for key, value in SIMULATION_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"\nExperiment runs:")
    for exp, runs in EXPERIMENT_RUNS.items():
        print(f"  {exp}: {runs} runs")
    print("\nStarting experiments...\n")

    start_time = datetime.now()

    output_dir = Path(__file__).parent.parent / "results"
    results = run_all_experiments(output_dir, seed=42)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Key findings
    salganik = results["salganik"].metrics
    counterfactual = results["counterfactual"].metrics
    variance = results["variance"].metrics["decomposition"]

    print("\nKey Findings:")
    print(f"  1. Social influence reduces quality-success correlation by {salganik['comparison']['corr_difference']:.2f}")
    print(f"  2. Social influence increases inequality (Gini ratio: {salganik['comparison']['gini_ratio']:.2f})")
    print(f"  3. Counterfactual distance: {counterfactual['counterfactual_distance']:.2f} (path dependence)")
    print(f"  4. Variance attribution:")
    print(f"       Quality: {variance['quality']*100:.1f}%")
    print(f"       Capital: {variance['capital']*100:.1f}%")
    print(f"       Social: {variance['social_influence']*100:.1f}%")

    # Save results
    print("\n" + "-" * 60)
    print("Saving results...")
    save_results(results, output_dir)

    elapsed = datetime.now() - start_time
    print(f"\nTotal runtime: {elapsed}")
    print("Done!")


if __name__ == "__main__":
    main()
