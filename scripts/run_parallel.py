#!/usr/bin/env python
"""Run simulations in parallel using multiprocessing.

Uses all available CPU cores for independent simulation runs.
Each run is fully independent so parallelization is trivial.

Usage:
    python scripts/run_parallel.py --experiment salganik --n-runs 100 --workers 12
    python scripts/run_parallel.py --experiment variance --n-runs 200 --workers 12
    python scripts/run_parallel.py --experiment all --n-runs 100 --workers 12
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.market import DEFAULT_PARAMS, CulturalMarket
from cultural_market.metrics import gini_coefficient, quality_success_correlation

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"


def run_single_salganik(args):
    """Run a single Salganik simulation (independent or social)."""
    condition, seed = args
    params = {**DEFAULT_PARAMS}
    if condition == "independent":
        params["si_gamma"] = 0.0

    market = CulturalMarket(params)
    market.initialize(seed=seed)
    market.run()

    qualities = [p.quality for p in market.producers]
    successes = [p.cumulative_success for p in market.producers]

    record = {
        "condition": condition,
        "quality_success_corr": float(quality_success_correlation(qualities, successes)),
        "gini": float(gini_coefficient(successes)),
        "seed": seed,
    }
    if condition == "social":
        record["canonical_set"] = list(market.get_canonical_set())

    return record


def run_single_variance(args):
    """Run a single variance decomposition simulation."""
    condition, params, seed = args

    market = CulturalMarket(params)
    market.initialize(seed=seed)
    market.run()

    qualities = [p.quality for p in market.producers]
    canonical = [float(p.canonical) for p in market.producers]

    return {
        "condition": condition,
        "canonical_variance": float(np.var(canonical)),
        "quality_canonical_corr": float(quality_success_correlation(qualities, canonical)),
        "seed": seed,
    }


def run_single_historical(seed):
    """Run a single historical scenario simulation."""
    params = {
        **DEFAULT_PARAMS,
        "n_producers": 300,
        "n_consumers": 5000,
        "capital_std": 2.0,
        "si_gamma": 0.7,
        "t_active": 30,
        "t_canon": 200,
    }
    market = CulturalMarket(params)
    market.initialize(seed=seed)
    market.run()

    qualities = [p.quality for p in market.producers]
    capitals = [p.initial_capital for p in market.producers]
    successes = [p.cumulative_success for p in market.producers]

    return {
        "quality_success_corr": float(quality_success_correlation(qualities, successes)),
        "capital_success_corr": float(quality_success_correlation(capitals, successes)),
        "canonical_set": list(market.get_canonical_set()),
        "seed": seed,
    }


def run_single_counterfactual(args):
    """Run a single counterfactual simulation with fixed qualities."""
    fixed_qualities, seed = args
    params = {**DEFAULT_PARAMS}

    market = CulturalMarket(params)
    market.initialize(seed=seed)

    for i, producer in enumerate(market.producers):
        producer.quality = float(fixed_qualities[i])

    market.run()

    return {
        "canonical_set": list(market.get_canonical_set()),
        "seed": seed,
    }


def run_salganik(n_runs, workers, output_file):
    """Run Salganik experiment in parallel."""
    rng = np.random.default_rng()
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs * 2)]

    tasks = []
    for i in range(n_runs):
        tasks.append(("independent", seeds[i]))
        tasks.append(("social", seeds[n_runs + i]))

    print(f"Running {len(tasks)} Salganik simulations on {workers} workers...")
    start = time.time()

    with Pool(workers) as pool:
        results = pool.map(run_single_salganik, tasks)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({elapsed/len(tasks):.1f}s per run effective)")

    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Appended {len(results)} records to {output_file}")


def run_variance(n_runs, workers, output_file):
    """Run variance decomposition in parallel."""
    rng = np.random.default_rng()
    base_params = DEFAULT_PARAMS.copy()

    conditions = {
        "full": base_params,
        "no_social": {**base_params, "si_gamma": 0.0},
        "homogeneous_capital": {**base_params, "capital_std": 0.01},
        "both_ablations": {**base_params, "si_gamma": 0.0, "capital_std": 0.01},
    }

    tasks = []
    for cond_name, cond_params in conditions.items():
        for _ in range(n_runs):
            seed = int(rng.integers(0, 2**31))
            tasks.append((cond_name, cond_params, seed))

    print(f"Running {len(tasks)} variance decomposition simulations on {workers} workers...")
    start = time.time()

    with Pool(workers) as pool:
        results = pool.map(run_single_variance, tasks)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({elapsed/len(tasks):.1f}s per run effective)")

    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Appended {len(results)} records to {output_file}")


def run_historical(n_runs, workers, output_file):
    """Run historical scenario in parallel."""
    rng = np.random.default_rng()
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_runs)]

    print(f"Running {n_runs} historical simulations on {workers} workers...")
    start = time.time()

    with Pool(workers) as pool:
        results = pool.map(run_single_historical, seeds)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({elapsed/n_runs:.1f}s per run effective)")

    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Appended {len(results)} records to {output_file}")


def run_counterfactual(n_runs, workers, output_file):
    """Run counterfactual experiment in parallel."""
    from cultural_market.distributions import generate_quality_distribution

    params = DEFAULT_PARAMS.copy()
    fixed_qualities = generate_quality_distribution(
        params["n_producers"],
        mean=params["quality_mean"],
        std=params["quality_std"],
        seed=42,
    )

    rng = np.random.default_rng()
    tasks = [(fixed_qualities, int(rng.integers(0, 2**31))) for _ in range(n_runs)]

    print(f"Running {n_runs} counterfactual simulations on {workers} workers...")
    start = time.time()

    with Pool(workers) as pool:
        results = pool.map(run_single_counterfactual, tasks)

    elapsed = time.time() - start
    print(f"Done in {elapsed:.0f}s ({elapsed/n_runs:.1f}s per run effective)")

    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Appended {len(results)} records to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Run simulations in parallel")
    parser.add_argument("--experiment", choices=["salganik", "variance", "historical", "counterfactual", "all"],
                        required=True)
    parser.add_argument("--n-runs", type=int, default=100, help="Runs per condition")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 4),
                        help="Number of parallel workers")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Using {args.workers} workers")

    experiments = [args.experiment] if args.experiment != "all" else [
        "salganik", "variance", "historical", "counterfactual"
    ]

    for exp in experiments:
        if exp == "salganik":
            run_salganik(args.n_runs, args.workers, RAW_DIR / "salganik_runs.jsonl")
        elif exp == "variance":
            run_variance(args.n_runs, args.workers, RAW_DIR / "variance_runs.jsonl")
        elif exp == "historical":
            run_historical(args.n_runs, args.workers, RAW_DIR / "historical_runs.jsonl")
        elif exp == "counterfactual":
            run_counterfactual(args.n_runs, args.workers, RAW_DIR / "counterfactual_runs.jsonl")


if __name__ == "__main__":
    main()
