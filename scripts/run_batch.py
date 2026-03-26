#!/usr/bin/env python
"""
Run a batch of cultural market simulations with raw data storage.

Results are saved incrementally so batches can be combined later.
Each batch appends to existing data files in results/raw/

Usage:
    python scripts/run_batch.py --n-runs 100
    python scripts/run_batch.py --n-runs 100  # Run again to add more
    python scripts/combine_batches.py          # Combine and analyze
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Set
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.market import CulturalMarket, DEFAULT_PARAMS
from cultural_market.metrics import (
    gini_coefficient,
    quality_success_correlation,
    jaccard_similarity,
)
from cultural_market.distributions import generate_quality_distribution


# Use default parameters from spec
SIMULATION_PARAMS = {
    "n_producers": DEFAULT_PARAMS["n_producers"],   # 1000
    "n_consumers": DEFAULT_PARAMS["n_consumers"],   # 10000
    "t_canon": DEFAULT_PARAMS["t_canon"],           # 100
    "t_active": DEFAULT_PARAMS["t_active"],         # 50
}

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"


def ensure_raw_dir():
    """Create raw data directory if needed."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def append_jsonl(filepath: Path, records: List[dict]):
    """Append records to a JSON Lines file."""
    with open(filepath, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def run_salganik_batch(n_runs: int, seed: int | None = None):
    """Run Salganik replication batch, saving raw per-run data."""
    print("\n" + "=" * 60)
    print("SALGANIK REPLICATION BATCH")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    results_file = RAW_DIR / "salganik_runs.jsonl"

    # Independent condition
    print("Running independent condition...")
    ind_params = {**SIMULATION_PARAMS, "si_gamma": 0.0}

    for _ in tqdm(range(n_runs), desc="Independent"):
        run_seed = int(rng.integers(0, 2**31))
        market = CulturalMarket(ind_params)
        market.initialize(seed=run_seed)
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        record = {
            "condition": "independent",
            "seed": run_seed,
            "quality_success_corr": quality_success_correlation(qualities, successes),
            "gini": gini_coefficient(successes),
            "timestamp": datetime.now().isoformat(),
        }
        append_jsonl(results_file, [record])

    # Social condition
    print("Running social condition...")
    for _ in tqdm(range(n_runs), desc="Social"):
        run_seed = int(rng.integers(0, 2**31))
        market = CulturalMarket(SIMULATION_PARAMS)
        market.initialize(seed=run_seed)
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]
        canonical_set = list(market.get_canonical_set())

        record = {
            "condition": "social",
            "seed": run_seed,
            "quality_success_corr": quality_success_correlation(qualities, successes),
            "gini": gini_coefficient(successes),
            "canonical_set": canonical_set,
            "timestamp": datetime.now().isoformat(),
        }
        append_jsonl(results_file, [record])

    print(f"Saved to {results_file}")


def run_counterfactual_batch(n_runs: int, quality_seed: int = 42):
    """Run counterfactual batch with fixed quality distribution."""
    print("\n" + "=" * 60)
    print("COUNTERFACTUAL BATCH")
    print("=" * 60)

    rng = np.random.default_rng()
    results_file = RAW_DIR / "counterfactual_runs.jsonl"
    qualities_file = RAW_DIR / "counterfactual_qualities.json"

    n_prod = SIMULATION_PARAMS["n_producers"]

    # Load or create fixed quality distribution
    if qualities_file.exists():
        with open(qualities_file) as f:
            fixed_qualities = np.array(json.load(f)["qualities"])
        print(f"Loaded existing quality distribution from {qualities_file}")
    else:
        fixed_qualities = generate_quality_distribution(
            n_prod,
            mean=SIMULATION_PARAMS.get("quality_mean", 0.0),
            std=SIMULATION_PARAMS.get("quality_std", 1.0),
            seed=quality_seed,
        )
        with open(qualities_file, "w") as f:
            json.dump({"qualities": fixed_qualities.tolist(), "seed": quality_seed}, f)
        print(f"Created fixed quality distribution, saved to {qualities_file}")

    for _ in tqdm(range(n_runs), desc="Counterfactual"):
        run_seed = int(rng.integers(0, 2**31))
        market = CulturalMarket(SIMULATION_PARAMS)
        market.initialize(seed=run_seed)

        # Override qualities
        for i, producer in enumerate(market.producers):
            producer.quality = float(fixed_qualities[i])

        market.run()

        canonical_set = list(market.get_canonical_set())

        record = {
            "seed": run_seed,
            "canonical_set": canonical_set,
            "timestamp": datetime.now().isoformat(),
        }
        append_jsonl(results_file, [record])

    print(f"Saved to {results_file}")


def run_variance_batch(n_runs: int, seed: int | None = None):
    """Run variance decomposition batch."""
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION BATCH")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    results_file = RAW_DIR / "variance_runs.jsonl"

    conditions = {
        "full": SIMULATION_PARAMS,
        "no_social": {**SIMULATION_PARAMS, "si_gamma": 0.0},
        "homogeneous_capital": {**SIMULATION_PARAMS, "capital_std": 0.01},
        "both_ablations": {**SIMULATION_PARAMS, "si_gamma": 0.0, "capital_std": 0.01},
    }

    for cond_name, cond_params in conditions.items():
        print(f"Running {cond_name}...")

        for _ in tqdm(range(n_runs), desc=cond_name):
            run_seed = int(rng.integers(0, 2**31))
            market = CulturalMarket(cond_params)
            market.initialize(seed=run_seed)
            market.run()

            qualities = [p.quality for p in market.producers]
            canonical = [float(p.canonical) for p in market.producers]

            record = {
                "condition": cond_name,
                "seed": run_seed,
                "quality_canonical_corr": quality_success_correlation(qualities, canonical),
                "canonical_variance": float(np.var(canonical)),
                "timestamp": datetime.now().isoformat(),
            }
            append_jsonl(results_file, [record])

    print(f"Saved to {results_file}")


def run_historical_batch(n_runs: int, seed: int | None = None):
    """Run historical scenario batch."""
    print("\n" + "=" * 60)
    print("HISTORICAL SCENARIO BATCH")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    results_file = RAW_DIR / "historical_runs.jsonl"

    historical_params = {
        **SIMULATION_PARAMS,
        "n_producers": 300,
        "n_consumers": 5000,
        "capital_std": 2.0,
        "si_gamma": 0.7,
        "t_active": 30,
        "t_canon": 200,
    }

    for _ in tqdm(range(n_runs), desc="Historical"):
        run_seed = int(rng.integers(0, 2**31))
        market = CulturalMarket(historical_params)
        market.initialize(seed=run_seed)
        market.run()

        qualities = [p.quality for p in market.producers]
        capitals = [p.initial_capital for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]
        canonical_set = list(market.get_canonical_set())

        record = {
            "seed": run_seed,
            "quality_success_corr": quality_success_correlation(qualities, successes),
            "capital_success_corr": quality_success_correlation(capitals, successes),
            "canonical_set": canonical_set,
            "timestamp": datetime.now().isoformat(),
        }
        append_jsonl(results_file, [record])

    print(f"Saved to {results_file}")


def run_sensitivity_batch(n_runs: int, seed: int | None = None):
    """Run sensitivity analysis batch."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS BATCH")
    print("=" * 60)

    rng = np.random.default_rng(seed)
    results_file = RAW_DIR / "sensitivity_runs.jsonl"

    param_names = ["si_gamma", "mee_lambda", "mee_tau", "capital_alpha", "success_beta", "capital_std"]
    vary_by = 0.5

    for param_name in param_names:
        base_value = SIMULATION_PARAMS.get(param_name, DEFAULT_PARAMS[param_name])

        for level, multiplier in [("low", 1 - vary_by), ("base", 1.0), ("high", 1 + vary_by)]:
            value = base_value * multiplier
            test_params = {**SIMULATION_PARAMS, param_name: value}

            print(f"Running {param_name}={value:.3f} ({level})...")

            for _ in tqdm(range(n_runs), desc=f"{param_name}/{level}"):
                run_seed = int(rng.integers(0, 2**31))
                market = CulturalMarket(test_params)
                market.initialize(seed=run_seed)
                market.run()

                qualities = [p.quality for p in market.producers]
                successes = [p.cumulative_success for p in market.producers]

                record = {
                    "param_name": param_name,
                    "level": level,
                    "value": value,
                    "seed": run_seed,
                    "quality_success_corr": quality_success_correlation(qualities, successes),
                    "gini": gini_coefficient(successes),
                    "timestamp": datetime.now().isoformat(),
                }
                append_jsonl(results_file, [record])

    print(f"Saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a batch of cultural market simulations"
    )
    parser.add_argument(
        "--n-runs", type=int, default=100,
        help="Number of runs per experiment/condition (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for this batch"
    )
    parser.add_argument(
        "--experiments", type=str, default="all",
        help="Comma-separated list of experiments: salganik,counterfactual,variance,historical,sensitivity,all"
    )

    args = parser.parse_args()

    ensure_raw_dir()

    print("=" * 60)
    print("CULTURAL MARKET SIMULATION - BATCH RUN")
    print("=" * 60)
    print(f"Runs per experiment: {args.n_runs}")
    print(f"Seed: {args.seed or 'random'}")
    print(f"Raw data directory: {RAW_DIR}")

    start_time = datetime.now()

    experiments = args.experiments.split(",")
    if "all" in experiments:
        experiments = ["salganik", "counterfactual", "variance", "historical", "sensitivity"]

    if "salganik" in experiments:
        run_salganik_batch(args.n_runs, args.seed)

    if "counterfactual" in experiments:
        run_counterfactual_batch(args.n_runs)

    if "variance" in experiments:
        run_variance_batch(args.n_runs, args.seed)

    if "historical" in experiments:
        run_historical_batch(args.n_runs, args.seed)

    if "sensitivity" in experiments:
        run_sensitivity_batch(args.n_runs, args.seed)

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print(f"Batch complete! Runtime: {elapsed}")
    print(f"Run 'python scripts/combine_batches.py' to analyze results")
    print("=" * 60)


if __name__ == "__main__":
    main()
