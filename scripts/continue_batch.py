#!/usr/bin/env python
"""
Continue the batch run from where it stopped.

Resumes:
- Variance: 1 more "full" run, then no_social, homogeneous_capital, both_ablations
- Historical: full 100 runs
- Sensitivity: full runs
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.market import CulturalMarket, DEFAULT_PARAMS
from cultural_market.metrics import (
    gini_coefficient,
    quality_success_correlation,
)

SIMULATION_PARAMS = {
    "n_producers": DEFAULT_PARAMS["n_producers"],
    "n_consumers": DEFAULT_PARAMS["n_consumers"],
    "t_canon": DEFAULT_PARAMS["t_canon"],
    "t_active": DEFAULT_PARAMS["t_active"],
}

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"


def append_jsonl(filepath: Path, records: list):
    with open(filepath, "a") as f:
        for record in records:
            f.write(json.dumps(record, default=str) + "\n")


def run_variance_continuation(rng):
    """Continue variance batch from where it stopped."""
    print("\n" + "=" * 60)
    print("VARIANCE DECOMPOSITION - CONTINUATION")
    print("=" * 60)

    results_file = RAW_DIR / "variance_runs.jsonl"

    # Remaining conditions to run
    conditions = {
        "full": (SIMULATION_PARAMS, 1),  # Only 1 more needed
        "no_social": ({**SIMULATION_PARAMS, "si_gamma": 0.0}, 100),
        "homogeneous_capital": ({**SIMULATION_PARAMS, "capital_std": 0.01}, 100),
        "both_ablations": ({**SIMULATION_PARAMS, "si_gamma": 0.0, "capital_std": 0.01}, 100),
    }

    for cond_name, (cond_params, n_runs) in conditions.items():
        print(f"Running {cond_name} ({n_runs} runs)...")

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


def run_historical_batch(n_runs: int, rng):
    """Run historical scenario batch."""
    print("\n" + "=" * 60)
    print("HISTORICAL SCENARIO BATCH")
    print("=" * 60)

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


def run_sensitivity_batch(n_runs: int, rng):
    """Run sensitivity analysis batch."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS BATCH")
    print("=" * 60)

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
    print("=" * 60)
    print("CULTURAL MARKET SIMULATION - CONTINUATION RUN")
    print("=" * 60)
    print(f"Started: {datetime.now()}")
    print(f"Raw data directory: {RAW_DIR}")

    start_time = datetime.now()
    rng = np.random.default_rng()

    # Continue variance (1 full + 300 other conditions)
    run_variance_continuation(rng)

    # Run historical (100 runs)
    run_historical_batch(100, rng)

    # Run sensitivity (6 params × 3 levels × 100 runs = 1800 runs)
    run_sensitivity_batch(100, rng)

    elapsed = datetime.now() - start_time
    print("\n" + "=" * 60)
    print(f"Continuation complete! Runtime: {elapsed}")
    print(f"Run 'python scripts/combine_batches.py' to analyze results")
    print("=" * 60)


if __name__ == "__main__":
    main()
