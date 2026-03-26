#!/usr/bin/env python
"""CLI script for running experiments."""

import argparse
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

EXPERIMENTS = {
    "salganik": experiment_salganik_replication,
    "counterfactual": experiment_counterfactual,
    "variance": experiment_variance_decomposition,
    "historical": experiment_historical_scenario,
    "sensitivity": experiment_sensitivity,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run cultural market simulation experiments"
    )
    parser.add_argument(
        "experiment",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        help="Experiment to run",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=100,
        help="Number of simulation runs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment == "all":
        experiments_to_run = list(EXPERIMENTS.keys())
    else:
        experiments_to_run = [args.experiment]

    for exp_name in experiments_to_run:
        print(f"\n{'='*50}")
        print(f"Running experiment: {exp_name}")
        print("=" * 50)

        exp_func = EXPERIMENTS[exp_name]

        # Handle different argument signatures
        if exp_name == "counterfactual":
            result = exp_func(n_runs=args.n_runs, quality_seed=42)
        elif exp_name == "variance":
            result = exp_func(n_runs=min(args.n_runs, 50), seed=args.seed)
        elif exp_name == "sensitivity":
            result = exp_func(n_runs=min(args.n_runs, 20), seed=args.seed)
        else:
            result = exp_func(n_runs=args.n_runs, seed=args.seed)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{exp_name}_{timestamp}.json"

        # Convert result to serializable format
        result_dict = {
            "name": result.name,
            "params": {
                k: float(v) if isinstance(v, (int, float)) else v
                for k, v in result.params.items()
            },
            "metrics": result.metrics,
        }

        with open(output_file, "w") as f:
            json.dump(result_dict, f, indent=2, default=str)

        print(f"\nResults saved to {output_file}")
        print("\nKey metrics:")
        for key, value in result.metrics.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    if isinstance(v, dict):
                        print(f"    {k}: ...")
                    else:
                        print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
