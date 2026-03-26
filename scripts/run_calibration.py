#!/usr/bin/env python
"""CLI script for running calibration."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.calibration import calibrate, load_calibration_targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate cultural market simulation against Salganik data"
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=5,
        help="Simulation runs per parameter evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/calibration_results.json",
        help="Output file for calibration results",
    )

    args = parser.parse_args()

    print("Loading calibration targets...")
    targets = load_calibration_targets()
    print(f"Loaded {len(targets)} calibration targets")

    for target in targets:
        print(f"  - {target.name}: {target.value} [{target.range_low}, {target.range_high}]")

    print("\nRunning calibration (this may take a while)...")
    result = calibrate(n_runs=args.n_runs, seed=args.seed)

    print("\nCalibration complete!")
    print(f"Final loss: {result['final_loss']:.4f}")
    print("\nCalibrated parameters:")
    for name, value in result["optimized_values"].items():
        print(f"  {name}: {value:.4f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
