#!/usr/bin/env python
"""Run additional simulations to reach power targets.

Appends new runs to existing JSONL files.
Focus on Salganik (need 352 per group, have 100).
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.market import DEFAULT_PARAMS, CulturalMarket
from cultural_market.metrics import gini_coefficient, quality_success_correlation

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"


def run_salganik_additional(n_additional=260):
    """Run additional Salganik runs, appending to existing file."""
    rng = np.random.default_rng()
    base_params = DEFAULT_PARAMS.copy()

    output_file = RAW_DIR / "salganik_runs.jsonl"

    # Independent condition
    ind_params = {**base_params, "si_gamma": 0.0}
    print(f"Running {n_additional} additional independent runs...")

    for i in tqdm(range(n_additional), desc="Independent"):
        market = CulturalMarket(ind_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]

        record = {
            "condition": "independent",
            "quality_success_corr": float(quality_success_correlation(qualities, successes)),
            "gini": float(gini_coefficient(successes)),
            "seed": int(rng.integers(0, 2**31)),
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    # Social condition
    print(f"\nRunning {n_additional} additional social runs...")

    for i in tqdm(range(n_additional), desc="Social"):
        market = CulturalMarket(base_params)
        market.initialize(seed=int(rng.integers(0, 2**31)))
        market.run()

        qualities = [p.quality for p in market.producers]
        successes = [p.cumulative_success for p in market.producers]
        canonical_set = market.get_canonical_set()

        record = {
            "condition": "social",
            "quality_success_corr": float(quality_success_correlation(qualities, successes)),
            "gini": float(gini_coefficient(successes)),
            "canonical_set": list(canonical_set),
            "seed": int(rng.integers(0, 2**31)),
        }
        with open(output_file, "a") as f:
            f.write(json.dumps(record) + "\n")


def run_variance_additional(n_additional=100):
    """Run additional variance decomposition runs."""
    rng = np.random.default_rng()
    base_params = DEFAULT_PARAMS.copy()

    output_file = RAW_DIR / "variance_runs.jsonl"

    conditions = {
        "full": base_params,
        "no_social": {**base_params, "si_gamma": 0.0},
        "homogeneous_capital": {**base_params, "capital_std": 0.01},
        "both_ablations": {**base_params, "si_gamma": 0.0, "capital_std": 0.01},
    }

    for cond_name, cond_params in conditions.items():
        print(f"\nRunning {n_additional} additional {cond_name} runs...")

        for i in tqdm(range(n_additional), desc=cond_name):
            market = CulturalMarket(cond_params)
            market.initialize(seed=int(rng.integers(0, 2**31)))
            market.run()

            qualities = [p.quality for p in market.producers]
            canonical = [float(p.canonical) for p in market.producers]

            record = {
                "condition": cond_name,
                "canonical_variance": float(np.var(canonical)),
                "quality_canonical_corr": float(quality_success_correlation(qualities, canonical)),
                "seed": int(rng.integers(0, 2**31)),
            }
            with open(output_file, "a") as f:
                f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    print("Running additional simulations for statistical power...")
    print(f"Raw data directory: {RAW_DIR}")

    run_salganik_additional(260)  # 100 + 260 = 360 per group (> 352 needed)
    run_variance_additional(100)  # 100 + 100 = 200 per condition

    print("\nDone! Run combine_batches.py and generate_analysis.py to update results.")
