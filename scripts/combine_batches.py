#!/usr/bin/env python
"""
Combine batch results and produce final analysis.

Reads raw data from results/raw/*.jsonl and produces summary statistics.
Can be run after each batch to see updated results.

Usage:
    python scripts/combine_batches.py
    python scripts/combine_batches.py --output results/analysis.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Set

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.metrics import jaccard_similarity, counterfactual_distance

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"


def read_jsonl(filepath: Path) -> List[dict]:
    """Read all records from a JSON Lines file."""
    records = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def analyze_salganik():
    """Analyze Salganik replication results."""
    records = read_jsonl(RAW_DIR / "salganik_runs.jsonl")

    if not records:
        return None

    independent = [r for r in records if r["condition"] == "independent"]
    social = [r for r in records if r["condition"] == "social"]

    if not independent or not social:
        return None

    ind_corrs = [r["quality_success_corr"] for r in independent]
    ind_ginis = [r["gini"] for r in independent]
    soc_corrs = [r["quality_success_corr"] for r in social]
    soc_ginis = [r["gini"] for r in social]

    # Compute canonical set overlaps for social condition
    soc_canonical_sets = [set(r.get("canonical_set", [])) for r in social if "canonical_set" in r]
    if len(soc_canonical_sets) > 1:
        overlaps = []
        for i, s1 in enumerate(soc_canonical_sets[:100]):  # Sample for efficiency
            for s2 in soc_canonical_sets[i+1:i+10]:
                overlaps.append(jaccard_similarity(s1, s2))
        canonical_overlap = float(np.mean(overlaps)) if overlaps else 1.0
    else:
        canonical_overlap = 1.0

    return {
        "n_independent_runs": len(independent),
        "n_social_runs": len(social),
        "independent": {
            "quality_success_corr_mean": float(np.mean(ind_corrs)),
            "quality_success_corr_std": float(np.std(ind_corrs)),
            "quality_success_corr_sem": float(np.std(ind_corrs) / np.sqrt(len(ind_corrs))),
            "gini_mean": float(np.mean(ind_ginis)),
            "gini_std": float(np.std(ind_ginis)),
        },
        "social": {
            "quality_success_corr_mean": float(np.mean(soc_corrs)),
            "quality_success_corr_std": float(np.std(soc_corrs)),
            "quality_success_corr_sem": float(np.std(soc_corrs) / np.sqrt(len(soc_corrs))),
            "gini_mean": float(np.mean(soc_ginis)),
            "gini_std": float(np.std(soc_ginis)),
        },
        "comparison": {
            "gini_ratio": float(np.mean(soc_ginis)) / float(np.mean(ind_ginis)),
            "corr_difference": float(np.mean(ind_corrs)) - float(np.mean(soc_corrs)),
            "canonical_overlap": canonical_overlap,
        },
    }


def analyze_counterfactual():
    """Analyze counterfactual results."""
    records = read_jsonl(RAW_DIR / "counterfactual_runs.jsonl")

    if not records:
        return None

    canonical_sets = [set(r["canonical_set"]) for r in records]

    # Load quality distribution
    qualities_file = RAW_DIR / "counterfactual_qualities.json"
    if qualities_file.exists():
        with open(qualities_file) as f:
            qualities = np.array(json.load(f)["qualities"])
    else:
        qualities = None

    # Compute canonical probabilities
    # Use qualities length as the authoritative producer count when available,
    # otherwise fall back to max ID found in canonical sets
    if qualities is not None:
        n_prod = len(qualities)
    else:
        all_producers = set()
        for cs in canonical_sets:
            all_producers.update(cs)
        n_prod = (max(all_producers) + 1) if all_producers else 0

    if n_prod > 0:
        canonical_counts = np.zeros(n_prod)
        for cs in canonical_sets:
            for pid in cs:
                if pid < n_prod:
                    canonical_counts[pid] += 1
        canonical_probs = canonical_counts / len(records)
    else:
        canonical_probs = np.array([])

    # Canonical probability by quality decile
    if qualities is not None and len(canonical_probs) > 0:
        quality_percentiles = np.argsort(np.argsort(qualities)) / len(qualities) * 100
        decile_probs = {}
        for decile in range(1, 11):
            lower = (decile - 1) * 10
            upper = decile * 10
            mask = (quality_percentiles >= lower) & (quality_percentiles < upper)
            if np.sum(mask) > 0:
                decile_probs[decile] = float(np.mean(canonical_probs[mask]))
            else:
                decile_probs[decile] = 0.0
    else:
        decile_probs = {}

    # Sample counterfactual distance (expensive for many sets)
    sample_size = min(200, len(canonical_sets))
    sampled_sets = canonical_sets[:sample_size]
    cf_distance = counterfactual_distance(sampled_sets)

    return {
        "n_runs": len(records),
        "counterfactual_distance": cf_distance,
        "canonical_probability_variance": float(np.var(canonical_probs)) if len(canonical_probs) > 0 else 0,
        "mean_canonical_prob": float(np.mean(canonical_probs)) if len(canonical_probs) > 0 else 0,
        "max_canonical_prob": float(np.max(canonical_probs)) if len(canonical_probs) > 0 else 0,
        "min_canonical_prob": float(np.min(canonical_probs)) if len(canonical_probs) > 0 else 0,
        "canonical_by_decile": decile_probs,
    }


def analyze_variance():
    """Analyze variance decomposition results."""
    records = read_jsonl(RAW_DIR / "variance_runs.jsonl")

    if not records:
        return None

    conditions = defaultdict(list)
    for r in records:
        conditions[r["condition"]].append(r)

    results = {}
    for cond_name, cond_records in conditions.items():
        variances = [r["canonical_variance"] for r in cond_records]
        corrs = [r["quality_canonical_corr"] for r in cond_records]
        results[cond_name] = {
            "n_runs": len(cond_records),
            "canonical_variance_mean": float(np.mean(variances)),
            "canonical_variance_std": float(np.std(variances)),
            "quality_canonical_corr_mean": float(np.mean(corrs)),
        }

    # Compute decomposition
    if "full" in results and results["full"]["canonical_variance_mean"] > 0:
        full_var = results["full"]["canonical_variance_mean"]

        social_contrib = (full_var - results.get("no_social", {}).get("canonical_variance_mean", full_var)) / full_var
        capital_contrib = (full_var - results.get("homogeneous_capital", {}).get("canonical_variance_mean", full_var)) / full_var
        quality_contrib = results.get("both_ablations", {}).get("canonical_variance_mean", 0) / full_var
        residual = 1 - social_contrib - capital_contrib - quality_contrib

        decomposition = {
            "quality": max(0, quality_contrib),
            "capital": max(0, capital_contrib),
            "social_influence": max(0, social_contrib),
            "residual": max(0, residual),
        }
    else:
        decomposition = {}

    return {
        "conditions": results,
        "decomposition": decomposition,
    }


def analyze_historical():
    """Analyze historical scenario results."""
    records = read_jsonl(RAW_DIR / "historical_runs.jsonl")

    if not records:
        return None

    quality_corrs = [r["quality_success_corr"] for r in records]
    capital_corrs = [r["capital_success_corr"] for r in records]
    canonical_sets = [set(r["canonical_set"]) for r in records]

    # Sample counterfactual distance
    sample_size = min(100, len(canonical_sets))
    cf_distance = counterfactual_distance(canonical_sets[:sample_size])

    # Canonical overlap
    if len(canonical_sets) > 1:
        overlaps = []
        for i in range(min(50, len(canonical_sets))):
            for j in range(i+1, min(i+10, len(canonical_sets))):
                overlaps.append(jaccard_similarity(canonical_sets[i], canonical_sets[j]))
        canonical_overlap = float(np.mean(overlaps)) if overlaps else 1.0
    else:
        canonical_overlap = 1.0

    return {
        "n_runs": len(records),
        "quality_success_corr_mean": float(np.mean(quality_corrs)),
        "quality_success_corr_std": float(np.std(quality_corrs)),
        "capital_success_corr_mean": float(np.mean(capital_corrs)),
        "capital_success_corr_std": float(np.std(capital_corrs)),
        "counterfactual_distance": cf_distance,
        "canonical_overlap_mean": canonical_overlap,
    }


def analyze_sensitivity():
    """Analyze sensitivity results."""
    records = read_jsonl(RAW_DIR / "sensitivity_runs.jsonl")

    if not records:
        return None

    # Group by param and level
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["param_name"]][r["level"]].append(r)

    sensitivities = {}
    for param_name, levels in grouped.items():
        level_results = {}
        for level, level_records in levels.items():
            corrs = [r["quality_success_corr"] for r in level_records]
            ginis = [r["gini"] for r in level_records]
            level_results[level] = {
                "n_runs": len(level_records),
                "value": level_records[0]["value"] if level_records else 0,
                "quality_success_corr": float(np.mean(corrs)),
                "quality_success_corr_std": float(np.std(corrs)),
                "gini": float(np.mean(ginis)),
                "gini_std": float(np.std(ginis)),
            }

        # Compute sensitivity
        if "low" in level_results and "high" in level_results and "base" in level_results:
            base_value = level_results["base"]["value"]
            if base_value != 0:
                corr_sens = (level_results["high"]["quality_success_corr"] - level_results["low"]["quality_success_corr"]) / (base_value)
                gini_sens = (level_results["high"]["gini"] - level_results["low"]["gini"]) / base_value
            else:
                corr_sens = gini_sens = 0
        else:
            corr_sens = gini_sens = 0

        sensitivities[param_name] = {
            "levels": level_results,
            "corr_sensitivity": float(corr_sens),
            "gini_sensitivity": float(gini_sens),
        }

    return {"sensitivities": sensitivities}


def print_summary(results: dict):
    """Print formatted summary of results."""
    print("\n" + "=" * 70)
    print("COMBINED ANALYSIS RESULTS")
    print("=" * 70)

    # Salganik
    if results.get("salganik"):
        sal = results["salganik"]
        print(f"\n{'─' * 70}")
        print("SALGANIK REPLICATION")
        print(f"  Total runs: {sal['n_independent_runs']} independent, {sal['n_social_runs']} social")
        print(f"\n  Independent condition:")
        print(f"    Quality-Success Correlation: {sal['independent']['quality_success_corr_mean']:.3f} ± {sal['independent']['quality_success_corr_std']:.3f}")
        print(f"    Gini: {sal['independent']['gini_mean']:.3f}")
        print(f"\n  Social condition:")
        print(f"    Quality-Success Correlation: {sal['social']['quality_success_corr_mean']:.3f} ± {sal['social']['quality_success_corr_std']:.3f}")
        print(f"    Gini: {sal['social']['gini_mean']:.3f}")
        print(f"\n  Comparison:")
        print(f"    Gini Ratio (social/independent): {sal['comparison']['gini_ratio']:.3f}")
        print(f"    Correlation Reduction: {sal['comparison']['corr_difference']:.3f}")

    # Counterfactual
    if results.get("counterfactual"):
        cf = results["counterfactual"]
        print(f"\n{'─' * 70}")
        print("COUNTERFACTUAL ANALYSIS")
        print(f"  Total runs: {cf['n_runs']}")
        print(f"  Counterfactual Distance: {cf['counterfactual_distance']:.3f}")
        print(f"    (0 = deterministic, 1 = completely random)")
        print(f"  Canonical Probability Variance: {cf['canonical_probability_variance']:.4f}")
        if cf.get("canonical_by_decile"):
            print(f"\n  Canonical Probability by Quality Decile:")
            for decile in range(1, 11):
                prob = cf["canonical_by_decile"].get(str(decile), cf["canonical_by_decile"].get(decile, 0))
                bar = "█" * int(prob * 20)
                print(f"    Decile {decile:2d}: {prob:.2f} {bar}")

    # Variance Decomposition
    if results.get("variance") and results["variance"].get("decomposition"):
        var = results["variance"]
        print(f"\n{'─' * 70}")
        print("VARIANCE DECOMPOSITION")
        total_runs = sum(c.get("n_runs", 0) for c in var.get("conditions", {}).values())
        print(f"  Total runs: {total_runs}")
        decomp = var["decomposition"]
        total = sum(decomp.values())
        if total > 0:
            for source in ["quality", "capital", "social_influence", "residual"]:
                pct = decomp.get(source, 0) / total * 100
                bar = "█" * int(pct / 5)
                print(f"    {source:20s}: {pct:5.1f}% {bar}")

    # Historical
    if results.get("historical"):
        hist = results["historical"]
        print(f"\n{'─' * 70}")
        print("HISTORICAL SCENARIO (18th-century Vienna)")
        print(f"  Total runs: {hist['n_runs']}")
        print(f"  Quality-Success Correlation: {hist['quality_success_corr_mean']:.3f} ± {hist['quality_success_corr_std']:.3f}")
        print(f"  Capital-Success Correlation: {hist['capital_success_corr_mean']:.3f} ± {hist['capital_success_corr_std']:.3f}")
        print(f"  Counterfactual Distance: {hist['counterfactual_distance']:.3f}")

    # Sensitivity
    if results.get("sensitivity"):
        sens = results["sensitivity"]["sensitivities"]
        print(f"\n{'─' * 70}")
        print("SENSITIVITY ANALYSIS")
        total_runs = sum(
            sum(lv.get("n_runs", 0) for lv in param.get("levels", {}).values())
            for param in sens.values()
        )
        print(f"  Total runs: {total_runs}")
        print(f"\n  Parameter Sensitivity (impact on quality-success correlation):")
        sorted_params = sorted(sens.keys(), key=lambda p: abs(sens[p]["corr_sensitivity"]), reverse=True)
        for param in sorted_params:
            s = sens[param]
            direction = "↑" if s["corr_sensitivity"] > 0 else "↓"
            print(f"    {param:15s}: {direction} {abs(s['corr_sensitivity']):.4f}")

    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Combine and analyze batch results")
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file for JSON results (optional)"
    )
    args = parser.parse_args()

    print("Loading raw data from", RAW_DIR)

    results = {
        "timestamp": datetime.now().isoformat(),
        "salganik": analyze_salganik(),
        "counterfactual": analyze_counterfactual(),
        "variance": analyze_variance(),
        "historical": analyze_historical(),
        "sensitivity": analyze_sensitivity(),
    }

    print_summary(results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
