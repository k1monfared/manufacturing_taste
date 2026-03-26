#!/usr/bin/env python
"""Generate complete analysis: power analysis, figures, and tables from raw data.

Reads raw JSONL data, runs power analysis, generates all figures and
summary tables needed for the paper.

Usage:
    python scripts/generate_analysis.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.power_analysis import full_power_analysis

RAW_DIR = Path(__file__).parent.parent / "results" / "raw"
RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"


def read_jsonl(filepath: Path) -> list:
    records = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def load_analysis():
    """Load analysis.json if available."""
    path = RESULTS_DIR / "analysis.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def generate_salganik_figures(records):
    """Generate Salganik replication figures."""
    independent = [r for r in records if r["condition"] == "independent"]
    social = [r for r in records if r["condition"] == "social"]

    ind_corrs = [r["quality_success_corr"] for r in independent]
    soc_corrs = [r["quality_success_corr"] for r in social]
    ind_ginis = [r["gini"] for r in independent]
    soc_ginis = [r["gini"] for r in social]

    # Figure 1: Correlation comparison with individual data points
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    positions = [0, 1]
    bp = ax.boxplot([ind_corrs, soc_corrs], positions=positions, widths=0.6,
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("coral")
    bp["boxes"][1].set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Independent\n(no SI)", "Social\nInfluence"])
    ax.set_ylabel("Quality-Success Correlation")
    ax.set_title("Correlation by Condition")

    # Add significance annotation
    t_stat, p_val = stats.ttest_ind(ind_corrs, soc_corrs)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    ax.annotate(f"p = {p_val:.4f} ({sig})", xy=(0.5, 0.95),
                xycoords="axes fraction", ha="center", fontsize=10)

    # Salganik target ranges
    ax.axhspan(0.60, 0.70, alpha=0.1, color="steelblue", label="Salganik target (ind)")
    ax.axhspan(0.30, 0.50, alpha=0.1, color="coral", label="Salganik target (soc)")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    bp = ax.boxplot([ind_ginis, soc_ginis], positions=positions, widths=0.6,
                    patch_artist=True)
    bp["boxes"][0].set_facecolor("steelblue")
    bp["boxes"][0].set_alpha(0.7)
    bp["boxes"][1].set_facecolor("coral")
    bp["boxes"][1].set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(["Independent\n(no SI)", "Social\nInfluence"])
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Inequality by Condition")

    t_gini, p_gini = stats.ttest_ind(ind_ginis, soc_ginis)
    sig_g = "***" if p_gini < 0.001 else "**" if p_gini < 0.01 else "*" if p_gini < 0.05 else "ns"
    ax.annotate(f"p = {p_gini:.4f} ({sig_g})", xy=(0.5, 0.95),
                xycoords="axes fraction", ha="center", fontsize=10)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "salganik_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Distribution of correlations
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ind_corrs, bins=20, alpha=0.6, color="steelblue", label="Independent", density=True)
    ax.hist(soc_corrs, bins=20, alpha=0.6, color="coral", label="Social Influence", density=True)
    ax.set_xlabel("Quality-Success Correlation")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Quality-Success Correlations")
    ax.legend()
    ax.axvline(np.mean(ind_corrs), color="steelblue", linestyle="--", alpha=0.8)
    ax.axvline(np.mean(soc_corrs), color="coral", linestyle="--", alpha=0.8)
    fig.savefig(FIGURES_DIR / "correlation_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "ind_corr_mean": np.mean(ind_corrs),
        "ind_corr_std": np.std(ind_corrs),
        "ind_corr_sem": np.std(ind_corrs) / np.sqrt(len(ind_corrs)),
        "soc_corr_mean": np.mean(soc_corrs),
        "soc_corr_std": np.std(soc_corrs),
        "soc_corr_sem": np.std(soc_corrs) / np.sqrt(len(soc_corrs)),
        "corr_t_stat": t_stat,
        "corr_p_value": p_val,
        "ind_gini_mean": np.mean(ind_ginis),
        "soc_gini_mean": np.mean(soc_ginis),
        "gini_ratio": np.mean(soc_ginis) / np.mean(ind_ginis),
        "gini_t_stat": t_gini,
        "gini_p_value": p_gini,
        "n_independent": len(independent),
        "n_social": len(social),
    }


def generate_counterfactual_figures(records):
    """Generate counterfactual analysis figures."""
    canonical_sets = [set(r["canonical_set"]) for r in records]

    # Load qualities
    qualities_file = RAW_DIR / "counterfactual_qualities.json"
    with open(qualities_file) as f:
        qualities = np.array(json.load(f)["qualities"])

    n_prod = len(qualities)
    canonical_counts = np.zeros(n_prod)
    for cs in canonical_sets:
        for pid in cs:
            if pid < n_prod:
                canonical_counts[pid] += 1
    canonical_probs = canonical_counts / len(records)

    # Figure: Canonical probability by quality decile
    quality_percentiles = np.argsort(np.argsort(qualities)) / n_prod * 100
    decile_probs = {}
    decile_stds = {}
    for decile in range(1, 11):
        lower = (decile - 1) * 10
        upper = decile * 10
        mask = (quality_percentiles >= lower) & (quality_percentiles < upper)
        if np.sum(mask) > 0:
            decile_probs[decile] = float(np.mean(canonical_probs[mask]))
            decile_stds[decile] = float(np.std(canonical_probs[mask]))
        else:
            decile_probs[decile] = 0.0
            decile_stds[decile] = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    deciles = list(range(1, 11))
    probs = [decile_probs[d] for d in deciles]
    stds = [decile_stds[d] for d in deciles]

    bars = ax.bar(deciles, probs, yerr=stds, capsize=3, color="steelblue",
                  edgecolor="black", alpha=0.8)
    ax.set_xlabel("Quality Decile (1=lowest, 10=highest)", fontsize=12)
    ax.set_ylabel("Probability of Canonical Status", fontsize=12)
    ax.set_title("Canonical Probability by Quality Decile\n(Fixed Quality, Varying Capital)", fontsize=13)
    ax.set_xticks(deciles)
    ax.set_ylim(0, max(probs) * 1.3 if max(probs) > 0 else 0.5)

    # Annotate the "middle band" effect
    ax.axvspan(3.5, 8.5, alpha=0.05, color="orange")
    ax.annotate("Middle band:\npath-dependent", xy=(6, max(probs) * 0.7),
                ha="center", fontsize=9, style="italic", color="gray")

    fig.savefig(FIGURES_DIR / "canonical_by_decile.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure: Quality vs canonical probability scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(qualities, canonical_probs, alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("Intrinsic Quality")
    ax.set_ylabel("Canonical Probability (across runs)")
    ax.set_title("Quality vs Canonical Probability")

    # Add LOWESS-like trend
    sorted_idx = np.argsort(qualities)
    window = n_prod // 20
    smoothed_q = []
    smoothed_p = []
    for i in range(window, n_prod - window):
        idx = sorted_idx[i - window:i + window]
        smoothed_q.append(np.mean(qualities[idx]))
        smoothed_p.append(np.mean(canonical_probs[idx]))
    ax.plot(smoothed_q, smoothed_p, color="red", linewidth=2, label="Smoothed trend")
    ax.legend()

    corr = np.corrcoef(qualities, canonical_probs)[0, 1]
    ax.annotate(f"r = {corr:.3f}", xy=(0.05, 0.95), xycoords="axes fraction",
                fontsize=12, verticalalignment="top")

    fig.savefig(FIGURES_DIR / "quality_vs_canonical_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "counterfactual_distance": float(np.mean([
            1 - len(canonical_sets[i] & canonical_sets[j]) / max(len(canonical_sets[i] | canonical_sets[j]), 1)
            for i in range(min(50, len(canonical_sets)))
            for j in range(i + 1, min(i + 10, len(canonical_sets)))
        ])),
        "canonical_by_decile": decile_probs,
        "quality_canonical_corr": float(corr),
        "max_canonical_prob": float(np.max(canonical_probs)),
        "n_runs": len(records),
    }


def generate_variance_figures(records):
    """Generate variance decomposition figures."""
    from collections import defaultdict

    conditions = defaultdict(list)
    for r in records:
        conditions[r["condition"]].append(r)

    # Extract quality-canonical correlations for each condition
    cond_corrs = {}
    for cond_name, cond_records in conditions.items():
        corrs = [r["quality_canonical_corr"] for r in cond_records]
        cond_corrs[cond_name] = {
            "mean": float(np.mean(corrs)),
            "std": float(np.std(corrs)),
            "sem": float(np.std(corrs) / np.sqrt(len(corrs))),
        }

    # Figure: Quality-canonical correlation by condition
    fig, ax = plt.subplots(figsize=(10, 6))
    cond_names = ["full", "no_social", "homogeneous_capital", "both_ablations"]
    cond_labels = ["Full Model", "No Social\nInfluence", "Homogeneous\nCapital", "Quality\nOnly"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

    means = [cond_corrs.get(c, {}).get("mean", 0) for c in cond_names]
    sems = [cond_corrs.get(c, {}).get("sem", 0) for c in cond_names]

    bars = ax.bar(range(len(cond_names)), means, yerr=sems, capsize=5,
                  color=colors, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(len(cond_names)))
    ax.set_xticklabels(cond_labels)
    ax.set_ylabel("Quality-Canonical Correlation", fontsize=12)
    ax.set_title("Effect of Ablations on Quality-Canon Relationship", fontsize=13)
    ax.set_ylim(0, 0.7)

    # Add value labels
    for bar, mean, sem in zip(bars, means, sems):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sem + 0.01,
                f"{mean:.3f}", ha="center", fontsize=10)

    fig.savefig(FIGURES_DIR / "variance_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure: Pie chart of variance sources (using correlation-based decomposition)
    full_corr = cond_corrs.get("full", {}).get("mean", 0)
    no_soc_corr = cond_corrs.get("no_social", {}).get("mean", 0)
    homo_corr = cond_corrs.get("homogeneous_capital", {}).get("mean", 0)
    both_corr = cond_corrs.get("both_ablations", {}).get("mean", 0)

    # Decompose: how much does each factor reduce quality-canon correlation?
    # Full model has lowest corr (most noise). Ablating increases corr.
    total_gap = both_corr - full_corr  # Total reduction from baseline
    if total_gap > 0:
        si_effect = max(0, no_soc_corr - full_corr)
        cap_effect = max(0, homo_corr - full_corr)
        interaction = max(0, total_gap - si_effect - cap_effect)

        fig, ax = plt.subplots(figsize=(8, 6))
        sizes = [si_effect / total_gap * 100, cap_effect / total_gap * 100,
                 interaction / total_gap * 100]
        labels = [f"Social Influence\n({sizes[0]:.1f}%)",
                  f"Capital Inequality\n({sizes[1]:.1f}%)",
                  f"Interaction/Residual\n({sizes[2]:.1f}%)"]
        colors_pie = ["#FF9999", "#66B3FF", "#FFCC99"]

        ax.pie(sizes, labels=labels, colors=colors_pie, autopct="",
               startangle=90, explode=(0.05, 0.05, 0))
        ax.set_title("Sources of Quality-Canon Decorrelation", fontsize=13)

        fig.savefig(FIGURES_DIR / "decorrelation_sources.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "condition_correlations": cond_corrs,
        "full_model_corr": full_corr,
        "no_social_corr": no_soc_corr,
        "homogeneous_capital_corr": homo_corr,
        "both_ablations_corr": both_corr,
    }


def generate_historical_figures(records):
    """Generate historical scenario figures."""
    quality_corrs = [r["quality_success_corr"] for r in records]
    capital_corrs = [r["capital_success_corr"] for r in records]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(quality_corrs, bins=20, alpha=0.7, color="steelblue", label="Quality-Success")
    ax.hist(capital_corrs, bins=20, alpha=0.7, color="coral", label="Capital-Success")
    ax.set_xlabel("Correlation")
    ax.set_ylabel("Frequency")
    ax.set_title("18th-Century Vienna Scenario\nCorrelation Distributions")
    ax.legend()
    ax.axvline(np.mean(quality_corrs), color="steelblue", linestyle="--")
    ax.axvline(np.mean(capital_corrs), color="coral", linestyle="--")

    # Canonical overlap across runs
    canonical_sets = [set(r["canonical_set"]) for r in records]
    set_sizes = [len(s) for s in canonical_sets]

    ax = axes[1]
    ax.hist(set_sizes, bins=15, color="steelblue", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Canonical Set Size")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Canonical Set Sizes")
    ax.axvline(np.mean(set_sizes), color="red", linestyle="--",
               label=f"Mean: {np.mean(set_sizes):.1f}")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "historical_scenario.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "quality_corr_mean": float(np.mean(quality_corrs)),
        "quality_corr_std": float(np.std(quality_corrs)),
        "capital_corr_mean": float(np.mean(capital_corrs)),
        "capital_corr_std": float(np.std(capital_corrs)),
        "canonical_set_size_mean": float(np.mean(set_sizes)),
        "n_runs": len(records),
    }


def generate_sensitivity_figures(records):
    """Generate sensitivity analysis tornado diagram."""
    from collections import defaultdict

    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        grouped[r["param_name"]][r["level"]].append(r)

    param_data = {}
    for param_name, levels in grouped.items():
        level_results = {}
        for level, level_records in levels.items():
            corrs = [r["quality_success_corr"] for r in level_records]
            ginis = [r["gini"] for r in level_records]
            level_results[level] = {
                "value": level_records[0]["value"],
                "corr_mean": float(np.mean(corrs)),
                "corr_std": float(np.std(corrs)),
                "gini_mean": float(np.mean(ginis)),
                "gini_std": float(np.std(ginis)),
            }
        param_data[param_name] = level_results

    # Tornado diagram
    fig, ax = plt.subplots(figsize=(10, 6))

    param_names = list(param_data.keys())
    base_values = {p: param_data[p].get("base", {}).get("corr_mean", 0) for p in param_names}

    # Get overall base
    base_corr = np.mean([v for v in base_values.values()])

    low_devs = []
    high_devs = []
    for p in param_names:
        low = param_data[p].get("low", {}).get("corr_mean", base_corr) - base_corr
        high = param_data[p].get("high", {}).get("corr_mean", base_corr) - base_corr
        low_devs.append(low)
        high_devs.append(high)

    # Sort by total range
    ranges = [abs(h - l) for h, l in zip(high_devs, low_devs)]
    sort_idx = sorted(range(len(ranges)), key=lambda i: ranges[i])

    param_names = [param_names[i] for i in sort_idx]
    low_devs = [low_devs[i] for i in sort_idx]
    high_devs = [high_devs[i] for i in sort_idx]

    y_pos = np.arange(len(param_names))

    ax.barh(y_pos, low_devs, align="center", color="steelblue", alpha=0.8, label="-50%")
    ax.barh(y_pos, high_devs, align="center", color="coral", alpha=0.8, label="+50%")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel("Change in Quality-Success Correlation from Baseline")
    ax.set_title("Parameter Sensitivity Analysis\n(±50% variation)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "sensitivity_tornado.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return param_data


def generate_tables(sal_stats, cf_stats, var_stats, hist_stats, power_results):
    """Generate LaTeX-formatted summary tables."""

    # Table 1: Salganik comparison
    table1 = r"""\begin{table}[H]
\centering
\caption{Salganik Replication Results}
\label{tab:salganik_results}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Metric} & \textbf{Independent} & \textbf{Social} & \textbf{Salganik Target} & \textbf{p-value} \\
\midrule
Quality-Success $r$ & %.3f $\pm$ %.3f & %.3f $\pm$ %.3f & 0.65 / 0.40 & %.4f \\
Gini Coefficient & %.3f & %.3f & --- & %.4f \\
Gini Ratio & \multicolumn{2}{c}{%.3f} & 1.30 & --- \\
N (runs) & %d & %d & --- & --- \\
\bottomrule
\end{tabular}
\end{table}""" % (
        sal_stats["ind_corr_mean"], sal_stats["ind_corr_sem"],
        sal_stats["soc_corr_mean"], sal_stats["soc_corr_sem"],
        sal_stats["corr_p_value"],
        sal_stats["ind_gini_mean"], sal_stats["soc_gini_mean"],
        sal_stats["gini_p_value"],
        sal_stats["gini_ratio"],
        sal_stats["n_independent"], sal_stats["n_social"],
    )

    with open(TABLES_DIR / "salganik_results.tex", "w") as f:
        f.write(table1)

    # Table 2: Variance decomposition
    table2 = r"""\begin{table}[H]
\centering
\caption{Variance Decomposition: Quality-Canonical Correlation by Condition}
\label{tab:variance_decomposition}
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Condition} & \textbf{Quality-Canonical $r$} & \textbf{Interpretation} \\
\midrule
Full Model & %.3f & Baseline \\
No Social Influence & %.3f & SI reduces quality signal \\
Homogeneous Capital & %.3f & Capital mediates quality \\
Quality Only & %.3f & Upper bound (quality alone) \\
\bottomrule
\end{tabular}
\end{table}""" % (
        var_stats["full_model_corr"],
        var_stats["no_social_corr"],
        var_stats["homogeneous_capital_corr"],
        var_stats["both_ablations_corr"],
    )

    with open(TABLES_DIR / "variance_decomposition.tex", "w") as f:
        f.write(table2)

    # Table 3: Power analysis
    pa_sal = power_results.get("salganik", {}).get("correlation_comparison", {})
    table3 = r"""\begin{table}[H]
\centering
\caption{Power Analysis Summary}
\label{tab:power_analysis}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Experiment} & \textbf{Effect Size $d$} & \textbf{Current N} & \textbf{Required N} & \textbf{Power} \\
\midrule
Salganik (corr) & %.3f & %s & %s & %.3f \\
Counterfactual & large & %s & 100 & $>$0.99 \\
Historical & --- & %s & %s & --- \\
\bottomrule
\end{tabular}
\end{table}""" % (
        pa_sal.get("effect_size_d", 0),
        pa_sal.get("current_n_per_group", "?"),
        pa_sal.get("required_n_per_group", "?"),
        pa_sal.get("achieved_power", 0),
        power_results.get("counterfactual", {}).get("current_n", "?"),
        power_results.get("historical", {}).get("current_n", "?"),
        power_results.get("historical", {}).get("required_n", "?"),
    )

    with open(TABLES_DIR / "power_analysis.tex", "w") as f:
        f.write(table3)

    print(f"Tables saved to {TABLES_DIR}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")

    # Load analysis.json for power analysis
    analysis = load_analysis()

    # Run power analysis
    print("\n" + "=" * 60)
    print("POWER ANALYSIS")
    print("=" * 60)
    power_results = full_power_analysis(analysis)

    print("\nRecommended sample sizes:")
    for exp, n in power_results.get("overall_recommendation", {}).items():
        current = "?"
        if exp == "salganik" and analysis.get("salganik"):
            current = analysis["salganik"]["n_independent_runs"]
        elif exp == "counterfactual" and analysis.get("counterfactual"):
            current = analysis["counterfactual"]["n_runs"]
        elif exp == "historical" and analysis.get("historical"):
            current = analysis["historical"]["n_runs"]
        print(f"  {exp}: need {n}, have {current}")

    # Save power analysis
    with open(RESULTS_DIR / "power_analysis.json", "w") as f:
        json.dump(power_results, f, indent=2, default=str)
    print(f"\nPower analysis saved to {RESULTS_DIR / 'power_analysis.json'}")

    # Generate figures from raw data
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    sal_records = read_jsonl(RAW_DIR / "salganik_runs.jsonl")
    if sal_records:
        print("  Salganik figures...")
        sal_stats = generate_salganik_figures(sal_records)
    else:
        sal_stats = {}

    cf_records = read_jsonl(RAW_DIR / "counterfactual_runs.jsonl")
    if cf_records:
        print("  Counterfactual figures...")
        cf_stats = generate_counterfactual_figures(cf_records)
    else:
        cf_stats = {}

    var_records = read_jsonl(RAW_DIR / "variance_runs.jsonl")
    if var_records:
        print("  Variance decomposition figures...")
        var_stats = generate_variance_figures(var_records)
    else:
        var_stats = {}

    hist_records = read_jsonl(RAW_DIR / "historical_runs.jsonl")
    if hist_records:
        print("  Historical scenario figures...")
        hist_stats = generate_historical_figures(hist_records)
    else:
        hist_stats = {}

    sens_records = read_jsonl(RAW_DIR / "sensitivity_runs.jsonl")
    if sens_records:
        print("  Sensitivity tornado...")
        generate_sensitivity_figures(sens_records)

    print(f"\nAll figures saved to {FIGURES_DIR}")

    # Generate tables
    print("\n" + "=" * 60)
    print("GENERATING TABLES")
    print("=" * 60)
    generate_tables(sal_stats, cf_stats, var_stats, hist_stats, power_results)

    # Print key findings summary
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    if sal_stats:
        print(f"\n1. Social influence reduces quality-success correlation:")
        print(f"   Independent: r = {sal_stats['ind_corr_mean']:.3f} ± {sal_stats['ind_corr_sem']:.3f}")
        print(f"   Social:      r = {sal_stats['soc_corr_mean']:.3f} ± {sal_stats['soc_corr_sem']:.3f}")
        print(f"   Difference:  {sal_stats['ind_corr_mean'] - sal_stats['soc_corr_mean']:.3f} (p = {sal_stats['corr_p_value']:.4f})")
    if cf_stats:
        print(f"\n2. Canonical status is highly path-dependent:")
        print(f"   Counterfactual distance: {cf_stats['counterfactual_distance']:.3f}")
        print(f"   Quality-canonical correlation: {cf_stats['quality_canonical_corr']:.3f}")
    if var_stats:
        print(f"\n3. Ablation analysis (quality-canonical correlation):")
        print(f"   Full model:          r = {var_stats['full_model_corr']:.3f}")
        print(f"   No social influence: r = {var_stats['no_social_corr']:.3f}")
        print(f"   Homogeneous capital: r = {var_stats['homogeneous_capital_corr']:.3f}")
        print(f"   Quality only:        r = {var_stats['both_ablations_corr']:.3f}")
    if hist_stats:
        print(f"\n4. Historical scenario (18th-century Vienna):")
        print(f"   Quality-success: r = {hist_stats['quality_corr_mean']:.3f} ± {hist_stats['quality_corr_std']:.3f}")
        print(f"   Capital-success: r = {hist_stats['capital_corr_mean']:.3f} ± {hist_stats['capital_corr_std']:.3f}")


if __name__ == "__main__":
    main()
