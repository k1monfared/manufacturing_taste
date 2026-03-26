"""Visualization functions for simulation results.

Includes:
- Quality vs success scatterplots
- Gini/inequality plots
- Counterfactual heatmaps
- Sensitivity tornado diagrams
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .experiments import ExperimentResult


def plot_quality_success_scatter(
    qualities: np.ndarray,
    successes: np.ndarray,
    canonical: np.ndarray | None = None,
    title: str = "Quality vs Success",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot quality vs cumulative success scatterplot.

    Args:
        qualities: Array of quality values
        successes: Array of success values
        canonical: Boolean array for canonical status (optional)
        title: Plot title
        ax: Matplotlib axes (created if None)

    Returns:
        Matplotlib axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    if canonical is not None:
        colors = ["blue" if c else "gray" for c in canonical]
        alphas = [0.8 if c else 0.3 for c in canonical]
        for q, s, col, alpha in zip(qualities, successes, colors, alphas):
            ax.scatter(q, s, c=col, alpha=alpha, s=20)
    else:
        ax.scatter(qualities, successes, alpha=0.5, s=20)

    ax.set_xlabel("Intrinsic Quality")
    ax.set_ylabel("Cumulative Success")
    ax.set_title(title)

    # Add correlation annotation
    corr = np.corrcoef(qualities, successes)[0, 1]
    ax.annotate(
        f"r = {corr:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        verticalalignment="top",
    )

    return ax


def plot_condition_comparison(
    result: "ExperimentResult",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plot comparison of independent vs social conditions.

    Args:
        result: ExperimentResult from salganik_replication
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    metrics = result.metrics

    # Correlation comparison
    ax = axes[0]
    conditions = ["Independent", "Social"]
    corrs = [
        metrics["independent"]["quality_success_corr_mean"],
        metrics["social"]["quality_success_corr_mean"],
    ]
    errs = [
        metrics["independent"]["quality_success_corr_std"],
        metrics["social"]["quality_success_corr_std"],
    ]

    ax.bar(conditions, corrs, yerr=errs, capsize=5, color=["steelblue", "coral"])
    ax.set_ylabel("Quality-Success Correlation")
    ax.set_title("Correlation by Condition")
    ax.set_ylim(0, 1)

    # Gini comparison
    ax = axes[1]
    ginis = [
        metrics["independent"]["gini_mean"],
        metrics["social"]["gini_mean"],
    ]
    errs = [
        metrics["independent"]["gini_std"],
        metrics["social"]["gini_std"],
    ]

    ax.bar(conditions, ginis, yerr=errs, capsize=5, color=["steelblue", "coral"])
    ax.set_ylabel("Gini Coefficient")
    ax.set_title("Inequality by Condition")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    return fig


def plot_counterfactual_summary(
    result: "ExperimentResult",
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot summary of counterfactual analysis.

    Args:
        result: ExperimentResult from counterfactual experiment
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.text(
        0.5,
        0.5,
        f"Counterfactual Distance: {result.metrics['counterfactual_distance']:.3f}\n"
        f"Canonical Prob Variance: {result.metrics['canonical_probability_variance']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
    )
    ax.set_title("Counterfactual Analysis Summary")
    ax.axis("off")

    return fig


def plot_variance_decomposition(
    result: "ExperimentResult",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Plot variance decomposition pie chart.

    Args:
        result: ExperimentResult from variance_decomposition
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    decomp = result.metrics["decomposition"]

    labels = ["Quality", "Capital", "Social Influence", "Residual"]
    sizes = [
        decomp["quality"],
        decomp["capital"],
        decomp["social_influence"],
        decomp["residual"],
    ]
    colors = ["#66b3ff", "#ff9999", "#99ff99", "#ffcc99"]

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0.05, 0.05, 0),
    )
    ax.set_title("Sources of Canonical Variance")

    return fig


def plot_sensitivity_tornado(
    result: "ExperimentResult",
    metric: str = "quality_success_corr",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot tornado diagram for sensitivity analysis.

    Args:
        result: ExperimentResult from sensitivity experiment
        metric: Which metric to show ('quality_success_corr' or 'gini')
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    sensitivities = result.metrics["sensitivities"]

    param_names = list(sensitivities.keys())
    low_values = []
    high_values = []
    base_value = None

    for param in param_names:
        levels = sensitivities[param]["levels"]
        low_values.append(levels["low"][metric])
        high_values.append(levels["high"][metric])
        if base_value is None:
            base_value = levels["base"][metric]

    y_pos = np.arange(len(param_names))

    # Calculate deviations from base
    low_dev = np.array(low_values) - base_value
    high_dev = np.array(high_values) - base_value

    # Sort by total range
    ranges = np.abs(high_dev - low_dev)
    sort_idx = np.argsort(ranges)

    param_names = [param_names[i] for i in sort_idx]
    low_dev = low_dev[sort_idx]
    high_dev = high_dev[sort_idx]

    ax.barh(y_pos, low_dev, align="center", color="steelblue", label="-50%")
    ax.barh(y_pos, high_dev, align="center", color="coral", label="+50%")

    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names)
    ax.set_xlabel(f"Change in {metric}")
    ax.set_title("Sensitivity Analysis")
    ax.legend()

    plt.tight_layout()
    return fig


def plot_canonical_by_decile(
    result: "ExperimentResult",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plot canonical probability by quality decile.

    Args:
        result: ExperimentResult from counterfactual experiment
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    decile_probs = result.metrics["canonical_by_decile"]

    deciles = list(range(1, 11))
    probs = [decile_probs.get(d, 0) for d in deciles]

    ax.bar(deciles, probs, color="steelblue", edgecolor="black")
    ax.set_xlabel("Quality Decile")
    ax.set_ylabel("Probability of Canonical Status")
    ax.set_title("Canonical Probability by Quality Decile")
    ax.set_xticks(deciles)
    ax.set_ylim(0, 1)

    return fig
