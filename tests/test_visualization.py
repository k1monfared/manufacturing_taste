"""Tests for the visualization module."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.experiments import ExperimentResult
from cultural_market.visualization import (
    plot_canonical_by_decile,
    plot_condition_comparison,
    plot_counterfactual_summary,
    plot_quality_success_scatter,
    plot_sensitivity_tornado,
    plot_variance_decomposition,
)


@pytest.fixture
def salganik_result():
    return ExperimentResult(
        name="salganik_replication",
        params={},
        metrics={
            "independent": {
                "quality_success_corr_mean": 0.6,
                "quality_success_corr_std": 0.1,
                "gini_mean": 0.45,
                "gini_std": 0.03,
            },
            "social": {
                "quality_success_corr_mean": 0.4,
                "quality_success_corr_std": 0.15,
                "gini_mean": 0.55,
                "gini_std": 0.05,
            },
            "comparison": {
                "gini_ratio": 1.22,
                "corr_difference": 0.2,
                "canonical_overlap": 0.3,
            },
        },
    )


@pytest.fixture
def counterfactual_result():
    return ExperimentResult(
        name="counterfactual",
        params={},
        metrics={
            "counterfactual_distance": 0.85,
            "canonical_probability_variance": 0.01,
            "canonical_by_decile": {d: d * 0.03 for d in range(1, 11)},
            "mean_canonical_prob": 0.05,
            "max_canonical_prob": 0.4,
        },
    )


@pytest.fixture
def variance_result():
    return ExperimentResult(
        name="variance_decomposition",
        params={},
        metrics={
            "decomposition": {
                "quality": 0.4,
                "capital": 0.3,
                "social_influence": 0.2,
                "residual": 0.1,
            },
        },
    )


@pytest.fixture
def sensitivity_result():
    return ExperimentResult(
        name="sensitivity",
        params={},
        metrics={
            "sensitivities": {
                "si_gamma": {
                    "levels": {
                        "low": {"value": 0.25, "quality_success_corr": 0.55, "gini": 0.45},
                        "base": {"value": 0.5, "quality_success_corr": 0.5, "gini": 0.47},
                        "high": {"value": 0.75, "quality_success_corr": 0.45, "gini": 0.5},
                    },
                    "corr_sensitivity": -0.2,
                    "gini_sensitivity": 0.1,
                },
                "mee_lambda": {
                    "levels": {
                        "low": {"value": 0.15, "quality_success_corr": 0.45, "gini": 0.5},
                        "base": {"value": 0.3, "quality_success_corr": 0.5, "gini": 0.47},
                        "high": {"value": 0.45, "quality_success_corr": 0.55, "gini": 0.43},
                    },
                    "corr_sensitivity": 0.33,
                    "gini_sensitivity": -0.23,
                },
            },
        },
    )


class TestQualitySuccessScatter:
    def test_returns_axes(self):
        q = np.random.randn(50)
        s = np.random.randn(50)
        ax = plot_quality_success_scatter(q, s)
        assert isinstance(ax, plt.Axes)
        plt.close("all")

    def test_with_canonical(self):
        q = np.random.randn(50)
        s = np.random.randn(50)
        c = np.random.choice([True, False], size=50)
        ax = plot_quality_success_scatter(q, s, canonical=c)
        assert isinstance(ax, plt.Axes)
        plt.close("all")


class TestConditionComparison:
    def test_returns_figure(self, salganik_result):
        fig = plot_condition_comparison(salganik_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestCounterfactualSummary:
    def test_returns_figure(self, counterfactual_result):
        fig = plot_counterfactual_summary(counterfactual_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestVarianceDecomposition:
    def test_returns_figure(self, variance_result):
        fig = plot_variance_decomposition(variance_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestSensitivityTornado:
    def test_returns_figure(self, sensitivity_result):
        fig = plot_sensitivity_tornado(sensitivity_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")

    def test_gini_metric(self, sensitivity_result):
        fig = plot_sensitivity_tornado(sensitivity_result, metric="gini")
        assert isinstance(fig, plt.Figure)
        plt.close("all")


class TestCanonicalByDecile:
    def test_returns_figure(self, counterfactual_result):
        fig = plot_canonical_by_decile(counterfactual_result)
        assert isinstance(fig, plt.Figure)
        plt.close("all")
