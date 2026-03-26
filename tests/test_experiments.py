"""Tests for the experiments module."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.experiments import (
    ExperimentResult,
    experiment_counterfactual,
    experiment_historical_scenario,
    experiment_salganik_replication,
    experiment_sensitivity,
    experiment_variance_decomposition,
)

FAST_PARAMS = {
    "n_producers": 20,
    "n_consumers": 50,
    "t_canon": 10,
    "t_active": 5,
}


class TestSalganikReplication:
    def test_returns_experiment_result(self):
        result = experiment_salganik_replication(
            n_runs=2, params=FAST_PARAMS, seed=42, progress=False
        )
        assert isinstance(result, ExperimentResult)
        assert result.name == "salganik_replication"

    def test_has_expected_metric_keys(self):
        result = experiment_salganik_replication(
            n_runs=2, params=FAST_PARAMS, seed=42, progress=False
        )
        assert "independent" in result.metrics
        assert "social" in result.metrics
        assert "comparison" in result.metrics
        assert "quality_success_corr_mean" in result.metrics["independent"]
        assert "gini_mean" in result.metrics["independent"]

    def test_correlation_values_valid(self):
        result = experiment_salganik_replication(
            n_runs=2, params=FAST_PARAMS, seed=42, progress=False
        )
        assert -1 <= result.metrics["independent"]["quality_success_corr_mean"] <= 1
        assert -1 <= result.metrics["social"]["quality_success_corr_mean"] <= 1


class TestCounterfactual:
    def test_returns_experiment_result(self):
        result = experiment_counterfactual(
            n_runs=2, quality_seed=42, params=FAST_PARAMS, progress=False
        )
        assert isinstance(result, ExperimentResult)
        assert result.name == "counterfactual"

    def test_has_expected_metrics(self):
        result = experiment_counterfactual(
            n_runs=2, quality_seed=42, params=FAST_PARAMS, progress=False
        )
        assert "counterfactual_distance" in result.metrics
        assert "canonical_probability_variance" in result.metrics
        assert "canonical_by_decile" in result.metrics

    def test_counterfactual_distance_valid(self):
        result = experiment_counterfactual(
            n_runs=3, quality_seed=42, params=FAST_PARAMS, progress=False
        )
        assert 0 <= result.metrics["counterfactual_distance"] <= 1


class TestVarianceDecomposition:
    def test_returns_experiment_result(self):
        result = experiment_variance_decomposition(
            n_runs=1, params=FAST_PARAMS, seed=42, progress=False
        )
        assert isinstance(result, ExperimentResult)

    def test_has_decomposition(self):
        result = experiment_variance_decomposition(
            n_runs=1, params=FAST_PARAMS, seed=42, progress=False
        )
        assert "decomposition" in result.metrics
        decomp = result.metrics["decomposition"]
        assert "quality" in decomp
        assert "capital" in decomp
        assert "social_influence" in decomp

    def test_has_conditions(self):
        result = experiment_variance_decomposition(
            n_runs=1, params=FAST_PARAMS, seed=42, progress=False
        )
        conds = result.metrics["conditions"]
        assert "full" in conds
        assert "no_social" in conds
        assert "homogeneous_capital" in conds
        assert "both_ablations" in conds


class TestHistoricalScenario:
    def test_returns_experiment_result(self):
        result = experiment_historical_scenario(
            n_runs=2, seed=42, progress=False
        )
        assert isinstance(result, ExperimentResult)
        assert result.name == "historical_scenario"

    def test_has_expected_metrics(self):
        result = experiment_historical_scenario(
            n_runs=2, seed=42, progress=False
        )
        assert "quality_success_corr_mean" in result.metrics
        assert "capital_success_corr_mean" in result.metrics
        assert "counterfactual_distance" in result.metrics


class TestSensitivity:
    def test_returns_experiment_result(self):
        # Use smaller variation to avoid beta going below 1
        result = experiment_sensitivity(
            base_params=FAST_PARAMS, vary_by=0.2, n_runs=1, seed=42, progress=False
        )
        assert isinstance(result, ExperimentResult)
        assert result.name == "sensitivity"

    def test_has_sensitivities(self):
        result = experiment_sensitivity(
            base_params=FAST_PARAMS, vary_by=0.2, n_runs=1, seed=42, progress=False
        )
        assert "sensitivities" in result.metrics
        sens = result.metrics["sensitivities"]
        assert len(sens) > 0
        for param_data in sens.values():
            assert "levels" in param_data
            assert "corr_sensitivity" in param_data
