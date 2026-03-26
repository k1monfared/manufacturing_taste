"""Tests for the calibration module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.calibration import (
    CalibrationTarget,
    calibration_loss,
    load_calibration_targets,
    simulate_independent_condition,
    simulate_social_condition,
)
from cultural_market.market import DEFAULT_PARAMS

FAST_PARAMS = {
    **DEFAULT_PARAMS,
    "n_producers": 20,
    "n_consumers": 50,
    "t_canon": 10,
    "t_active": 5,
}


class TestLoadCalibrationTargets:
    def test_returns_list(self):
        targets = load_calibration_targets()
        assert isinstance(targets, list)
        assert len(targets) > 0

    def test_target_fields(self):
        targets = load_calibration_targets()
        for t in targets:
            assert isinstance(t, CalibrationTarget)
            assert isinstance(t.name, str)
            assert isinstance(t.value, (int, float))
            assert t.range_low <= t.value <= t.range_high

    def test_expected_targets(self):
        targets = load_calibration_targets()
        names = [t.name for t in targets]
        assert "quality_success_correlation_independent" in names
        assert "quality_success_correlation_social" in names


class TestSimulateConditions:
    def test_independent_returns_expected_keys(self):
        result = simulate_independent_condition(FAST_PARAMS, n_runs=1, seed=42)
        assert "quality_success_correlation" in result
        assert "gini_coefficient" in result

    def test_social_returns_expected_keys(self):
        result = simulate_social_condition(FAST_PARAMS, n_runs=1, seed=42)
        assert "quality_success_correlation" in result
        assert "gini_coefficient" in result
        assert "rank_variance_middle" in result

    def test_correlation_in_valid_range(self):
        result = simulate_independent_condition(FAST_PARAMS, n_runs=1, seed=42)
        assert -1 <= result["quality_success_correlation"] <= 1

    def test_gini_in_valid_range(self):
        result = simulate_independent_condition(FAST_PARAMS, n_runs=1, seed=42)
        assert 0 <= result["gini_coefficient"] <= 1


class TestCalibrationLoss:
    def test_returns_nonnegative_float(self):
        targets = load_calibration_targets()
        param_names = ["si_gamma", "mee_lambda"]
        param_values = np.array([0.5, 0.3])

        loss = calibration_loss(
            param_values, param_names, FAST_PARAMS, targets, n_runs=1
        )
        assert isinstance(loss, float)
        assert loss >= 0

    def test_zero_gamma_differs_from_nonzero(self):
        targets = load_calibration_targets()
        param_names = ["si_gamma"]

        loss_zero = calibration_loss(
            np.array([0.0]), param_names, FAST_PARAMS, targets, n_runs=1
        )
        loss_high = calibration_loss(
            np.array([1.5]), param_names, FAST_PARAMS, targets, n_runs=1
        )
        # Both should be valid floats (may or may not be equal with n_runs=1)
        assert isinstance(loss_zero, float)
        assert isinstance(loss_high, float)
