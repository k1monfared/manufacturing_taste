"""Tests for the power analysis module."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cultural_market.power_analysis import (
    achieved_power_two_sample,
    full_power_analysis,
    power_analysis_from_data,
    required_n_correlation,
    required_n_one_sample,
    required_n_two_sample_t,
)


class TestRequiredNTwoSample:
    def test_large_effect_small_n(self):
        n = required_n_two_sample_t(0.8, alpha=0.05, power=0.80)
        assert 20 <= n <= 30  # ~25 per group for d=0.8

    def test_medium_effect(self):
        n = required_n_two_sample_t(0.5, alpha=0.05, power=0.80)
        assert 50 <= n <= 70  # ~64 per group for d=0.5

    def test_small_effect_large_n(self):
        n = required_n_two_sample_t(0.2, alpha=0.05, power=0.80)
        assert 350 <= n <= 450  # ~394 per group for d=0.2

    def test_invalid_effect_size(self):
        with pytest.raises(ValueError):
            required_n_two_sample_t(0)
        with pytest.raises(ValueError):
            required_n_two_sample_t(-0.5)

    def test_higher_power_needs_more(self):
        n80 = required_n_two_sample_t(0.5, power=0.80)
        n90 = required_n_two_sample_t(0.5, power=0.90)
        assert n90 > n80


class TestRequiredNOneSample:
    def test_returns_positive_int(self):
        n = required_n_one_sample(0.5)
        assert isinstance(n, int)
        assert n > 0

    def test_smaller_than_two_sample(self):
        n_one = required_n_one_sample(0.5)
        n_two = required_n_two_sample_t(0.5)
        assert n_one < n_two


class TestRequiredNCorrelation:
    def test_returns_positive_int(self):
        n = required_n_correlation(0.3)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_corr_needs_less(self):
        n_small = required_n_correlation(0.2)
        n_large = required_n_correlation(0.5)
        assert n_large < n_small


class TestAchievedPower:
    def test_more_samples_more_power(self):
        p50 = achieved_power_two_sample(50, 0.5)
        p200 = achieved_power_two_sample(200, 0.5)
        assert p200 > p50

    def test_power_between_0_and_1(self):
        p = achieved_power_two_sample(100, 0.5)
        assert 0 <= p <= 1

    def test_large_n_near_one(self):
        p = achieved_power_two_sample(10000, 0.5)
        assert p > 0.99


class TestPowerAnalysisFromData:
    def test_with_different_means(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.6, 0.15, 100)
        g2 = rng.normal(0.4, 0.15, 100)

        result = power_analysis_from_data(g1.tolist(), g2.tolist())
        assert result["effect_size_d"] > 1.0
        assert result["achieved_power"] > 0.95
        assert result["p_value"] < 0.01
        assert result["required_n_per_group"] < 100

    def test_with_identical_means(self):
        rng = np.random.default_rng(42)
        g1 = rng.normal(0.5, 0.15, 50)
        g2 = rng.normal(0.5, 0.15, 50)

        result = power_analysis_from_data(g1.tolist(), g2.tolist())
        assert result["effect_size_d"] < 0.5  # Small or zero


class TestFullPowerAnalysis:
    def test_with_mock_data(self):
        mock_analysis = {
            "salganik": {
                "n_independent_runs": 100,
                "n_social_runs": 100,
                "independent": {
                    "quality_success_corr_mean": 0.57,
                    "quality_success_corr_std": 0.14,
                    "gini_mean": 0.46,
                    "gini_std": 0.03,
                },
                "social": {
                    "quality_success_corr_mean": 0.51,
                    "quality_success_corr_std": 0.15,
                    "gini_mean": 0.47,
                    "gini_std": 0.04,
                },
            },
            "counterfactual": {
                "n_runs": 100,
                "counterfactual_distance": 0.88,
            },
            "historical": {
                "n_runs": 100,
                "quality_success_corr_std": 0.08,
            },
        }

        result = full_power_analysis(mock_analysis)
        assert "salganik" in result
        assert "counterfactual" in result
        assert "overall_recommendation" in result
        assert result["counterfactual"]["sufficient"] is True
