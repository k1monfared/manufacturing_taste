"""Tests for mechanism functions."""

import numpy as np
import pytest

from cultural_market.mechanisms import (
    exposure_from_capital,
    exposure_from_success,
    mee_peak_exposure,
    mee_peak_value,
    mere_exposure_effect,
    social_influence,
)


class TestMereExposureEffect:
    """Test MEE inverted-U function."""

    def test_zero_exposure_returns_zero(self) -> None:
        assert mere_exposure_effect(0) == 0.0

    def test_peak_at_tau(self) -> None:
        tau = 15.0
        lambda_ = 0.3
        peak_e = mee_peak_exposure(tau)
        assert peak_e == tau

        # Value at peak should be maximum
        values = [
            mere_exposure_effect(e, lambda_, tau) for e in [5, 10, 15, 20, 25, 30]
        ]
        assert values[2] == max(values)  # E=15 is peak

    def test_inverted_u_shape(self) -> None:
        # Should increase then decrease
        tau = 15.0
        e_values = np.array([0, 5, 10, 15, 20, 30, 50])
        mee_values = mere_exposure_effect(e_values)

        # Increasing up to peak
        assert mee_values[1] > mee_values[0]
        assert mee_values[2] > mee_values[1]

        # Decreasing after peak
        assert mee_values[4] < mee_values[3]
        assert mee_values[5] < mee_values[4]

    def test_array_input(self) -> None:
        E = np.array([0, 5, 10, 15, 20])
        result = mere_exposure_effect(E)
        assert result.shape == E.shape

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            mere_exposure_effect(10, lambda_=-1)
        with pytest.raises(ValueError):
            mere_exposure_effect(10, tau=0)

    def test_peak_value_formula(self) -> None:
        lambda_ = 0.3
        tau = 15.0
        expected = lambda_ * tau / np.e
        assert mee_peak_value(lambda_, tau) == pytest.approx(expected)


class TestSocialInfluence:
    """Test SI log function."""

    def test_zero_success_returns_zero(self) -> None:
        assert social_influence(0, S_median=1.0) == 0.0

    def test_increases_with_success(self) -> None:
        si_low = social_influence(1.0, S_median=1.0)
        si_high = social_influence(10.0, S_median=1.0)
        assert si_high > si_low

    def test_diminishing_returns(self) -> None:
        # Log function should show diminishing returns
        S_median = 1.0
        increments = [
            float(social_influence(2, S_median) - social_influence(1, S_median)),
            float(social_influence(10, S_median) - social_influence(9, S_median)),
        ]
        assert increments[0] > increments[1]

    def test_gamma_scaling(self) -> None:
        si_low_gamma = social_influence(5, S_median=1.0, gamma=0.5)
        si_high_gamma = social_influence(5, S_median=1.0, gamma=1.0)
        assert si_high_gamma == pytest.approx(2 * si_low_gamma)

    def test_invalid_params_raise(self) -> None:
        with pytest.raises(ValueError):
            social_influence(5, S_median=1.0, gamma=-1)
        with pytest.raises(ValueError):
            social_influence(5, S_median=0)


class TestExposureFunctions:
    """Test exposure allocation functions."""

    def test_capital_exposure_concave(self) -> None:
        # Diminishing returns
        K = np.array([1, 4, 9, 16])
        E = exposure_from_capital(K, alpha=0.5)

        # With alpha=0.5, E = sqrt(K)
        expected = np.sqrt(K)
        np.testing.assert_array_almost_equal(E, expected)

    def test_success_exposure_convex(self) -> None:
        # Increasing returns
        S = np.array([1, 2, 3, 4])
        S_median = 1.0
        E = exposure_from_success(S, S_median, beta=2.0)

        # With beta=2, E = S^2
        expected = S**2
        np.testing.assert_array_almost_equal(E, expected)

    def test_capital_exposure_invalid_alpha(self) -> None:
        with pytest.raises(ValueError):
            exposure_from_capital(np.array([1, 2]), alpha=1.5)
        with pytest.raises(ValueError):
            exposure_from_capital(np.array([1, 2]), alpha=0)

    def test_success_exposure_invalid_beta(self) -> None:
        with pytest.raises(ValueError):
            exposure_from_success(np.array([1, 2]), S_median=1.0, beta=0.5)

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValueError):
            exposure_from_capital(np.array([-1, 2]), alpha=0.5)
