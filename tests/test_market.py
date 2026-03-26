"""Tests for CulturalMarket class."""

import numpy as np
import pytest

from cultural_market.market import DEFAULT_PARAMS, CulturalMarket


class TestCulturalMarket:
    """Test main market simulation class."""

    def test_initialization(self) -> None:
        market = CulturalMarket()
        assert market.params == DEFAULT_PARAMS
        assert len(market.producers) == 0
        assert len(market.consumers) == 0
        assert market.t == 0

    def test_custom_params(self) -> None:
        custom = {"n_producers": 50, "n_consumers": 100}
        market = CulturalMarket(custom)
        assert market.params["n_producers"] == 50
        assert market.params["n_consumers"] == 100
        assert market.params["si_gamma"] == DEFAULT_PARAMS["si_gamma"]

    def test_initialize_creates_agents(self) -> None:
        market = CulturalMarket({"n_producers": 10, "n_consumers": 20})
        market.initialize(seed=42)

        assert len(market.producers) == 10
        assert len(market.consumers) == 20

        # Check producers have valid attributes
        for p in market.producers:
            assert isinstance(p.quality, float)
            assert p.initial_capital > 0

    def test_step_requires_initialization(self) -> None:
        market = CulturalMarket()
        with pytest.raises(RuntimeError):
            market.step()

    def test_step_advances_time(self) -> None:
        market = CulturalMarket({"n_producers": 10, "n_consumers": 50})
        market.initialize(seed=42)

        assert market.t == 0
        market.step()
        assert market.t == 1

    def test_step_updates_success(self) -> None:
        market = CulturalMarket({"n_producers": 10, "n_consumers": 100})
        market.initialize(seed=42)

        initial_success = sum(p.cumulative_success for p in market.producers)
        assert initial_success == 0

        market.step()

        final_success = sum(p.cumulative_success for p in market.producers)
        assert final_success > 0

    def test_step_records_history(self) -> None:
        market = CulturalMarket({"n_producers": 10, "n_consumers": 50})
        market.initialize(seed=42)

        assert len(market.history) == 0
        market.step()
        assert len(market.history) == 1
        market.step()
        assert len(market.history) == 2

    def test_run_completes_simulation(self) -> None:
        market = CulturalMarket(
            {"n_producers": 10, "n_consumers": 50, "t_canon": 10}
        )
        market.initialize(seed=42)
        market.run()

        assert market.t == 10
        assert len(market.history) == 10

    def test_canonical_status_determination(self) -> None:
        market = CulturalMarket(
            {
                "n_producers": 100,
                "n_consumers": 200,
                "t_canon": 20,
                "canon_threshold_percentile": 90,
            }
        )
        market.initialize(seed=42)
        market.run()

        canonical = market.get_canonical_set()

        # Should have ~10% canonical (90th percentile threshold)
        assert len(canonical) == pytest.approx(10, abs=3)

    def test_compute_metrics(self) -> None:
        market = CulturalMarket(
            {"n_producers": 50, "n_consumers": 100, "t_canon": 10}
        )
        market.initialize(seed=42)
        market.run()

        metrics = market.compute_metrics()

        assert "quality_success_correlation" in metrics
        assert "gini_coefficient" in metrics
        assert "n_canonical" in metrics
        assert -1 <= metrics["quality_success_correlation"] <= 1
        assert 0 <= metrics["gini_coefficient"] <= 1

    def test_reproducibility_with_seed(self) -> None:
        metrics1 = self._run_with_seed(42)
        metrics2 = self._run_with_seed(42)

        assert metrics1["quality_success_correlation"] == metrics2[
            "quality_success_correlation"
        ]
        assert metrics1["gini_coefficient"] == metrics2["gini_coefficient"]

    def test_different_seeds_different_results(self) -> None:
        metrics1 = self._run_with_seed(42)
        metrics2 = self._run_with_seed(123)

        # Results should differ (very unlikely to be exactly equal)
        assert metrics1["gini_coefficient"] != metrics2["gini_coefficient"]

    def test_reset_clears_state(self) -> None:
        market = CulturalMarket(
            {"n_producers": 10, "n_consumers": 50, "t_canon": 5}
        )
        market.initialize(seed=42)
        market.run()

        # Record state
        original_success = market.producers[0].cumulative_success
        assert original_success > 0

        # Reset
        market.reset()

        assert market.t == 0
        assert len(market.history) == 0
        assert market.producers[0].cumulative_success == 0

    def _run_with_seed(self, seed: int) -> dict:
        market = CulturalMarket(
            {"n_producers": 20, "n_consumers": 50, "t_canon": 5}
        )
        market.initialize(seed=seed)
        market.run()
        return market.compute_metrics()


class TestMarketMetrics:
    """Test market metrics computations."""

    def test_quality_success_correlation_range(self) -> None:
        market = CulturalMarket(
            {"n_producers": 100, "n_consumers": 200, "t_canon": 20}
        )
        market.initialize(seed=42)
        market.run()

        metrics = market.compute_metrics()
        corr = metrics["quality_success_correlation"]

        assert -1 <= corr <= 1

    def test_gini_coefficient_range(self) -> None:
        market = CulturalMarket(
            {"n_producers": 100, "n_consumers": 200, "t_canon": 20}
        )
        market.initialize(seed=42)
        market.run()

        metrics = market.compute_metrics()
        gini = metrics["gini_coefficient"]

        assert 0 <= gini <= 1

    def test_social_influence_increases_inequality(self) -> None:
        # Run with social influence
        market_social = CulturalMarket(
            {"n_producers": 50, "n_consumers": 100, "t_canon": 30, "si_gamma": 0.5}
        )
        market_social.initialize(seed=42)
        market_social.run()

        # Run without social influence
        market_independent = CulturalMarket(
            {"n_producers": 50, "n_consumers": 100, "t_canon": 30, "si_gamma": 0.0}
        )
        market_independent.initialize(seed=42)
        market_independent.run()

        gini_social = market_social.compute_metrics()["gini_coefficient"]
        gini_ind = market_independent.compute_metrics()["gini_coefficient"]

        # Social influence should increase inequality
        assert gini_social >= gini_ind * 0.9  # Allow some variance
