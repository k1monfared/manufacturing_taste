"""Tests for agent classes."""

import numpy as np
import pytest

from cultural_market.agents import Consumer, Gatekeeper, Producer


class TestProducer:
    """Test Producer class."""

    def test_initialization(self) -> None:
        p = Producer(id=0, quality=0.5, initial_capital=100.0)
        assert p.id == 0
        assert p.quality == 0.5
        assert p.initial_capital == 100.0
        assert p.capital == 100.0
        assert p.cumulative_success == 0.0
        assert p.canonical is False

    def test_update_capital(self) -> None:
        p = Producer(id=0, quality=0.5, initial_capital=100.0)
        p.update_capital(success=50.0, reinvestment_rate=0.1)

        assert p.capital == 105.0  # 100 + 0.1*50
        assert p.cumulative_success == 50.0

    def test_multiple_updates_accumulate(self) -> None:
        p = Producer(id=0, quality=0.5, initial_capital=100.0)
        p.update_capital(success=10.0)
        p.update_capital(success=20.0)
        p.update_capital(success=30.0)

        assert p.cumulative_success == 60.0

    def test_reset(self) -> None:
        p = Producer(id=0, quality=0.5, initial_capital=100.0)
        p.update_capital(50.0)
        p.canonical = True

        p.reset()

        assert p.capital == 100.0
        assert p.cumulative_success == 0.0
        assert p.canonical is False


class TestConsumer:
    """Test Consumer class."""

    def test_initialization(self) -> None:
        c = Consumer(id=0, mee_susceptibility=1.0, si_susceptibility=0.8)
        assert c.id == 0
        assert c.mee_susceptibility == 1.0
        assert c.si_susceptibility == 0.8
        assert len(c.exposure_history) == 0

    def test_record_exposure(self) -> None:
        c = Consumer(id=0, mee_susceptibility=1.0, si_susceptibility=0.8)

        c.record_exposure(producer_id=5)
        assert c.get_exposure_count(5) == 1

        c.record_exposure(producer_id=5, count=3)
        assert c.get_exposure_count(5) == 4

        assert c.get_exposure_count(99) == 0  # Never seen

    def test_perceive_quality_base_case(self) -> None:
        c = Consumer(id=0, mee_susceptibility=0.0, si_susceptibility=0.0)
        p = Producer(id=1, quality=0.5, initial_capital=10.0)

        rng = np.random.default_rng(42)
        perceived = c.perceive_quality(
            producer=p,
            social_signal=5.0,
            S_median=1.0,
            noise_std=0.0,  # No noise for deterministic test
            rng=rng,
        )

        # With zero susceptibilities and zero noise, should equal quality
        assert perceived == pytest.approx(p.quality)

    def test_perceive_quality_with_exposure(self) -> None:
        c = Consumer(id=0, mee_susceptibility=1.0, si_susceptibility=1.0)
        p = Producer(id=1, quality=0.5, initial_capital=10.0)

        # Record some exposure
        c.record_exposure(1, count=10)

        rng = np.random.default_rng(42)
        perceived = c.perceive_quality(
            producer=p,
            social_signal=5.0,
            S_median=1.0,
            noise_std=0.0,
            rng=rng,
        )

        # Should be quality + MEE contribution + SI contribution
        assert perceived > p.quality

    def test_reset(self) -> None:
        c = Consumer(id=0, mee_susceptibility=1.0, si_susceptibility=0.8)
        c.record_exposure(1, count=5)
        c.record_exposure(2, count=3)

        c.reset()

        assert len(c.exposure_history) == 0


class TestGatekeeper:
    """Test Gatekeeper class."""

    def test_initialization(self) -> None:
        g = Gatekeeper(id=0, influence=0.5)
        assert g.id == 0
        assert g.influence == 0.5
        assert g.capital_bias == 0.3

    def test_allocate_exposure_distributes_total(self) -> None:
        g = Gatekeeper(id=0, influence=0.5)
        producers = [
            Producer(id=i, quality=float(i), initial_capital=10.0) for i in range(5)
        ]

        total_exposure = 100.0
        allocation = g.allocate_exposure(producers, total_exposure)

        assert sum(allocation.values()) == pytest.approx(total_exposure)
        assert len(allocation) == len(producers)

    def test_allocate_exposure_all_ids_present(self) -> None:
        g = Gatekeeper(id=0, influence=0.5)
        producers = [
            Producer(id=i, quality=0.5, initial_capital=10.0) for i in range(3)
        ]

        allocation = g.allocate_exposure(producers, 100.0)

        assert set(allocation.keys()) == {0, 1, 2}

    def test_higher_quality_gets_more_exposure(self) -> None:
        g = Gatekeeper(id=0, influence=0.5, quality_perception_noise=0.0)
        producers = [
            Producer(id=0, quality=0.0, initial_capital=1.0),
            Producer(id=1, quality=2.0, initial_capital=1.0),
        ]

        rng = np.random.default_rng(42)
        allocation = g.allocate_exposure(producers, 100.0, rng=rng)

        # Higher quality should get more exposure
        assert allocation[1] > allocation[0]
