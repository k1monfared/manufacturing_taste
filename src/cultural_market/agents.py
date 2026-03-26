"""Agent classes for cultural market simulation.

Three agent types:
- Producer: Creates cultural products, has quality and capital
- Consumer: Perceives quality, influenced by exposure and social signals
- Gatekeeper: Controls exposure allocation (optional extension)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .mechanisms import mere_exposure_effect, social_influence


@dataclass
class Producer:
    """A cultural producer (composer, artist, musician).

    Attributes:
        id: Unique identifier
        quality: Intrinsic quality Q_i of their product
        initial_capital: Starting capital K_i0
        capital: Current capital (updated based on success)
        cumulative_success: Total success accumulated over time
        canonical: Whether producer has achieved canonical status
    """

    id: int
    quality: float
    initial_capital: float
    capital: float = field(init=False)
    cumulative_success: float = 0.0
    canonical: bool = False

    def __post_init__(self) -> None:
        self.capital = self.initial_capital

    def update_capital(
        self,
        success: float,
        reinvestment_rate: float = 0.1,
    ) -> None:
        """Update capital based on period success.

        Args:
            success: Success in current period
            reinvestment_rate: Fraction of success reinvested as capital
        """
        self.capital += reinvestment_rate * success
        self.cumulative_success += success

    def reset(self) -> None:
        """Reset producer to initial state for re-running simulation."""
        self.capital = self.initial_capital
        self.cumulative_success = 0.0
        self.canonical = False


@dataclass
class Consumer:
    """A consumer who perceives and evaluates cultural products.

    Attributes:
        id: Unique identifier
        mee_susceptibility: alpha_j - sensitivity to mere exposure effect
        si_susceptibility: beta_j - sensitivity to social influence
        exposure_history: Dict mapping producer_id to exposure count
    """

    id: int
    mee_susceptibility: float
    si_susceptibility: float
    exposure_history: Dict[int, int] = field(default_factory=dict)

    def record_exposure(self, producer_id: int, count: int = 1) -> None:
        """Record exposure to a producer's product.

        Args:
            producer_id: ID of producer whose product was encountered
            count: Number of exposures (default 1)
        """
        self.exposure_history[producer_id] = (
            self.exposure_history.get(producer_id, 0) + count
        )

    def get_exposure_count(self, producer_id: int) -> int:
        """Get total exposure count for a producer."""
        return self.exposure_history.get(producer_id, 0)

    def perceive_quality(
        self,
        producer: Producer,
        social_signal: float,
        S_median: float,
        mee_lambda: float = 0.3,
        mee_tau: float = 15.0,
        si_gamma: float = 0.5,
        noise_std: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> float:
        """Form perceived quality judgment.

        P_jit = Q_i + alpha_j * MEE(E_jit) + beta_j * SI(S_bar_it) + eta_jit

        Args:
            producer: Producer whose product is being evaluated
            social_signal: Average observed success S_bar for this producer
            S_median: Median success for normalization
            mee_lambda: MEE strength parameter
            mee_tau: MEE saturation parameter
            si_gamma: Social influence strength
            noise_std: Standard deviation of taste noise
            rng: Random number generator

        Returns:
            Perceived quality value
        """
        if rng is None:
            rng = np.random.default_rng()

        exposure = self.get_exposure_count(producer.id)

        # Mere exposure contribution
        mee = self.mee_susceptibility * float(
            mere_exposure_effect(exposure, lambda_=mee_lambda, tau=mee_tau)
        )

        # Social influence contribution
        si = self.si_susceptibility * float(
            social_influence(social_signal, S_median, gamma=si_gamma)
        )

        # Idiosyncratic taste noise
        noise = rng.normal(0, noise_std)

        return producer.quality + mee + si + noise

    def reset(self) -> None:
        """Reset consumer for re-running simulation."""
        self.exposure_history.clear()


@dataclass
class Gatekeeper:
    """Institutional gatekeeper controlling exposure allocation.

    Optional extension for modeling critics, curators, programmers.

    Attributes:
        id: Unique identifier
        influence: Proportion of total exposure this gatekeeper controls
        capital_bias: Weight given to producer capital in allocation
        quality_perception_noise: Noise in gatekeeper's quality perception
    """

    id: int
    influence: float
    capital_bias: float = 0.3
    quality_perception_noise: float = 0.5

    def allocate_exposure(
        self,
        producers: List[Producer],
        total_exposure: float,
        rng: np.random.Generator | None = None,
    ) -> Dict[int, float]:
        """Allocate exposure across producers.

        Allocation based on perceived quality (with bias toward capital).

        Args:
            producers: List of producers to allocate across
            total_exposure: Total exposure to distribute
            rng: Random number generator

        Returns:
            Dict mapping producer_id to exposure allocation
        """
        if rng is None:
            rng = np.random.default_rng()

        # Compute perceived quality with capital bias
        scores = []
        for p in producers:
            noise = rng.normal(0, self.quality_perception_noise)
            perceived = p.quality + self.capital_bias * np.log1p(p.capital) + noise
            scores.append(max(perceived, 0))

        scores_arr = np.array(scores)

        # Softmax allocation
        if np.sum(scores_arr) == 0:
            # Uniform if all scores are zero
            allocations = np.ones(len(producers)) / len(producers)
        else:
            allocations = scores_arr / np.sum(scores_arr)

        return {
            p.id: total_exposure * alloc for p, alloc in zip(producers, allocations)
        }
