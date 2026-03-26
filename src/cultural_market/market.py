"""Main cultural market simulation class.

Orchestrates producers, consumers, exposure allocation, and
canonical status determination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

import numpy as np

from .agents import Consumer, Producer
from .distributions import (
    generate_capital_distribution,
    generate_consumer_susceptibilities,
    generate_quality_distribution,
)
from .mechanisms import compute_total_exposure
from .metrics import gini_coefficient, quality_success_correlation


# Default parameters from spec
DEFAULT_PARAMS = {
    # Population sizes
    "n_producers": 1000,
    "n_consumers": 10000,
    # Quality distribution (truncated normal)
    "quality_mean": 0.0,
    "quality_std": 1.0,
    # Capital distribution (log-normal)
    "capital_mean": 0.0,
    "capital_std": 1.5,
    # Mere exposure effect
    "mee_lambda": 0.3,
    "mee_tau": 15,
    # Social influence
    "si_gamma": 0.5,
    # Exposure functions
    "capital_alpha": 0.5,
    "success_beta": 1.5,
    # Consumer heterogeneity
    "alpha_mean": 1.0,
    "alpha_std": 0.2,
    "beta_mean": 1.0,
    "beta_std": 0.2,
    # Simulation
    "t_active": 50,
    "t_canon": 100,
    "canon_threshold_percentile": 95,
    # Noise
    "exposure_noise_std": 0.1,
    "taste_noise_std": 0.5,
    # Capital dynamics
    "reinvestment_rate": 0.1,
}


@dataclass
class PeriodSnapshot:
    """Record of market state at a single time period."""

    t: int
    exposures: np.ndarray
    successes: np.ndarray
    cumulative_successes: np.ndarray
    gini: float
    quality_success_corr: float


class CulturalMarket:
    """Main simulation class for cultural market dynamics.

    Simulates:
    - Exposure allocation based on capital and prior success
    - Consumer encounters and exposure history
    - Perceived quality formation with MEE and SI
    - Success aggregation and capital dynamics
    - Canonical status determination
    """

    def __init__(self, params: dict | None = None) -> None:
        """Initialize market with parameters.

        Args:
            params: Dict of parameters (merged with defaults)
        """
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.producers: List[Producer] = []
        self.consumers: List[Consumer] = []
        self.history: List[PeriodSnapshot] = []
        self.t: int = 0
        self._rng: np.random.Generator | None = None
        self._initialized: bool = False

    def initialize(self, seed: int | None = None) -> None:
        """Initialize producers and consumers.

        Args:
            seed: Random seed for reproducibility
        """
        self._rng = np.random.default_rng(seed)

        # Generate producer attributes
        n_prod = self.params["n_producers"]
        qualities = generate_quality_distribution(
            n_prod,
            mean=self.params["quality_mean"],
            std=self.params["quality_std"],
            seed=self._rng.integers(0, 2**31),
        )
        capitals = generate_capital_distribution(
            n_prod,
            mean=self.params["capital_mean"],
            std=self.params["capital_std"],
            seed=self._rng.integers(0, 2**31),
        )

        self.producers = [
            Producer(id=i, quality=float(q), initial_capital=float(k))
            for i, (q, k) in enumerate(zip(qualities, capitals))
        ]

        # Generate consumer attributes
        n_cons = self.params["n_consumers"]
        alpha_vals, beta_vals = generate_consumer_susceptibilities(
            n_cons,
            mee_mean=self.params["alpha_mean"],
            mee_std=self.params["alpha_std"],
            si_mean=self.params["beta_mean"],
            si_std=self.params["beta_std"],
            seed=self._rng.integers(0, 2**31),
        )

        self.consumers = [
            Consumer(id=j, mee_susceptibility=float(a), si_susceptibility=float(b))
            for j, (a, b) in enumerate(zip(alpha_vals, beta_vals))
        ]

        self.t = 0
        self.history.clear()
        self._initialized = True

    def _compute_S_median(self) -> float:
        """Compute median success for normalization."""
        successes = [p.cumulative_success for p in self.producers]
        median = float(np.median(successes))
        # Avoid division by zero in early periods
        return max(median, 1.0)

    def step(self) -> None:
        """Run one simulation period."""
        if not self._initialized:
            raise RuntimeError("Must call initialize() before step()")

        assert self._rng is not None

        S_median = self._compute_S_median()
        n_prod = len(self.producers)

        # 1. Compute exposure for each producer
        capitals = np.array([p.capital for p in self.producers])
        prior_success = np.array([p.cumulative_success for p in self.producers])

        exposures = compute_total_exposure(
            capital=capitals,
            prior_success=prior_success,
            S_median=S_median,
            alpha=self.params["capital_alpha"],
            beta=self.params["success_beta"],
            noise_std=self.params["exposure_noise_std"],
            rng=self._rng,
        )

        # Normalize to total exposure budget (proportional to consumer count)
        total_exposure_budget = self.params["n_consumers"] * 0.1
        exposures = exposures / exposures.sum() * total_exposure_budget

        # 2. Allocate consumer encounters based on exposure
        encounter_probs = exposures / exposures.sum()

        period_success = np.zeros(n_prod)

        for consumer in self.consumers:
            # Each consumer samples a fixed number of products
            n_encounters = min(10, n_prod)
            encountered_ids = self._rng.choice(
                n_prod,
                size=n_encounters,
                replace=False,
                p=encounter_probs,
            )

            for prod_id in encountered_ids:
                consumer.record_exposure(prod_id)

                # Consumer evaluates and contributes to success
                producer = self.producers[prod_id]
                social_signal = producer.cumulative_success / max(self.t, 1)

                perceived = consumer.perceive_quality(
                    producer=producer,
                    social_signal=social_signal,
                    S_median=S_median,
                    mee_lambda=self.params["mee_lambda"],
                    mee_tau=self.params["mee_tau"],
                    si_gamma=self.params["si_gamma"],
                    noise_std=self.params["taste_noise_std"],
                    rng=self._rng,
                )

                # Success is probabilistic based on perceived quality
                # Using sigmoid to map to [0, 1]
                success_prob = 1 / (1 + np.exp(-perceived))
                if self._rng.random() < success_prob:
                    period_success[prod_id] += 1

        # 3. Update producer capital
        for i, producer in enumerate(self.producers):
            producer.update_capital(
                period_success[i],
                reinvestment_rate=self.params["reinvestment_rate"],
            )

        # 4. Record snapshot
        cumulative = np.array([p.cumulative_success for p in self.producers])
        qualities = np.array([p.quality for p in self.producers])

        snapshot = PeriodSnapshot(
            t=self.t,
            exposures=exposures.copy(),
            successes=period_success.copy(),
            cumulative_successes=cumulative.copy(),
            gini=gini_coefficient(cumulative),
            quality_success_corr=quality_success_correlation(qualities, cumulative),
        )
        self.history.append(snapshot)

        self.t += 1

    def run(self, periods: int | None = None) -> None:
        """Run simulation for specified periods.

        Args:
            periods: Number of periods (default: t_canon from params)
        """
        periods = periods or self.params["t_canon"]
        for _ in range(periods):
            self.step()
        self.determine_canonical_status()

    def determine_canonical_status(self) -> None:
        """Mark producers above threshold as canonical."""
        successes = [p.cumulative_success for p in self.producers]
        threshold = float(
            np.percentile(successes, self.params["canon_threshold_percentile"])
        )
        for p in self.producers:
            p.canonical = p.cumulative_success > threshold

    def get_canonical_set(self) -> Set[int]:
        """Return IDs of canonical producers."""
        return {p.id for p in self.producers if p.canonical}

    def compute_metrics(self) -> dict:
        """Compute summary metrics for the simulation run."""
        qualities = np.array([p.quality for p in self.producers])
        successes = np.array([p.cumulative_success for p in self.producers])
        capitals = np.array([p.initial_capital for p in self.producers])

        canonical_mask = np.array([p.canonical for p in self.producers])

        return {
            "quality_success_correlation": quality_success_correlation(
                qualities, successes
            ),
            "capital_success_correlation": quality_success_correlation(
                capitals, successes
            ),
            "quality_canonical_correlation": quality_success_correlation(
                qualities,
                canonical_mask.astype(float),
            ),
            "gini_coefficient": gini_coefficient(successes),
            "n_canonical": int(np.sum(canonical_mask)),
            "mean_canonical_quality": float(np.mean(qualities[canonical_mask]))
            if np.any(canonical_mask)
            else 0.0,
        }

    def reset(self) -> None:
        """Reset market for re-running with same agents."""
        for p in self.producers:
            p.reset()
        for c in self.consumers:
            c.reset()
        self.t = 0
        self.history.clear()
