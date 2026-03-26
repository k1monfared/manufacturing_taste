"""Cultural Market Simulation Package.

A computational framework for analyzing survivorship bias in aesthetic canons.

Core classes:
    - Producer: Cultural producers with quality and capital
    - Consumer: Consumers who perceive quality with biases
    - CulturalMarket: Main simulation orchestrator

Key functions:
    - run calibration against Salganik data
    - execute experiments (replication, counterfactual, etc.)
    - compute metrics (Gini, correlations, Jaccard)
"""

from .agents import Producer, Consumer, Gatekeeper
from .market import CulturalMarket, DEFAULT_PARAMS
from .mechanisms import (
    mere_exposure_effect,
    social_influence,
    exposure_from_capital,
    exposure_from_success,
)
from .distributions import (
    generate_quality_distribution,
    generate_capital_distribution,
)
from .metrics import (
    gini_coefficient,
    quality_success_correlation,
    jaccard_similarity,
    counterfactual_distance,
)

__version__ = "0.1.0"
__all__ = [
    "Producer",
    "Consumer",
    "Gatekeeper",
    "CulturalMarket",
    "DEFAULT_PARAMS",
    "mere_exposure_effect",
    "social_influence",
    "exposure_from_capital",
    "exposure_from_success",
    "generate_quality_distribution",
    "generate_capital_distribution",
    "gini_coefficient",
    "quality_success_correlation",
    "jaccard_similarity",
    "counterfactual_distance",
]
