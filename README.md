# Cultural Market Simulation

**Status**: 🟡 MVP | **Mode**: 🔀 Hybrid | **Updated**: 2026-03-25

A computational framework for analyzing survivorship bias in aesthetic canons.

## Overview

This project investigates how capital, exposure, and social influence determine which cultural products achieve canonical status. The core hypothesis is the **capital-exposure-canonization loop**: initial capital advantages translate into exposure advantages, which through mere-exposure effects and social influence dynamics become encoded as quality judgments, creating self-reinforcing cycles largely independent of intrinsic quality except at distributional extremes.

## Key Findings

Our simulations demonstrate:

1. **Social influence reduces meritocracy**: Quality-success correlation drops from r=0.57 (independent) to r=0.51 (social influence active), p=0.002
2. **Canonical status is highly path-dependent**: Counterfactual distance of 0.88 — different capital allocations produce almost entirely different canons
3. **Capital inequality is the primary mediator**: Equalizing capital increases quality-canonical correlation from 0.32 to 0.46 (a 30% improvement)
4. **Historical scenario shows near-maximal path dependence**: 18th-century Vienna model yields CF distance of 0.97, with only 2.5% canonical overlap across counterfactual runs

## Core Mechanisms

1. **Mere Exposure Effect (MEE)**: Familiarity breeds liking (inverted-U relationship)
2. **Social Influence**: Observed popularity affects perceived quality
3. **Cumulative Advantage**: Early success begets further exposure (Matthew effect)

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from cultural_market import CulturalMarket

# Create and run a simulation
market = CulturalMarket()
market.initialize(seed=42)
market.run()

# Analyze results
metrics = market.compute_metrics()
print(f"Quality-Success Correlation: {metrics['quality_success_correlation']:.3f}")
print(f"Gini Coefficient: {metrics['gini_coefficient']:.3f}")
print(f"Canonical Producers: {metrics['n_canonical']}")
```

## Running Experiments

```bash
# Run batch experiments
python scripts/run_batch.py --n-runs 100 --experiments all

# Combine results and generate analysis
python scripts/combine_batches.py --output results/analysis.json

# Generate figures, tables, and power analysis
python scripts/generate_analysis.py

# Run additional simulations for statistical power
python scripts/run_additional_sims.py
```

## Project Structure

```
manufacturing_taste/
├── src/cultural_market/     # Main package
│   ├── agents.py            # Producer, Consumer, Gatekeeper classes
│   ├── market.py            # CulturalMarket simulation class
│   ├── mechanisms.py        # MEE, social influence functions
│   ├── distributions.py     # Quality and capital distributions
│   ├── calibration.py       # Parameter calibration against Salganik data
│   ├── experiments.py       # 5 experiment runners
│   ├── metrics.py           # Gini, correlations, Jaccard, counterfactual distance
│   ├── visualization.py     # Plotting functions
│   └── power_analysis.py    # Statistical power analysis
├── tests/                   # 90 unit tests
├── data/                    # Calibration targets (Salganik et al. 2006)
├── results/                 # Experiment outputs, figures, tables
├── notebooks/               # Jupyter notebooks (exploration, calibration, experiments)
├── scripts/                 # CLI scripts for running analyses
└── paper/                   # Academic paper (LaTeX + PDF)
```

## Tests

```bash
python -m pytest tests/ -q
```

## Paper

See `paper/paper.pdf` for the full academic paper describing the theoretical framework, simulation architecture, calibration, and results.

## License

MIT License - see LICENSE file for details.
