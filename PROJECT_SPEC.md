# Cultural Market Simulation Project Specification

## Overview

Create a GitHub repository called `cultural-market-simulation` that implements a computational framework for analyzing survivorship bias in aesthetic canons. The project investigates how capital, exposure, and social influence determine which cultural products achieve canonical status.

This is a research simulation project based on a formal paper. The core hypothesis is the **capital-exposure-canonization loop**: initial capital advantages translate into exposure advantages, which through mere-exposure effects and social influence dynamics become encoded as quality judgments, creating self-reinforcing cycles largely independent of intrinsic quality except at distributional extremes.

---

## Repository Structure

```
cultural-market-simulation/
├── README.md                    # Project overview and usage
├── LICENSE                      # MIT License
├── .gitignore                   # Python, LaTeX, Jupyter ignores
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── paper/
│   ├── paper.tex                # LaTeX source (provided below)
│   ├── paper.pdf                # Compiled paper
│   └── paper.md                 # Markdown version
├── src/
│   └── cultural_market/
│       ├── __init__.py          # Package init
│       ├── agents.py            # Producer, Consumer, Gatekeeper classes
│       ├── market.py            # CulturalMarket simulation class
│       ├── mechanisms.py        # MEE, social influence, cumulative advantage functions
│       ├── distributions.py     # Quality and capital distribution generators
│       ├── calibration.py       # Parameter calibration against Salganik data
│       ├── experiments.py       # Experiment runners (replication, counterfactual, etc.)
│       ├── metrics.py           # Gini coefficient, correlations, Jaccard similarity
│       └── visualization.py     # Plotting functions
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_mechanisms.py
│   └── test_market.py
├── data/
│   ├── .gitkeep
│   └── salganik_targets.json    # Calibration targets from literature
├── results/
│   └── .gitkeep
├── notebooks/
│   ├── 01_exploration.ipynb     # Initial exploration
│   ├── 02_calibration.ipynb     # Calibration workflow
│   └── 03_experiments.ipynb     # Running experiments
└── scripts/
    ├── run_calibration.py       # CLI for calibration
    └── run_experiments.py       # CLI for experiments
```

---

## Core Equations

### 1. Exposure Allocation

The exposure product `i` receives at time `t`:

```
E_it = f(K_it) + g(S_{i,t-1}) + ε_it
```

Where:
- `f(K)` = capital-driven exposure (concave: diminishing returns)
- `g(S)` = success-driven exposure (convex: increasing returns) 
- `ε` = random variation

Implementation:
```python
def exposure_from_capital(K, alpha=0.5):
    """Concave function: f(K) = K^alpha where alpha < 1"""
    return np.power(K, alpha)

def exposure_from_success(S, S_median, beta=1.5):
    """Convex function: g(S) = (S/S_median)^beta where beta > 1"""
    return np.power(S / S_median, beta)
```

### 2. Perceived Quality Formation

Consumer `j`'s perceived quality of product `i` at time `t`:

```
P_jit = Q_i + α_j · MEE(E_jit) + β_j · SI(S̄_it) + η_jit
```

Where:
- `Q_i` = intrinsic quality
- `α_j` = consumer's susceptibility to mere exposure
- `β_j` = consumer's susceptibility to social influence
- `MEE()` = mere exposure effect function
- `SI()` = social influence function
- `η` = idiosyncratic taste noise

### 3. Mere Exposure Effect (MEE)

Inverted-U relationship (liking increases then decreases with exposure):

```
MEE(E) = λ · E · exp(-E/τ)
```

Where:
- `λ` = maximum effect strength (default: 0.3)
- `τ` = saturation parameter (default: 15)

```python
def mere_exposure_effect(E, lambda_=0.3, tau=15):
    """Inverted-U mere exposure effect."""
    return lambda_ * E * np.exp(-E / tau)
```

### 4. Social Influence Function

Log transformation to prevent runaway effects:

```
SI(S̄) = γ · log(1 + S̄/S̄_median)
```

Where:
- `γ` = social influence strength (default: 0.5)
- `S̄` = average observed success
- `S̄_median` = median success (normalization)

```python
def social_influence(S_mean, S_median, gamma=0.5):
    """Social influence based on observed popularity."""
    return gamma * np.log1p(S_mean / S_median)
```

### 5. Canonical Status

Binary threshold:

```
C_i = 1 if Σ_t S_it > θ else 0
```

Where `θ` is the canonization threshold (default: 95th percentile of cumulative success).

---

## Default Parameters

```python
DEFAULT_PARAMS = {
    # Population sizes
    "n_producers": 1000,
    "n_consumers": 10000,
    
    # Quality distribution (truncated normal)
    "quality_mean": 0.0,
    "quality_std": 1.0,
    
    # Capital distribution (log-normal)
    "capital_mean": 0.0,
    "capital_std": 1.5,  # High inequality
    
    # Mere exposure effect
    "mee_lambda": 0.3,   # Effect strength
    "mee_tau": 15,       # Saturation point
    
    # Social influence
    "si_gamma": 0.5,     # Influence strength
    
    # Exposure functions
    "capital_alpha": 0.5,   # Concavity of capital->exposure
    "success_beta": 1.5,    # Convexity of success->exposure
    
    # Consumer heterogeneity
    "alpha_mean": 1.0,   # Mean MEE susceptibility
    "alpha_std": 0.2,
    "beta_mean": 1.0,    # Mean SI susceptibility  
    "beta_std": 0.2,
    
    # Simulation
    "t_active": 50,      # Active market periods
    "t_canon": 100,      # Total periods
    "canon_threshold_percentile": 95,
    
    # Noise
    "exposure_noise_std": 0.1,
    "taste_noise_std": 0.5,
}
```

---

## Calibration Targets

From Salganik, Dodds & Watts (2006) MusicLab experiments:

```json
{
    "source": "Salganik et al. 2006",
    "targets": {
        "quality_success_correlation_independent": {
            "value": 0.65,
            "range": [0.60, 0.70],
            "description": "Correlation between intrinsic quality and success without social influence"
        },
        "quality_success_correlation_social": {
            "value": 0.40,
            "range": [0.30, 0.50],
            "description": "Correlation between intrinsic quality and success with social influence"
        },
        "gini_ratio_social_independent": {
            "value": 1.30,
            "range": [1.25, 1.50],
            "description": "Ratio of Gini coefficient (social condition / independent condition)"
        },
        "rank_variance_middle_quality": {
            "value": 200,
            "range": [150, 250],
            "description": "Variance in rank across worlds for middle-quality songs"
        }
    },
    "notes": {
        "best_rarely_did_poorly": "Top decile quality rarely ranked below median",
        "worst_rarely_did_well": "Bottom decile quality rarely ranked above median",
        "middle_unpredictable": "Middle 80% could achieve any rank"
    }
}
```

---

## Agent Classes

### Producer

```python
@dataclass
class Producer:
    id: int
    quality: float          # Q_i ~ TruncatedNormal(μ_Q, σ_Q)
    initial_capital: float  # K_i0 ~ LogNormal(μ_K, σ_K)
    capital: float          # Current capital (updated based on success)
    cumulative_success: float = 0.0
    canonical: bool = False
    
    def update_capital(self, success: float, reinvestment_rate: float = 0.1):
        """Update capital based on success."""
        self.capital += reinvestment_rate * success
        self.cumulative_success += success
```

### Consumer

```python
@dataclass
class Consumer:
    id: int
    mee_susceptibility: float   # α_j ~ Normal(μ_α, σ_α)
    si_susceptibility: float    # β_j ~ Normal(μ_β, σ_β)
    exposure_history: Dict[int, int] = field(default_factory=dict)  # producer_id -> count
    
    def perceive_quality(self, producer: Producer, social_signal: float, 
                         mee_func, si_func, noise_std: float) -> float:
        """Form perceived quality judgment."""
        exposure = self.exposure_history.get(producer.id, 0)
        mee = self.mee_susceptibility * mee_func(exposure)
        si = self.si_susceptibility * si_func(social_signal)
        noise = np.random.normal(0, noise_std)
        return producer.quality + mee + si + noise
```

### Gatekeeper (optional, for extended model)

```python
@dataclass
class Gatekeeper:
    id: int
    influence: float  # How much exposure they control
    bias: Dict[str, float] = field(default_factory=dict)  # e.g., {"capital": 0.3}
    
    def allocate_exposure(self, producers: List[Producer], 
                          total_exposure: float) -> Dict[int, float]:
        """Decide how to allocate exposure across producers."""
        pass
```

---

## Market Simulation Class

```python
class CulturalMarket:
    def __init__(self, params: dict = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.producers: List[Producer] = []
        self.consumers: List[Consumer] = []
        self.history: List[dict] = []  # Per-period snapshots
        self.t = 0
        
    def initialize(self, seed: int = None):
        """Initialize producers and consumers with random draws."""
        pass
    
    def step(self):
        """Run one simulation period."""
        # 1. Allocate exposure based on capital and prior success
        # 2. Consumers encounter products (update exposure histories)
        # 3. Consumers form quality perceptions
        # 4. Aggregate into success scores
        # 5. Update producer capital
        # 6. Record history
        pass
    
    def run(self, periods: int = None):
        """Run simulation for specified periods."""
        periods = periods or self.params["t_canon"]
        for _ in range(periods):
            self.step()
            self.t += 1
        self.determine_canonical_status()
        
    def determine_canonical_status(self):
        """Mark producers above threshold as canonical."""
        threshold = np.percentile(
            [p.cumulative_success for p in self.producers],
            self.params["canon_threshold_percentile"]
        )
        for p in self.producers:
            p.canonical = p.cumulative_success > threshold
    
    def get_canonical_set(self) -> Set[int]:
        """Return IDs of canonical producers."""
        return {p.id for p in self.producers if p.canonical}
    
    def compute_metrics(self) -> dict:
        """Compute summary metrics."""
        pass
```

---

## Experiments to Implement

### Experiment 1: Salganik Replication

```python
def experiment_salganik_replication(n_runs: int = 100) -> dict:
    """
    Replicate Salganik conditions.
    
    Run with:
    - gamma=0 (independent condition)
    - gamma=calibrated (social influence condition)
    
    Compare quality-success correlations, Gini coefficients, rank variance.
    """
    pass
```

### Experiment 2: Counterfactual Canon Formation

```python
def experiment_counterfactual(n_runs: int = 1000, seed_quality: int = 42) -> dict:
    """
    Fix quality distribution, vary only capital allocation seeds.
    
    Measure:
    - Canonical probability for each producer across runs
    - Jaccard similarity between canonical sets
    - Quality vs canonical probability variance
    """
    pass
```

### Experiment 3: Variance Decomposition

```python
def experiment_variance_decomposition(n_runs: int = 100) -> dict:
    """
    Partition canonical variance into:
    - Quality differences
    - Initial capital differences  
    - Social influence amplification
    - Residual randomness
    
    Run ablations: full model, gamma=0, homogeneous K, both.
    """
    pass
```

### Experiment 4: Historical Scenario (18th-century Vienna)

```python
def experiment_historical_scenario(n_runs: int = 500) -> dict:
    """
    Stylized 18th-century musical culture.
    
    Parameters:
    - n_producers: 300 (estimated active composers)
    - capital_std: 2.0 (higher inequality - few major courts)
    - Constrained exposure (no recording, limited publishing)
    """
    pass
```

### Experiment 5: Sensitivity Analysis

```python
def experiment_sensitivity(base_params: dict, 
                           vary_by: float = 0.5) -> dict:
    """
    Vary each parameter ±50% from calibrated values.
    Report sensitivity of key outcomes.
    """
    pass
```

---

## Metrics Module

```python
def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient of inequality."""
    sorted_values = np.sort(values)
    n = len(values)
    cumsum = np.cumsum(sorted_values)
    return (2 * np.sum((np.arange(1, n+1) * sorted_values)) / (n * cumsum[-1])) - (n + 1) / n

def quality_success_correlation(producers: List[Producer]) -> float:
    """Pearson correlation between quality and cumulative success."""
    qualities = [p.quality for p in producers]
    successes = [p.cumulative_success for p in producers]
    return np.corrcoef(qualities, successes)[0, 1]

def jaccard_similarity(set1: Set, set2: Set) -> float:
    """Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def canonical_probability_by_quality_decile(
    runs: List[CulturalMarket]
) -> Dict[int, float]:
    """
    For each quality decile, compute probability of achieving canonical status.
    """
    pass

def counterfactual_distance(runs: List[CulturalMarket]) -> float:
    """
    Average Jaccard distance between canonical sets across runs.
    Higher = more path-dependent.
    """
    canonical_sets = [m.get_canonical_set() for m in runs]
    distances = []
    for i, s1 in enumerate(canonical_sets):
        for s2 in canonical_sets[i+1:]:
            distances.append(1 - jaccard_similarity(s1, s2))
    return np.mean(distances)
```

---

## Requirements

```
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.2.0
tqdm>=4.65.0
jupyter>=1.0.0
pytest>=7.3.0
```

---

## README Content

The README should include:

1. **Project title and one-line description**
2. **Motivation**: Why study survivorship bias in cultural markets?
3. **Core hypothesis**: Capital-exposure-canonization loop
4. **Key findings** (placeholder until experiments run)
5. **Installation instructions**
6. **Quick start example**
7. **Project structure explanation**
8. **Link to paper PDF**
9. **Citation format**
10. **License**

---

## .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
.env
.venv
env/
venv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# LaTeX
*.aux
*.bbl
*.blg
*.log
*.out
*.toc
*.synctex.gz
*.fdb_latexmk
*.fls

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
results/*.pkl
results/*.json
!results/.gitkeep
data/*.csv
!data/.gitkeep
!data/salganik_targets.json
```

---

## Instructions for Claude Code

1. Create the GitHub repository `cultural-market-simulation`
2. Initialize with the structure above
3. Implement all Python modules with full docstrings and type hints
4. Create the paper/ directory and note that LaTeX files will be added manually
5. Set up pytest configuration
6. Create initial notebooks with markdown cells explaining each step
7. Write comprehensive README
8. Push to GitHub
9. Report back the repository URL

The paper.tex, paper.pdf, and paper.md files will be provided separately - create placeholder files or note their expected location.
