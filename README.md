# Baseball Statistics Analysis

A modern baseball analytics project that prioritizes meaningful statistics over traditional metrics. Built with Python and the `pybaseball` library, this project demonstrates why stats like wRC+, FIP, and K-BB% tell better stories than RBI, wins, and saves.

## Features

- **Modern Analytics Focus**: Emphasizes advanced metrics that better reflect player performance
- **Player Classification**: Automatically categorizes hitters and pitchers into meaningful archetypes
- **Educational Examples**: Demonstrates why certain stats mislead and others reveal truth
- **Interactive Analysis**: Jupyter notebooks for hands-on exploration
- **Statcast Integration**: Quality contact analysis using advanced tracking data

## Quick Start

### Prerequisites
- Python 3.13+
- `uv` package manager

### Installation

```bash
# Clone and navigate to project
git clone <repository-url>
cd baseball_stats

# Install dependencies
uv sync

# Install development tools
uv add --dev black flake8 mypy pytest pre-commit
```

### Basic Usage

```python
from src.baseball_stats import *

# Get modern hitting stats for 2024
hitting_data = get_modern_hitting_stats(2024)

# Demonstrate misleading vs. meaningful stats
demonstrate_stat_lies()

# Analyze hitter types
analyze_hitter_types()

# Get comprehensive learning examples
quick_learning_examples()
```

### Interactive Analysis

```bash
# Start Jupyter Lab
uv run jupyter lab

# Open the analytics overview notebook
# Navigate to notebooks/analytics_overview.ipynb
```

## Key Statistical Philosophy

### Prioritized Stats
- **wRC+**: Weighted runs created, park/era adjusted
- **FIP**: Fielding Independent Pitching
- **K-BB%**: Strikeout minus walk rate
- **OPS**: On-base plus slugging
- **ISO**: Isolated power (SLG - AVG)

### Contextual Indicators
- **BABIP**: Batting average on balls in play (luck indicator)
- **LOB%**: Left on base percentage (strand rate)
- **Positional Adjustments**: Context-aware performance evaluation

### Avoided Traditional Stats
- **RBI**: Heavily context-dependent
- **Wins/Saves**: Team-dependent metrics
- **Raw Batting Average**: Ignores walks and power

## Player Classification System

### Hitters
- **Three True Outcomes**: High K%, high BB%, high HR%
- **Contact Artist**: Low K%, high contact rate
- **Elite Hitter**: High wRC+, well-rounded approach
- **Average Hitter**: League-average performance

### Pitchers
- **Flamethrower**: High velocity, high strikeout rate
- **Power Pitcher**: Above-average strikeouts and velocity
- **Crafty Veteran**: Below-average velocity, above-average results
- **Effective Pitcher**: Solid overall performance

## Development

### Code Quality
```bash
# Format code
uv run black .

# Lint code
uv run flake8 .

# Type check
uv run mypy src/

# Run tests
uv run pytest tests/
```

### Pre-commit Hooks
```bash
# Install hooks
uv run pre-commit install

# Run hooks manually
uv run pre-commit run --all-files
```

## Data Sources

- **Baseball Reference**: Historical statistics
- **FanGraphs**: Advanced metrics
- **Statcast**: Tracking data and quality of contact metrics
- **pybaseball**: Python interface to baseball data
