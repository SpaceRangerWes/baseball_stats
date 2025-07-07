# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
This is a baseball statistics analysis project built with Python, focused on modern analytics that prioritize meaningful stats over traditional metrics. The project uses the `pybaseball` library to fetch MLB data and emphasizes stats like wRC+, FIP, K-BB%, and OPS over traditional stats like RBI, wins, and saves.

## Development Commands

### Package Management
This project uses `uv` for dependency management:
```bash
# Install dependencies
uv sync

# Add new dependencies
uv add package-name

# Add development dependencies  
uv add --dev package-name

# Run Python with project dependencies
uv run python main.py
```

### Development Tools
```bash
# Code formatting
uv run black .

# Linting
uv run flake8 .

# Type checking
uv run mypy src/

# Run tests
uv run pytest tests/

# Run single test
uv run pytest tests/test_specific.py::test_function_name

# Install pre-commit hooks
uv run pre-commit install
```

### Running Python Commands
**IMPORTANT**: All Python commands must be run with `uv run` to use the project's dependencies:
```bash
# Run main script
uv run python main.py

# Run any Python script
uv run python script_name.py

# Interactive Python shell with project dependencies
uv run python

# Run specific module
uv run python -m baseball_stats.analysis
```

### Jupyter Notebooks
```bash
# Start Jupyter server
uv run jupyter lab

# Run specific notebooks
uv run jupyter execute notebooks/analytics_overview.ipynb
uv run jupyter execute notebooks/batter_analysis.ipynb
```

## Code Architecture

### Core Module Structure
- `src/baseball_stats/__init__.py` - Main module exports with clean function imports
- `src/baseball_stats/data.py` - Data extraction layer with domain-specific classes and caching
- `src/baseball_stats/analysis.py` - Statistical analysis and player classification functions
- `src/baseball_stats/data_loader.py` - Unified data loader providing single interface to all data
- `src/baseball_stats/batter_analysis.py` - Specialized batting analysis functions and visualizations
- `main.py` - Entry point script 
- `notebooks/` - Jupyter notebooks for interactive analysis
  - `analytics_overview.ipynb` - General analytics overview and exploration
  - `batter_analysis.ipynb` - Comprehensive batting statistics analysis with interactive player selection

### Key Components

#### Data Layer (`data.py`)
- **CacheManager** - Intelligent caching with different expiration for current vs historical data
- **Domain Classes**: HittingData, PitchingData, TeamData, PlayerData, StatcastData
- **Raw Data Functions**: `get_qualified_hitters()`, `get_qualified_pitchers()`, `get_player_lookup()`
- **Statcast Integration**: Advanced metrics, exit velocity, launch angle, barrel rates

#### Analysis Layer (`analysis.py`)
- **Modern Hitting Stats**: Focus on wRC+, OBP, ISO, K-BB%, advanced Statcast metrics
- **Modern Pitching Stats**: Prioritize FIP, WHIP, K-BB%, GB/FB ratios over traditional ERA
- **Player Classification**: 
  - Hitters: Three True Outcomes, Contact Artist, Elite Hitter, Average Hitter
  - Pitchers: Flamethrower, Power Pitcher, Crafty Veteran, Effective Pitcher, Average Pitcher
- **Advanced Analytics**: Correlation analysis, outlier detection, luck indicators
- **Contact Quality**: Barrel rates, exit velocity analysis, expected stats vs actual

#### Data Loader (`data_loader.py`)
- **BaseballDataLoader** - Unified interface to load all data types
- **Intelligent Defaults**: Qualified thresholds (100+ PA, 20+ IP), configurable parameters
- **Comprehensive Coverage**: Hitting, pitching, fielding, baserunning, team data
- **Smart Caching**: Automatic data caching with appropriate expiration times
- **Error Handling**: Graceful handling of data source failures with fallback options

#### Batter Analysis (`batter_analysis.py`)
- **Specialized Visualizations**: Distribution plots, box plots, player comparisons
- **Performance Analysis**: Player profiling, league comparisons, outlier detection
- **Interactive Functions**: Player selection, metric correlation analysis
- **Modern Analytics Focus**: Emphasis on wRC+, OPS, OBP, and advanced metrics

### Statistical Philosophy
The codebase implements a specific analytical philosophy:
- **Prioritized Stats**: wRC+, OBP, FIP, K-BB%, OPS, WHIP, ISO, Barrel%, xwOBA
- **Contextual Stats**: BABIP (luck indicator), LOB% (strand rate), positional adjustments
- **Avoided Stats**: RBI (context-dependent), Wins/Saves (team-dependent), raw batting average
- **Advanced Metrics**: Exit velocity, launch angle, expected stats, sprint speed, baserunning value

### Player Classification System
- **Hitters**: Three True Outcomes, Contact Artist, Elite Hitter, Average Hitter
- **Pitchers**: Flamethrower, Power Pitcher, Crafty Veteran, Effective Pitcher, Average Pitcher
- **Classification Logic**: Based on K%, BB%, ISO, contact rates, and Statcast metrics

## Data Sources
- Uses `pybaseball` library with intelligent caching system
- **Data Providers**: Baseball Reference, FanGraphs, Statcast
- **Caching Strategy**: Current year data expires in 1 hour, historical data persists
- **Cache Location**: `.baseball_stats_cache/` directory
- **Offline Capability**: Cached data allows some offline analysis

## Python Environment
- **Python Version**: 3.13+ (specified in pyproject.toml)
- **Package Manager**: `uv` for dependency management
- **Core Dependencies**: pandas, numpy, matplotlib, seaborn, plotly, pybaseball
- **Development Tools**: black, flake8, mypy, pytest, pre-commit

## Common Workflows

### Adding New Analysis Functions
1. **Data Functions**: Add to `src/baseball_stats/data.py` in appropriate domain class
2. **Analysis Functions**: Add to `src/baseball_stats/analysis.py` with proper classification logic
3. **Export Functions**: Update `src/baseball_stats/__init__.py` imports and `__all__` list
4. **Follow Patterns**: Focus on meaningful stats, include educational value
5. **Use BaseballDataLoader**: Leverage unified interface for data access

### Working with Notebooks
- **Import Pattern**: `from src.baseball_stats import *` or `from baseball_stats import BaseballDataLoader`
- **Data Loading**: Use `BaseballDataLoader` for consistent data access
- **Analysis Pattern**: Load data once, perform multiple analyses
- **Caching**: Automatic caching speeds up repeated queries
- **Interactive Features**: 
  - `batter_analysis.ipynb` includes interactive player selection (enter name or "random")
  - Global logging level set to INFO to suppress DEBUG messages
  - Box plot visualizations for distribution analysis
  - Comprehensive player profiles with modern analytics

### Working with the Data Loader
```python
# Initialize with defaults
loader = BaseballDataLoader(year=2024)
loader.load_all()

# Access specific data
hitting_data = loader.get_hitting_data()
pitching_data = loader.get_pitching_data()
```

### Testing Data Analysis
- **Multiple Years**: Test with different years (2020-2025)
- **Qualified Thresholds**: Verify 100+ PA for hitters, 20+ IP for pitchers
- **Edge Cases**: Test player classification boundary conditions
- **Data Validation**: Check for missing values and data quality issues

## Available Notebooks

### `notebooks/analytics_overview.ipynb`
- General analytics overview and exploration
- Broad analysis of team and player performance
- Introduction to modern baseball analytics concepts

### `notebooks/batter_analysis.ipynb`
- **Comprehensive batting analysis** with interactive features
- **Key Features**:
  - Distribution analysis using box plots for AVG, OBP, OPS, wRC+
  - Interactive player selection (enter name or type "random")
  - Complete player profiles with traditional and modern stats
  - Performance context vs league averages
  - Global INFO logging to reduce debug noise
- **Usage**: Run cells sequentially, interact with player selection prompt
- **Focus**: Modern analytics emphasizing wRC+, OBP, ISO, and advanced metrics

## Module Exports

The `src/baseball_stats/__init__.py` provides comprehensive exports including:
- **Core Data Functions**: All data loading and extraction functions
- **Analysis Functions**: Player classification, performance analysis, correlation analysis
- **Batter Analysis Functions**: Specialized batting visualizations and analysis
- **Unified Interface**: `BaseballDataLoader` as the primary data access point