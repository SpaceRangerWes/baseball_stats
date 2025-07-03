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

### Jupyter Notebooks
```bash
# Start Jupyter server
uv run jupyter lab

# Run notebook from command line
uv run jupyter execute notebooks/learning_notebook.ipynb
```

## Code Architecture

### Core Module Structure
- `src/baseball_stats/__init__.py` - Main module containing all baseball analysis functions
- `main.py` - Entry point script 
- `notebooks/` - Jupyter notebooks for interactive analysis
- `tests/` - Test files

### Key Components

#### Statistical Analysis Functions
The main module is organized around educational baseball statistics:

1. **Data Fetching Functions**
   - `get_modern_hitting_stats(year)` - Fetches batting data with focus on meaningful stats
   - `get_modern_pitching_stats(year)` - Fetches pitching data prioritizing FIP over ERA

2. **Analysis Functions**
   - `demonstrate_stat_lies()` - Shows examples of misleading vs. truthful stats
   - `analyze_hitter_types()` - Classifies hitters into archetypes (Three True Outcomes, Contact Artist, etc.)
   - `quality_contact_analysis(player_name)` - Statcast analysis for individual players
   - `get_player_story(player_name)` - Generates narrative analysis of player performance

3. **Learning Functions**
   - `quick_learning_examples()` - Runs comprehensive examples demonstrating key concepts
   - `show_context_matters()` - Demonstrates positional adjustments and context

### Statistical Philosophy
The codebase implements a specific analytical philosophy:
- **Prioritized Stats**: wRC+, OBP, FIP, K-BB%, OPS, WHIP, ISO
- **Contextual Stats**: BABIP (luck indicator), LOB% (strand rate), positional adjustments
- **Avoided Stats**: RBI (context-dependent), Wins/Saves (team-dependent), raw batting average

### Player Classification System
- **Hitters**: Three True Outcomes, Contact Artist, Elite Hitter, Average Hitter
- **Pitchers**: Flamethrower, Power Pitcher, Crafty Veteran, Effective Pitcher, Average Pitcher

## Data Sources
- Uses `pybaseball` library with caching enabled
- Fetches from Baseball Reference, FanGraphs, and Statcast
- Requires internet connection for fresh data
- Cached data stored locally for performance

## Python Environment
- Python 3.13 specified in `.python-version`
- Virtual environment managed by `uv`
- Key dependencies: pandas, numpy, matplotlib, seaborn, plotly, pybaseball

## Common Workflows

### Adding New Analysis Functions
1. Add function to `src/baseball_stats/__init__.py`
2. Follow existing pattern of focusing on meaningful stats
3. Include educational comments explaining why certain stats matter
4. Add player classification logic when appropriate

### Working with Notebooks
- Import from main module: `from src.baseball_stats import *`
- Use relative imports may fail - import directly from installed package
- Notebooks are designed for interactive learning and experimentation

### Testing Data Analysis
- Test with different years of data
- Verify qualified player thresholds (100+ PA for hitters, 20+ IP for pitchers)
- Check for edge cases in player classification logic