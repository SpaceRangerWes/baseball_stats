"""
Baseball Stats Learning Tool - Based on Modern Analytics
Focus on stats that actually matter, skip the lies

This module provides a clean separation between:
- Data extraction (data.py)
- Statistical analysis (analysis.py)
- Batter analysis (batter_analysis.py)
"""

# Import key functions from each module for easy access
from .data import (
    get_raw_hitting_stats,
    get_raw_pitching_stats,
    get_qualified_hitters,
    get_qualified_pitchers,
    get_player_lookup,
    get_player_statcast_data,
    get_team_stats,
    get_standings,
)

# Import the unified data loader
from .data_loader import BaseballDataLoader

from .analysis import (
    get_modern_hitting_stats,
    get_modern_pitching_stats,
    classify_hitter_type,
    classify_pitcher_type,
    analyze_hitter_types,
    get_hitter_examples_by_type,
    find_stat_examples,
    get_elite_hitters,
    calculate_quality_contact_metrics,
    analyze_player_contact_quality,
    get_player_performance_rating,
    find_player_by_name,
    create_correlation_heatmap,
    identify_outliers,
    display_luck_analysis,
    analyze_contact_quality,
    create_distribution_comparison,
    get_pitcher_splits_data,
    calculate_pitcher_rolling_averages,
    compare_pitcher_vs_population,
    track_pitcher_metrics_longitudinally,
    identify_pitcher_performance_patterns,
)

# Import batter analysis functions
from .batter_analysis import (
    create_distribution_plots,
    analyze_single_player,
    get_player_performance_summary,
    create_player_comparison_plot,
    get_league_leaders,
    analyze_metric_correlation,
    identify_outliers as identify_stat_outliers,
)

# Export all public functions
__all__ = [
    "BaseballDataLoader",
    "get_raw_hitting_stats",
    "get_raw_pitching_stats",
    "get_qualified_hitters",
    "get_qualified_pitchers",
    "get_player_lookup",
    "get_player_statcast_data",
    "get_team_stats",
    "get_standings",
    "get_modern_hitting_stats",
    "get_modern_pitching_stats",
    "classify_hitter_type",
    "classify_pitcher_type",
    "analyze_hitter_types",
    "get_hitter_examples_by_type",
    "find_stat_examples",
    "get_elite_hitters",
    "calculate_quality_contact_metrics",
    "analyze_player_contact_quality",
    "get_player_performance_rating",
    "find_player_by_name",
    "create_correlation_heatmap",
    "identify_outliers",
    "display_luck_analysis",
    "analyze_contact_quality",
    "create_distribution_comparison",
    "get_pitcher_splits_data",
    "calculate_pitcher_rolling_averages",
    "compare_pitcher_vs_population",
    "track_pitcher_metrics_longitudinally",
    "identify_pitcher_performance_patterns",
    # Batter analysis functions
    "create_distribution_plots",
    "analyze_single_player",
    "get_player_performance_summary",
    "create_player_comparison_plot",
    "get_league_leaders",
    "analyze_metric_correlation",
    "identify_stat_outliers",
]
