"""
Baseball Stats Learning Tool - Based on Modern Analytics
Focus on stats that actually matter, skip the lies

This module provides a clean separation between:
- Data extraction (data.py)
- Statistical analysis (analysis.py)  
- Educational storytelling (stories.py)
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
    get_standings
)

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
    find_player_by_name
)

from .stories import (
    demonstrate_stat_lies,
    tell_hitter_type_stories,
    show_context_matters,
    get_player_story,
    quality_contact_story,
    quick_learning_examples,
    tell_complete_story,
    tell_modern_bat_tracking_story,
    tell_defensive_story,
    tell_baserunning_story,
    tell_situational_story,
    tell_comprehensive_modern_story,
    demonstrate_modern_analytics_revolution
)

# Maintain backward compatibility with the original API
quality_contact_analysis = quality_contact_story

# Export all public functions
__all__ = [
    # Data functions
    'get_raw_hitting_stats',
    'get_raw_pitching_stats', 
    'get_qualified_hitters',
    'get_qualified_pitchers',
    'get_player_lookup',
    'get_player_statcast_data',
    'get_team_stats',
    'get_standings',
    
    # Analysis functions
    'get_modern_hitting_stats',
    'get_modern_pitching_stats',
    'classify_hitter_type',
    'classify_pitcher_type',
    'analyze_hitter_types',
    'get_hitter_examples_by_type',
    'find_stat_examples',
    'get_elite_hitters',
    'calculate_quality_contact_metrics',
    'analyze_player_contact_quality',
    'get_player_performance_rating',
    'find_player_by_name',
    
    # Story functions
    'demonstrate_stat_lies',
    'tell_hitter_type_stories',
    'show_context_matters',
    'get_player_story',
    'quality_contact_story',
    'quick_learning_examples',
    'tell_complete_story',
    'tell_modern_bat_tracking_story',
    'tell_defensive_story',
    'tell_baserunning_story',
    'tell_situational_story',
    'tell_comprehensive_modern_story',
    'demonstrate_modern_analytics_revolution',
    
    # Backward compatibility
    'quality_contact_analysis'
]
    