"""
Batter analysis functions for comprehensive hitter evaluation.

This module contains functions for analyzing batter performance using modern
sabermetrics and creating visualizations for hitter evaluation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Tuple, Any
from scipy import stats

from .data_loader import BaseballDataLoader

logger = logging.getLogger(__name__)


def create_distribution_plots(
    hitting_data: pd.DataFrame,
    metrics: List[str] = ['AVG', 'OBP', 'OPS', 'wRC+'],
    figsize: Tuple[int, int] = (15, 10),
    bins: int = 30,
    alpha: float = 0.7
) -> plt.Figure:
    """
    Create a 2x2 grid of distribution plots for key hitting metrics.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        metrics: List of metrics to plot (default: ['AVG', 'OBP', 'OPS', 'wRC+'])
        figsize: Figure size as (width, height) tuple
        bins: Number of bins for histograms
        alpha: Transparency level for plots
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValueError: If required metrics are not found in the data
    """
    # Validate that all metrics exist in the data
    missing_metrics = [metric for metric in metrics if metric not in hitting_data.columns]
    if missing_metrics:
        logger.warning(f"Missing metrics in data: {missing_metrics}")
        # Use only available metrics
        metrics = [metric for metric in metrics if metric in hitting_data.columns]
        if not metrics:
            raise ValueError("No valid metrics found in data")
    
    # Create the figure and subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Distribution of Key Hitting Metrics', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Define colors for each metric
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
    
    for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics for 2x2 grid
        ax = axes_flat[i]
        
        # Create histogram
        data_clean = hitting_data[metric].dropna()
        if len(data_clean) > 0:
            ax.hist(data_clean, bins=bins, alpha=alpha, 
                    color=colors[i % len(colors)], edgecolor='black', linewidth=0.5)
            
            # Add mean line
            mean_val = data_clean.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            
            # Add median line
            median_val = data_clean.median()
            ax.axvline(median_val, color='orange', linestyle='--', linewidth=2,
                      label=f'Median: {median_val:.3f}')
            
            # Formatting
            ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel(metric, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'{metric} has no valid data', 
                    transform=ax.transAxes, ha='center', va='center')
            ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
    
    # Hide unused subplots if we have fewer than 4 metrics
    for i in range(len(metrics), 4):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    return fig


def get_player_performance_summary(
    hitting_data: pd.DataFrame,
    player_name: str,
    metrics: List[str] = ['AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'ISO', 'K%', 'BB%']
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance summary for a player with percentiles and z-scores.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        player_name: Name of the player to analyze
        metrics: List of metrics to include in summary
        
    Returns:
        Dictionary containing player stats, percentiles, and z-scores
        
    Raises:
        ValueError: If player not found in data
    """
    # Find the name column
    name_column = None
    for col in hitting_data.columns:
        if 'name' in col.lower() or col in ['Name', 'Player', 'player_name']:
            name_column = col
            break
    
    if not name_column:
        raise ValueError("No name column found in data")
    
    # Find the player
    player_data = hitting_data[hitting_data[name_column].str.contains(player_name, case=False, na=False)]
    if player_data.empty:
        raise ValueError(f"Player '{player_name}' not found in data")
    
    player_stats = player_data.iloc[0]
    summary = {
        'player_name': player_stats[name_column],
        'team': player_stats.get('Team', player_stats.get('Tm', 'Unknown')),
        'games': player_stats.get('G', 0),
        'plate_appearances': player_stats.get('PA', 0),
        'stats': {},
        'percentiles': {},
        'z_scores': {},
        'league_averages': {}
    }
    
    # Calculate stats, percentiles, and z-scores for each metric
    for metric in metrics:
        if metric in hitting_data.columns:
            player_value = player_stats[metric]
            if pd.notna(player_value):
                league_values = hitting_data[metric].dropna()
                
                if len(league_values) > 1:
                    # Calculate percentile (what percentage of players this player is better than)
                    percentile = stats.percentileofscore(league_values, player_value)
                    
                    # Calculate z-score (how many standard deviations from mean)
                    z_score = (player_value - league_values.mean()) / league_values.std()
                    
                    # Store values
                    summary['stats'][metric] = player_value
                    summary['percentiles'][metric] = percentile
                    summary['z_scores'][metric] = z_score
                    summary['league_averages'][metric] = league_values.mean()
    
    return summary


def analyze_single_player(
    hitting_data: pd.DataFrame,
    player_name: str,
    display_metrics: List[str] = ['AVG', 'OBP', 'SLG', 'OPS', 'wRC+', 'ISO', 'K%', 'BB%'],
    show_percentiles: bool = True,
    show_z_scores: bool = True
) -> None:
    """
    Display comprehensive analysis for a single player.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        player_name: Name of the player to analyze
        display_metrics: List of metrics to display
        show_percentiles: Whether to show percentile rankings
        show_z_scores: Whether to show z-scores
        
    Raises:
        ValueError: If player not found in data
    """
    try:
        # Get player performance summary
        summary = get_player_performance_summary(hitting_data, player_name, display_metrics)
        
        # Display header
        print(f"\n{'='*60}")
        print(f"PLAYER ANALYSIS: {summary['player_name']}")
        print(f"{'='*60}")
        print(f"Team: {summary['team']}")
        print(f"Games: {summary['games']}")
        print(f"Plate Appearances: {summary['plate_appearances']}")
        print(f"{'='*60}")
        
        # Display stats with context
        print(f"\n{'Metric':<8} {'Value':<8} {'Lg Avg':<8}", end='')
        if show_percentiles:
            print(f" {'Percentile':<12}", end='')
        if show_z_scores:
            print(f" {'Z-Score':<8}", end='')
        print(f" {'Rating':<12}")
        print("-" * 60)
        
        for metric in display_metrics:
            if metric in summary['stats']:
                value = summary['stats'][metric]
                league_avg = summary['league_averages'][metric]
                percentile = summary['percentiles'][metric]
                z_score = summary['z_scores'][metric]
                
                # Determine rating based on percentile
                if percentile >= 90:
                    rating = "Elite"
                elif percentile >= 75:
                    rating = "Above Average"
                elif percentile >= 50:
                    rating = "Average"
                elif percentile >= 25:
                    rating = "Below Average"
                else:
                    rating = "Poor"
                
                # Format output
                print(f"{metric:<8} {value:<8.3f} {league_avg:<8.3f}", end='')
                if show_percentiles:
                    print(f" {percentile:<12.1f}", end='')
                if show_z_scores:
                    print(f" {z_score:<8.2f}", end='')
                print(f" {rating:<12}")
        
        # Add interpretation
        print(f"\n{'='*60}")
        print("INTERPRETATION:")
        print("- Percentile shows what % of players this player is better than")
        print("- Z-Score shows how many standard deviations from league average")
        print("- Rating categories: Elite (90th+), Above Avg (75th+), Average (50th+)")
        print(f"{'='*60}")
        
    except ValueError as e:
        logger.error(f"Error analyzing player: {e}")
        print(f"Error: {e}")


def create_player_comparison_plot(
    hitting_data: pd.DataFrame,
    player_names: List[str],
    metrics: List[str] = ['OBP', 'SLG', 'wRC+', 'K%', 'BB%'],
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create a radar chart comparing multiple players across key metrics.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        player_names: List of player names to compare
        metrics: List of metrics to include in comparison
        figsize: Figure size as (width, height) tuple
        
    Returns:
        matplotlib Figure object
        
    Raises:
        ValueError: If any player not found in data
    """
    # Find name column
    name_column = None
    for col in hitting_data.columns:
        if 'name' in col.lower() or col in ['Name', 'Player', 'player_name']:
            name_column = col
            break
    
    if not name_column:
        raise ValueError("No name column found in data")
    
    # Validate players exist
    missing_players = []
    for player in player_names:
        if not hitting_data[name_column].str.contains(player, case=False, na=False).any():
            missing_players.append(player)
    
    if missing_players:
        raise ValueError(f"Players not found in data: {missing_players}")
    
    # Get percentiles for each player
    player_percentiles = []
    actual_names = []
    for player in player_names:
        try:
            summary = get_player_performance_summary(hitting_data, player, metrics)
            percentiles = [summary['percentiles'].get(metric, 0) for metric in metrics]
            player_percentiles.append(percentiles)
            actual_names.append(summary['player_name'])
        except ValueError:
            continue
    
    if not player_percentiles:
        raise ValueError("No valid players found for comparison")
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    # Calculate angles for each metric
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each player
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (player, percentiles) in enumerate(zip(actual_names, player_percentiles)):
        # Close the plot
        values = percentiles + percentiles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=player, 
                color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20th', '40th', '60th', '80th', '100th'])
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Player Comparison - Percentile Rankings', size=16, fontweight='bold', pad=20)
    
    return fig


def get_league_leaders(
    hitting_data: pd.DataFrame,
    metric: str,
    top_n: int = 10,
    min_pa: int = 100,
    ascending: bool = False
) -> pd.DataFrame:
    """
    Get league leaders for a specific metric.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        metric: The metric to rank by
        top_n: Number of players to return
        min_pa: Minimum plate appearances to qualify
        ascending: Whether to sort in ascending order (for stats like K% where lower is better)
        
    Returns:
        DataFrame with top players for the metric
        
    Raises:
        ValueError: If metric not found in data
    """
    if metric not in hitting_data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    # Find name column
    name_column = None
    for col in hitting_data.columns:
        if 'name' in col.lower() or col in ['Name', 'Player', 'player_name']:
            name_column = col
            break
    
    if not name_column:
        raise ValueError("No name column found in data")
    
    # Filter by minimum plate appearances
    pa_column = 'PA' if 'PA' in hitting_data.columns else None
    if pa_column:
        qualified_data = hitting_data[hitting_data[pa_column] >= min_pa].copy()
    else:
        qualified_data = hitting_data.copy()
    
    # Sort by metric
    sorted_data = qualified_data.sort_values(metric, ascending=ascending)
    
    # Select columns for display
    display_cols = [name_column, metric]
    if 'Team' in hitting_data.columns:
        display_cols.insert(1, 'Team')
    elif 'Tm' in hitting_data.columns:
        display_cols.insert(1, 'Tm')
    if pa_column:
        display_cols.insert(-1, pa_column)
    
    # Return top N players
    leaders = sorted_data.head(top_n)[display_cols].copy()
    leaders['Rank'] = range(1, len(leaders) + 1)
    
    # Reorder columns to put Rank first
    cols = ['Rank'] + [col for col in leaders.columns if col != 'Rank']
    return leaders[cols]


def analyze_metric_correlation(
    hitting_data: pd.DataFrame,
    metrics: List[str] = ['OBP', 'SLG', 'wRC+', 'ISO', 'K%', 'BB%'],
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a correlation matrix heatmap for hitting metrics.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        metrics: List of metrics to include in correlation analysis
        figsize: Figure size as (width, height) tuple
        
    Returns:
        matplotlib Figure object
    """
    # Filter to only available metrics
    available_metrics = [metric for metric in metrics if metric in hitting_data.columns]
    
    if len(available_metrics) < 2:
        raise ValueError("Need at least 2 metrics for correlation analysis")
    
    # Calculate correlation matrix
    correlation_data = hitting_data[available_metrics].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(correlation_data, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(correlation_data, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
    
    plt.title('Hitting Metrics Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def identify_outliers(
    hitting_data: pd.DataFrame,
    metric: str,
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identify statistical outliers for a given metric.
    
    Args:
        hitting_data: DataFrame containing hitting statistics
        metric: The metric to analyze for outliers
        method: Method to use ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame containing outlier players
        
    Raises:
        ValueError: If metric not found or invalid method
    """
    if metric not in hitting_data.columns:
        raise ValueError(f"Metric '{metric}' not found in data")
    
    if method not in ['iqr', 'zscore']:
        raise ValueError("Method must be 'iqr' or 'zscore'")
    
    # Find name column
    name_column = None
    for col in hitting_data.columns:
        if 'name' in col.lower() or col in ['Name', 'Player', 'player_name']:
            name_column = col
            break
    
    if not name_column:
        raise ValueError("No name column found in data")
    
    # Select columns for analysis
    cols = [name_column, metric]
    if 'Team' in hitting_data.columns:
        cols.insert(1, 'Team')
    elif 'Tm' in hitting_data.columns:
        cols.insert(1, 'Tm')
    
    data = hitting_data[cols].copy()
    data = data.dropna(subset=[metric])
    
    if method == 'iqr':
        # Interquartile Range method
        Q1 = data[metric].quantile(0.25)
        Q3 = data[metric].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers = data[(data[metric] < lower_bound) | (data[metric] > upper_bound)].copy()
        outliers['outlier_type'] = outliers[metric].apply(
            lambda x: 'Low' if x < lower_bound else 'High'
        )
    
    else:  # zscore method
        # Z-score method
        mean_val = data[metric].mean()
        std_val = data[metric].std()
        data['z_score'] = (data[metric] - mean_val) / std_val
        
        outliers = data[abs(data['z_score']) > threshold].copy()
        outliers['outlier_type'] = outliers['z_score'].apply(
            lambda x: 'Low' if x < -threshold else 'High'
        )
    
    return outliers.sort_values(metric, ascending=False)