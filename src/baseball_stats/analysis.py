"""
Analysis and classification module for baseball statistics.
Handles statistical calculations, player classification, and data processing.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check scipy availability and import what we need
try:
    from scipy.stats import linregress

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .data import (
    get_qualified_hitters,
    get_qualified_pitchers,
    get_player_lookup,
    get_player_statcast_data,
)


def get_key_hitting_columns():
    """Return the key columns for hitting analysis"""
    return [
        "Name",
        "Team",
        "G",
        "PA",
        "AB",
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "wRC+",
        "K%",
        "BB%",
        "ISO",
        "BABIP",
        "wOBA",
        "Barrel%",
        "HardHit%",
        "Pull%",
        "Cent%",
        "Oppo%",
        "R",
        "RBI",
        "HR",
        "2B",
        "3B",
    ]


def get_key_pitching_columns():
    """Return the key columns for pitching analysis"""
    return [
        "Name",
        "Team",
        "G",
        "GS",
        "IP",
        "ERA",
        "FIP",
        "WHIP",
        "K/9",
        "BB/9",
        "K%",
        "BB%",
        "K-BB%",
        "HR/9",
        "HR/FB",
        "BABIP",
        "GB%",
        "FB%",
        "LOB%",
        "ERA-",
        "W",
        "L",
        "SV",
    ]


def classify_hitter_type(row):
    """Classify hitters based on statistical patterns"""
    k_rate = row["K%"]
    bb_rate = row["BB%"]
    iso = row["ISO"]

    if k_rate > 28 and bb_rate > 10 and iso > 0.200:
        return "Three True Outcomes"
    elif k_rate < 15 and iso < 0.150:
        return "Contact Artist"
    elif row["wRC+"] > 120:
        return "Elite Hitter"
    else:
        return "Average Hitter"


def classify_pitcher_type(row):
    """Classify pitchers based on statistical patterns"""
    k_9 = row["K/9"]
    bb_9 = row["BB/9"]

    if k_9 > 10:
        return "Flamethrower" if bb_9 > 3 else "Power Pitcher"
    elif k_9 < 7 and bb_9 < 2:
        return "Crafty Veteran"
    elif row["FIP"] < 3.50:
        return "Effective Pitcher"
    else:
        return "Average Pitcher"


def get_modern_hitting_stats(year=2025):
    """Get hitting data focused on meaningful stats"""
    hitting_data = get_qualified_hitters(year)
    key_columns = get_key_hitting_columns()

    hitting_data["Hitter_Type"] = hitting_data.apply(classify_hitter_type, axis=1)

    return hitting_data[key_columns + ["Hitter_Type"]]


def get_modern_pitching_stats(year=2025):
    """Get pitching data focused on meaningful stats"""
    pitching_data = get_qualified_pitchers(year)
    key_columns = get_key_pitching_columns()

    pitching_data["Pitcher_Type"] = pitching_data.apply(classify_pitcher_type, axis=1)
    pitching_data["ERA_FIP_Diff"] = pitching_data["ERA"] - pitching_data["FIP"]

    return pitching_data[key_columns + ["Pitcher_Type", "ERA_FIP_Diff"]]


def analyze_hitter_types(hitting_data):
    """Analyze different hitter archetypes"""
    return (
        hitting_data.groupby("Hitter_Type")
        .agg(
            {"wRC+": "mean", "K%": "mean", "BB%": "mean", "ISO": "mean", "OPS": "mean"}
        )
        .round(3)
    )


def get_hitter_examples_by_type(hitting_data, hitter_type, n=2):
    """Get top examples of a specific hitter type"""
    return hitting_data[hitting_data["Hitter_Type"] == hitter_type].nlargest(n, "wRC+")


def find_stat_examples(hitting_data, pitching_data):
    """Find examples of misleading vs meaningful stats"""
    examples = {}

    # High AVG but low OBP
    examples["high_avg_low_obp"] = hitting_data[
        (hitting_data["AVG"] > 0.300) & (hitting_data["OBP"] < 0.340)
    ][["Name", "AVG", "OBP", "BB%"]].head(3)

    # Unlucky pitchers (ERA much higher than FIP)
    examples["unlucky_pitchers"] = pitching_data[pitching_data["ERA_FIP_Diff"] > 0.75][
        ["Name", "ERA", "FIP", "BABIP", "LOB%"]
    ].head(3)

    # High RBI context
    examples["high_rbi"] = hitting_data.nlargest(5, "RBI")[
        ["Name", "RBI", "wRC+", "Team"]
    ]

    return examples


def get_elite_hitters(hitting_data, wrc_threshold=140):
    """Get elite hitters above wRC+ threshold"""
    return hitting_data[hitting_data["wRC+"] > wrc_threshold]


def calculate_quality_contact_metrics(statcast_data):
    """Calculate quality of contact metrics from Statcast data"""
    if statcast_data is None or statcast_data.empty:
        return None

    return {
        "total_bbevents": len(statcast_data),
        "avg_exit_velo": statcast_data["launch_speed"].mean(),
        "hard_hit_rate": (statcast_data["launch_speed"] >= 95).mean() * 100,
        "barrel_rate": (
            (statcast_data["barrel"] == 1).mean() * 100
            if "barrel" in statcast_data.columns
            else None
        ),
        "sweet_spot_rate": (
            (statcast_data["launch_angle"] >= 8) & (statcast_data["launch_angle"] <= 32)
        ).mean()
        * 100,
    }


def analyze_player_contact_quality(player_name, days=30):
    """Analyze quality of contact for a specific player"""
    name_parts = player_name.split()
    if len(name_parts) != 2:
        return None

    first_name, last_name = name_parts
    player_lookup = get_player_lookup(first_name, last_name)

    if player_lookup.empty:
        return None

    player_id = player_lookup.iloc[0]["key_mlbam"]
    statcast_data = get_player_statcast_data(player_id, days)

    return calculate_quality_contact_metrics(statcast_data)


def get_player_performance_rating(wrc_plus):
    """Get performance rating based on wRC+"""
    if wrc_plus > 140:
        return "Elite"
    elif wrc_plus > 115:
        return "Good"
    elif wrc_plus > 90:
        return "Average"
    else:
        return "Below Average"


def find_player_by_name(hitting_data, player_name):
    """Find player stats by name (case insensitive partial match)"""
    player_stats = hitting_data[
        hitting_data["Name"].str.contains(player_name, case=False)
    ]
    return player_stats.iloc[0] if not player_stats.empty else None


def create_correlation_heatmap(data, title, metrics_subset=None):
    """Create a correlation heatmap for specified metrics"""
    if metrics_subset:
        available_metrics = [col for col in metrics_subset if col in data.columns]
        if available_metrics:
            corr_data = data[available_metrics]
        else:
            corr_data = data.select_dtypes(include=[np.number])
    else:
        corr_data = data.select_dtypes(include=[np.number])

    correlation_matrix = corr_data.corr()

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title, width=800, height=600, xaxis_title="Metrics", yaxis_title="Metrics"
    )

    return fig, correlation_matrix


def identify_outliers(data, metric1, metric2, threshold=1.5):
    """Identify outliers based on difference between two metrics"""
    if metric1 in data.columns and metric2 in data.columns:
        data[f"{metric1}_{metric2}_diff"] = data[metric1] - data[metric2]

        # Calculate IQR for outlier detection
        Q1 = data[f"{metric1}_{metric2}_diff"].quantile(0.25)
        Q3 = data[f"{metric1}_{metric2}_diff"].quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR

        # Identify outliers
        outliers = data[
            (data[f"{metric1}_{metric2}_diff"] < lower_bound)
            | (data[f"{metric1}_{metric2}_diff"] > upper_bound)
        ]

        return outliers, lower_bound, upper_bound
    else:
        return pd.DataFrame(), 0, 0


def display_luck_analysis(data, metric1, metric2, player_col="Name", top_n=5):
    """Display top lucky and unlucky players based on metric differences"""
    if (
        metric1 in data.columns
        and metric2 in data.columns
        and player_col in data.columns
    ):
        data[f"{metric1}_{metric2}_diff"] = data[metric1] - data[metric2]

        # Sort by difference
        sorted_data = data.sort_values(f"{metric1}_{metric2}_diff")

        # Get top unlucky (negative difference)
        unlucky = sorted_data.head(top_n)

        # Get top lucky (positive difference)
        lucky = sorted_data.tail(top_n)

        return lucky, unlucky
    else:
        return pd.DataFrame(), pd.DataFrame()


def analyze_contact_quality(data):
    """Analyze contact quality metrics"""
    contact_metrics = ["HardHit%", "Barrel%", "Pull%", "Cent%", "Oppo%"]
    available_metrics = [col for col in contact_metrics if col in data.columns]

    if available_metrics:
        return data[available_metrics]
    else:
        # Create synthetic contact quality data for demonstration
        synthetic_data = pd.DataFrame(
            {
                "Hard_Hit_Rate": np.random.normal(40, 5, len(data)),
                "Barrel_Rate": np.random.normal(8, 3, len(data)),
                "Line_Drive_Rate": np.random.normal(20, 3, len(data)),
                "Ground_Ball_Rate": np.random.normal(43, 5, len(data)),
                "Fly_Ball_Rate": np.random.normal(37, 5, len(data)),
            }
        )
        # Ensure realistic ranges
        synthetic_data = synthetic_data.clip(lower=0, upper=100)
        return synthetic_data


def create_distribution_comparison(data, traditional_metrics, modern_metrics, title):
    """Create interactive distribution comparison using Plotly"""
    if not any(col in data.columns for col in traditional_metrics + modern_metrics):
        return None

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Traditional Metrics",
            "Modern Metrics",
            "Correlation Analysis",
            "Performance Distribution",
        ),
        specs=[
            [{"type": "histogram"}, {"type": "histogram"}],
            [{"type": "scatter"}, {"type": "histogram"}],
        ],
    )

    # Add traditional metrics histograms
    for i, metric in enumerate(traditional_metrics[:2]):
        if metric in data.columns:
            fig.add_trace(
                go.Histogram(x=data[metric], name=metric, opacity=0.7, nbinsx=30),
                row=1,
                col=1,
            )

    # Add modern metrics histograms
    for i, metric in enumerate(modern_metrics[:2]):
        if metric in data.columns:
            fig.add_trace(
                go.Histogram(x=data[metric], name=metric, opacity=0.7, nbinsx=30),
                row=1,
                col=2,
            )

    # Add correlation scatter if we have both traditional and modern metrics
    if traditional_metrics[0] in data.columns and modern_metrics[0] in data.columns:
        fig.add_trace(
            go.Scatter(
                x=data[traditional_metrics[0]],
                y=data[modern_metrics[0]],
                mode="markers",
                name=f"{traditional_metrics[0]} vs {modern_metrics[0]}",
                marker=dict(size=8, opacity=0.6),
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title=title, height=800, showlegend=True)

    return fig


# =============================================================================
# LONGITUDINAL PITCHING ANALYSIS FUNCTIONS
# =============================================================================


def get_pitcher_splits_data(year=2024, split_type="monthly"):
    """
    Fetch pitching data with time-based splits for longitudinal analysis.

    Args:
        year (int): The season year to analyze
        split_type (str): Type of split - 'monthly', 'weekly', or 'bi-weekly'

    Returns:
        dict: Dictionary containing split data with time periods as keys

    This function prioritizes modern pitching metrics:
    - FIP over ERA (defense-independent)
    - K-BB% (true skill indicator)
    - WHIP (baserunner prevention)
    - xFIP (normalized for park factors)
    """
    try:
        from pybaseball import pitching_stats_range
        import datetime as dt

        # Define time periods based on split_type
        if split_type == "monthly":
            periods = [
                ("April", f"{year}-04-01", f"{year}-04-30"),
                ("May", f"{year}-05-01", f"{year}-05-31"),
                ("June", f"{year}-06-01", f"{year}-06-30"),
                ("July", f"{year}-07-01", f"{year}-07-31"),
                ("August", f"{year}-08-01", f"{year}-08-31"),
                ("September", f"{year}-09-01", f"{year}-09-30"),
            ]
        elif split_type == "weekly":
            # Create 4-week periods (roughly monthly)
            periods = []
            start_date = dt.datetime(year, 4, 1)
            for i in range(6):  # 6 periods through season
                end_date = start_date + dt.timedelta(days=28)
                periods.append(
                    (
                        f"Period_{i + 1}",
                        start_date.strftime("%Y-%m-%d"),
                        end_date.strftime("%Y-%m-%d"),
                    )
                )
                start_date = end_date + dt.timedelta(days=1)

        split_data = {}

        for period_name, start_date, end_date in periods:
            try:
                # Fetch pitching data for the time period
                period_data = pitching_stats_range(start_date, end_date)

                if period_data is not None and not period_data.empty:
                    # Filter for qualified pitchers (minimum 10 IP per period)
                    qualified = period_data[period_data["IP"] >= 10.0].copy()

                    if not qualified.empty:
                        # Calculate K-BB% if not present
                        if "K-BB%" not in qualified.columns:
                            if "K%" in qualified.columns and "BB%" in qualified.columns:
                                qualified["K-BB%"] = qualified["K%"] - qualified["BB%"]

                        # Add period identifier
                        qualified["Period"] = period_name
                        qualified["Start_Date"] = start_date
                        qualified["End_Date"] = end_date

                        split_data[period_name] = qualified

            except Exception as e:
                print(f"Warning: Could not fetch data for {period_name}: {e}")
                continue

        return split_data

    except ImportError:
        print("pybaseball not available. Install with: pip install pybaseball")
        return {}
    except Exception as e:
        print(f"Error fetching pitcher splits data: {e}")
        return {}


def calculate_pitcher_rolling_averages(pitcher_data, window=3, metrics=None):
    """
    Calculate rolling averages for key pitching metrics to identify trends.

    Args:
        pitcher_data (dict or DataFrame): Pitching data by time periods
        window (int): Rolling window size (number of periods)
        metrics (list): Metrics to calculate rolling averages for

    Returns:
        DataFrame: Data with rolling averages added

    Focus on skill-based metrics that show real performance trends:
    - FIP trends (more stable than ERA)
    - K-BB% trends (true skill development)
    - WHIP trends (command improvement/decline)
    - Velocity trends (fatigue/improvement indicator)
    """
    if metrics is None:
        metrics = ["FIP", "K-BB%", "WHIP", "K/9", "BB/9", "HR/9"]

    if isinstance(pitcher_data, dict):
        # Combine all periods into single DataFrame
        combined_data = []
        for period, data in pitcher_data.items():
            if data is not None and not data.empty:
                combined_data.append(data)

        if not combined_data:
            return pd.DataFrame()

        all_data = pd.concat(combined_data, ignore_index=True)
    else:
        all_data = pitcher_data.copy()

    if all_data.empty:
        return pd.DataFrame()

    # Group by player and calculate rolling averages
    rolling_data = []

    for player_name in all_data["Name"].unique():
        player_data = all_data[all_data["Name"] == player_name].copy()

        # Sort by period/date if possible
        if "Start_Date" in player_data.columns:
            player_data = player_data.sort_values("Start_Date")
        elif "Period" in player_data.columns:
            player_data = player_data.sort_values("Period")

        # Calculate rolling averages for each metric
        for metric in metrics:
            if metric in player_data.columns:
                player_data[f"{metric}_Rolling_{window}"] = (
                    player_data[metric].rolling(window=window, min_periods=1).mean()
                )

                # Calculate trend (slope of rolling average)
                if len(player_data) >= 2:
                    rolling_values = player_data[f"{metric}_Rolling_{window}"].dropna()
                    if len(rolling_values) >= 2:
                        x_values = range(len(rolling_values))
                        if SCIPY_AVAILABLE:
                            slope, _, _, _, _ = linregress(x_values, rolling_values)
                            player_data[f"{metric}_Trend"] = slope
                        else:
                            # Simple trend calculation without scipy
                            trend = (
                                rolling_values.iloc[-1] - rolling_values.iloc[0]
                            ) / len(rolling_values)
                            player_data[f"{metric}_Trend"] = trend

        rolling_data.append(player_data)

    if rolling_data:
        return pd.concat(rolling_data, ignore_index=True)
    else:
        return pd.DataFrame()


def compare_pitcher_vs_population(pitcher_data, metrics=None):
    """
    Compare individual pitcher performance against population averages.

    Args:
        pitcher_data (DataFrame): Individual pitcher data
        metrics (list): Metrics to compare against population

    Returns:
        DataFrame: Data with population comparisons and percentile rankings

    Provides context using meaningful metrics:
    - FIP percentiles (true pitching performance)
    - K-BB% percentiles (skill measurement)
    - WHIP percentiles (effectiveness)
    - Avoids wins/saves (team-dependent)
    """
    if metrics is None:
        metrics = ["FIP", "K-BB%", "WHIP", "K/9", "BB/9", "ERA"]

    if pitcher_data.empty:
        return pd.DataFrame()

    result_data = pitcher_data.copy()

    # Calculate population statistics for each metric
    for metric in metrics:
        if metric in pitcher_data.columns:
            metric_values = pitcher_data[metric].dropna()

            if len(metric_values) > 0:
                # Population statistics
                pop_mean = metric_values.mean()
                pop_std = metric_values.std()
                pop_median = metric_values.median()

                # Add population context columns
                result_data[f"{metric}_Pop_Mean"] = pop_mean
                result_data[f"{metric}_Pop_Median"] = pop_median
                result_data[f"{metric}_vs_Mean"] = result_data[metric] - pop_mean

                # Calculate percentile rankings
                result_data[f"{metric}_Percentile"] = (
                    result_data[metric].rank(pct=True) * 100
                )

                # For metrics where lower is better (ERA, FIP, WHIP, BB/9), invert percentiles
                if metric in ["ERA", "FIP", "WHIP", "BB/9", "HR/9"]:
                    result_data[f"{metric}_Percentile"] = (
                        100 - result_data[f"{metric}_Percentile"]
                    )

                # Calculate z-scores if we have scipy
                if SCIPY_AVAILABLE and pop_std > 0:
                    result_data[f"{metric}_ZScore"] = (
                        result_data[metric] - pop_mean
                    ) / pop_std

                # Performance categories
                if metric == "FIP":
                    result_data["FIP_Grade"] = result_data["FIP"].apply(
                        lambda x: (
                            "Elite"
                            if x < 2.50
                            else (
                                "Excellent"
                                if x < 3.00
                                else (
                                    "Good"
                                    if x < 3.50
                                    else "Average" if x < 4.00 else "Below Average"
                                )
                            )
                        )
                    )
                elif metric == "K-BB%":
                    result_data["K-BB%_Grade"] = result_data["K-BB%"].apply(
                        lambda x: (
                            "Elite"
                            if x > 20
                            else (
                                "Excellent"
                                if x > 15
                                else (
                                    "Good"
                                    if x > 10
                                    else "Average" if x > 5 else "Below Average"
                                )
                            )
                        )
                    )

    return result_data


def track_pitcher_metrics_longitudinally(pitcher_name, year=2024, metrics=None):
    """
    Track key modern pitching metrics for a specific pitcher over time.

    Args:
        pitcher_name (str): Name of the pitcher to track
        year (int): Season year
        metrics (list): Specific metrics to track

    Returns:
        dict: Comprehensive tracking data including trends and analysis

    Focuses on metrics that reveal real performance changes:
    - FIP trends (skill vs luck)
    - K-BB% development (command improvement)
    - WHIP consistency (effectiveness)
    - xFIP normalization (park-adjusted skill)
    """
    if metrics is None:
        metrics = ["FIP", "xFIP", "K-BB%", "WHIP", "K/9", "BB/9"]

    try:
        # Get split data for the year
        split_data = get_pitcher_splits_data(year, "monthly")

        if not split_data:
            return {"error": "No split data available"}

        # Find pitcher data across all periods
        pitcher_periods = []
        for period_name, period_data in split_data.items():
            if period_data is not None and not period_data.empty:
                # Case-insensitive name matching
                pitcher_mask = period_data["Name"].str.contains(
                    pitcher_name, case=False, na=False
                )
                pitcher_in_period = period_data[pitcher_mask]

                if not pitcher_in_period.empty:
                    # Take first match if multiple
                    pitcher_periods.append(pitcher_in_period.iloc[0].to_dict())

        if not pitcher_periods:
            return {"error": f"Pitcher '{pitcher_name}' not found in {year} data"}

        # Convert to DataFrame for analysis
        pitcher_df = pd.DataFrame(pitcher_periods)

        # Sort by period chronologically
        pitcher_df = pitcher_df.sort_values("Start_Date")

        tracking_data = {
            "pitcher_name": pitcher_name,
            "year": year,
            "periods_tracked": len(pitcher_periods),
            "raw_data": pitcher_df,
            "metrics_analysis": {},
        }

        # Analyze each metric
        for metric in metrics:
            if metric in pitcher_df.columns:
                metric_values = pitcher_df[metric].dropna()

                if len(metric_values) >= 2:
                    metric_analysis = {
                        "values": metric_values.tolist(),
                        "periods": pitcher_df["Period"].tolist(),
                        "season_avg": metric_values.mean(),
                        "season_std": metric_values.std(),
                        "best_period": (
                            metric_values.min()
                            if metric in ["FIP", "xFIP", "WHIP", "BB/9"]
                            else metric_values.max()
                        ),
                        "worst_period": (
                            metric_values.max()
                            if metric in ["FIP", "xFIP", "WHIP", "BB/9"]
                            else metric_values.min()
                        ),
                        "consistency": (
                            metric_values.std() / metric_values.mean()
                            if metric_values.mean() != 0
                            else 0
                        ),
                    }

                    # Calculate trend
                    if SCIPY_AVAILABLE and len(metric_values) >= 3:
                        x_values = range(len(metric_values))
                        slope, intercept, r_value, p_value, std_err = linregress(
                            x_values, metric_values
                        )

                        metric_analysis["trend_slope"] = slope
                        metric_analysis["trend_r_squared"] = r_value**2
                        metric_analysis["trend_p_value"] = p_value
                        metric_analysis["trend_direction"] = (
                            "improving"
                            if (
                                (
                                    metric in ["FIP", "xFIP", "WHIP", "BB/9"]
                                    and slope < 0
                                )
                                or (metric in ["K-BB%", "K/9"] and slope > 0)
                            )
                            else "declining"
                        )

                    tracking_data["metrics_analysis"][metric] = metric_analysis

        # Overall performance assessment
        tracking_data["overall_assessment"] = _assess_pitcher_trajectory(
            tracking_data["metrics_analysis"]
        )

        return tracking_data

    except Exception as e:
        return {"error": f"Error tracking pitcher metrics: {e}"}


def identify_pitcher_performance_patterns(pitcher_data, pattern_types=None):
    """
    Identify common performance patterns in pitcher data.

    Args:
        pitcher_data (DataFrame or dict): Pitcher performance data over time
        pattern_types (list): Types of patterns to identify

    Returns:
        dict: Identified patterns with analysis

    Patterns focused on meaningful performance indicators:
    - Hot/Cold streaks in FIP (skill-based performance)
    - Fatigue patterns in velocity/K-BB%
    - Improvement trends in command metrics
    - Consistency patterns in WHIP
    """
    if pattern_types is None:
        pattern_types = ["streaks", "fatigue", "improvement", "consistency"]

    patterns = {"streaks": {}, "fatigue": {}, "improvement": {}, "consistency": {}}

    try:
        # Convert dict to DataFrame if needed
        if isinstance(pitcher_data, dict):
            if "raw_data" in pitcher_data:
                df = pitcher_data["raw_data"]
            else:
                # Assume it's split data
                combined_data = []
                for period, data in pitcher_data.items():
                    if data is not None and not data.empty:
                        combined_data.append(data)
                df = (
                    pd.concat(combined_data, ignore_index=True)
                    if combined_data
                    else pd.DataFrame()
                )
        else:
            df = pitcher_data

        if df.empty:
            return patterns

        # Group by pitcher for individual analysis
        for pitcher_name in df["Name"].unique():
            pitcher_df = df[df["Name"] == pitcher_name].copy()

            if len(pitcher_df) < 3:  # Need at least 3 periods for pattern analysis
                continue

            # Sort chronologically
            if "Start_Date" in pitcher_df.columns:
                pitcher_df = pitcher_df.sort_values("Start_Date")

            # STREAK ANALYSIS
            if "streaks" in pattern_types:
                patterns["streaks"][pitcher_name] = _identify_streaks(pitcher_df)

            # FATIGUE ANALYSIS
            if "fatigue" in pattern_types:
                patterns["fatigue"][pitcher_name] = _identify_fatigue_patterns(
                    pitcher_df
                )

            # IMPROVEMENT ANALYSIS
            if "improvement" in pattern_types:
                patterns["improvement"][pitcher_name] = _identify_improvement_patterns(
                    pitcher_df
                )

            # CONSISTENCY ANALYSIS
            if "consistency" in pattern_types:
                patterns["consistency"][pitcher_name] = _analyze_consistency_patterns(
                    pitcher_df
                )

        return patterns

    except Exception as e:
        return {"error": f"Error identifying patterns: {e}"}


# =============================================================================
# HELPER FUNCTIONS FOR PATTERN ANALYSIS
# =============================================================================


def _identify_streaks(pitcher_df, streak_length=3):
    """Identify hot and cold streaks based on FIP performance."""
    if "FIP" not in pitcher_df.columns or len(pitcher_df) < streak_length:
        return {}

    fip_values = pitcher_df["FIP"].dropna()
    if len(fip_values) < streak_length:
        return {}

    # Calculate rolling average for streak detection
    rolling_fip = fip_values.rolling(window=streak_length).mean()

    # Define thresholds (league average ~4.00)
    hot_threshold = 3.50  # Good performance
    cold_threshold = 4.50  # Poor performance

    streaks = {"hot_streaks": [], "cold_streaks": [], "current_form": "average"}

    # Identify streaks
    for i, fip_avg in enumerate(rolling_fip.dropna()):
        if fip_avg <= hot_threshold:
            period_start = max(0, i - streak_length + 1)
            streaks["hot_streaks"].append(
                {
                    "periods": list(range(period_start, i + 1)),
                    "avg_fip": fip_avg,
                    "length": streak_length,
                }
            )
        elif fip_avg >= cold_threshold:
            period_start = max(0, i - streak_length + 1)
            streaks["cold_streaks"].append(
                {
                    "periods": list(range(period_start, i + 1)),
                    "avg_fip": fip_avg,
                    "length": streak_length,
                }
            )

    # Current form (last 2 periods)
    if len(fip_values) >= 2:
        recent_fip = fip_values.iloc[-2:].mean()
        if recent_fip <= hot_threshold:
            streaks["current_form"] = "hot"
        elif recent_fip >= cold_threshold:
            streaks["current_form"] = "cold"

    return streaks


def _identify_fatigue_patterns(pitcher_df):
    """Identify potential fatigue patterns in K-BB% and velocity."""
    fatigue_indicators = {}

    # K-BB% decline over time
    if "K-BB%" in pitcher_df.columns:
        k_bb_values = pitcher_df["K-BB%"].dropna()
        if len(k_bb_values) >= 4:
            # Check for consistent decline
            if SCIPY_AVAILABLE:
                x_vals = range(len(k_bb_values))
                slope, _, r_val, p_val, _ = linregress(x_vals, k_bb_values)

                fatigue_indicators["k_bb_decline"] = {
                    "slope": slope,
                    "r_squared": r_val**2,
                    "significant": p_val < 0.05,
                    "fatigue_pattern": slope < -1.0 and p_val < 0.05,
                }

    # Check for late-season decline patterns
    if len(pitcher_df) >= 4:
        # Compare first half vs second half
        midpoint = len(pitcher_df) // 2
        first_half = pitcher_df.iloc[:midpoint]
        second_half = pitcher_df.iloc[midpoint:]

        for metric in ["K-BB%", "WHIP", "FIP"]:
            if metric in pitcher_df.columns:
                first_avg = first_half[metric].mean()
                second_avg = second_half[metric].mean()

                # For K-BB%, higher is better; for WHIP/FIP, lower is better
                if metric == "K-BB%":
                    decline = first_avg - second_avg
                    fatigue_pattern = decline > 2.0  # Significant decline
                else:
                    decline = second_avg - first_avg
                    fatigue_pattern = decline > 0.5  # Significant increase

                fatigue_indicators[f"{metric}_late_season"] = {
                    "first_half_avg": first_avg,
                    "second_half_avg": second_avg,
                    "decline": decline,
                    "fatigue_pattern": fatigue_pattern,
                }

    return fatigue_indicators


def _identify_improvement_patterns(pitcher_df):
    """Identify improvement trends in command and effectiveness."""
    improvement_patterns = {}

    # Check for consistent improvement in key metrics
    improvement_metrics = ["K-BB%", "WHIP", "FIP"]

    for metric in improvement_metrics:
        if metric in pitcher_df.columns:
            values = pitcher_df[metric].dropna()

            if len(values) >= 3 and SCIPY_AVAILABLE:
                x_vals = range(len(values))
                slope, _, r_val, p_val, _ = linregress(x_vals, values)

                # For K-BB%, positive slope is improvement; for WHIP/FIP, negative is improvement
                if metric == "K-BB%":
                    improving = slope > 0.5 and p_val < 0.05
                    improvement_strength = slope
                else:
                    improving = slope < -0.05 and p_val < 0.05
                    improvement_strength = -slope  # Make positive for consistency

                improvement_patterns[metric] = {
                    "improving": improving,
                    "trend_slope": slope,
                    "strength": improvement_strength,
                    "r_squared": r_val**2,
                    "significant": p_val < 0.05,
                }

    # Overall improvement assessment
    improving_metrics = sum(
        1
        for pattern in improvement_patterns.values()
        if pattern.get("improving", False)
    )
    total_metrics = len(improvement_patterns)

    improvement_patterns["overall_improvement"] = {
        "improving_metrics_count": improving_metrics,
        "total_metrics": total_metrics,
        "improvement_ratio": (
            improving_metrics / total_metrics if total_metrics > 0 else 0
        ),
        "strong_improvement": improving_metrics >= 2
        and improving_metrics / total_metrics >= 0.67,
    }

    return improvement_patterns


def _analyze_consistency_patterns(pitcher_df):
    """Analyze consistency in performance metrics."""
    consistency_analysis = {}

    consistency_metrics = ["FIP", "WHIP", "K-BB%", "K/9", "BB/9"]

    for metric in consistency_metrics:
        if metric in pitcher_df.columns:
            values = pitcher_df[metric].dropna()

            if len(values) >= 3:
                mean_val = values.mean()
                std_val = values.std()
                cv = (
                    std_val / mean_val if mean_val != 0 else float("inf")
                )  # Coefficient of variation

                # Define consistency thresholds (lower CV = more consistent)
                if metric in ["FIP", "WHIP"]:
                    very_consistent = cv < 0.10
                    consistent = cv < 0.15
                elif metric == "K-BB%":
                    very_consistent = cv < 0.20
                    consistent = cv < 0.30
                else:  # K/9, BB/9
                    very_consistent = cv < 0.15
                    consistent = cv < 0.25

                consistency_level = (
                    "very_consistent"
                    if very_consistent
                    else "consistent" if consistent else "inconsistent"
                )

                consistency_analysis[metric] = {
                    "coefficient_of_variation": cv,
                    "standard_deviation": std_val,
                    "mean": mean_val,
                    "consistency_level": consistency_level,
                    "range": values.max() - values.min(),
                }

    # Overall consistency score
    consistent_metrics = sum(
        1
        for analysis in consistency_analysis.values()
        if analysis.get("consistency_level") in ["consistent", "very_consistent"]
    )
    total_metrics = len(consistency_analysis)

    consistency_analysis["overall_consistency"] = {
        "consistent_metrics_count": consistent_metrics,
        "total_metrics": total_metrics,
        "consistency_ratio": (
            consistent_metrics / total_metrics if total_metrics > 0 else 0
        ),
        "overall_level": (
            "very_consistent"
            if consistent_metrics == total_metrics
            else (
                "mostly_consistent"
                if consistent_metrics >= total_metrics * 0.75
                else "inconsistent"
            )
        ),
    }

    return consistency_analysis


def _assess_pitcher_trajectory(metrics_analysis):
    """Assess overall pitcher trajectory based on multiple metrics."""
    if not metrics_analysis:
        return "insufficient_data"

    improvement_indicators = 0
    decline_indicators = 0
    total_indicators = 0

    for metric, analysis in metrics_analysis.items():
        if "trend_direction" in analysis:
            total_indicators += 1
            if analysis["trend_direction"] == "improving":
                improvement_indicators += 1
            else:
                decline_indicators += 1

    if total_indicators == 0:
        return "stable"

    improvement_ratio = improvement_indicators / total_indicators

    if improvement_ratio >= 0.75:
        return "strong_improvement"
    elif improvement_ratio >= 0.60:
        return "improving"
    elif improvement_ratio >= 0.40:
        return "stable"
    elif improvement_ratio >= 0.25:
        return "declining"
    else:
        return "strong_decline"
