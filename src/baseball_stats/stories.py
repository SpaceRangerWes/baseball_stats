"""
Storytelling and educational demonstration module.
Handles narrative generation and educational examples.
"""

from .analysis import (
    get_modern_hitting_stats, get_modern_pitching_stats,
    analyze_hitter_types, get_hitter_examples_by_type,
    find_stat_examples, get_elite_hitters, 
    analyze_player_contact_quality, get_player_performance_rating,
    find_player_by_name, analyze_bat_tracking_metrics,
    analyze_baserunning_advanced, analyze_defensive_metrics,
    calculate_advanced_pitcher_metrics, get_situational_performance,
    classify_modern_hitter_type
)
from .data import get_player_lookup, get_player_statcast_data, get_bat_tracking_data


def demonstrate_stat_lies(hitting_data, pitching_data):
    """Show examples of stats that lie vs stats that tell truth"""
    print("=== STATS THAT LIE EXAMPLES ===\n")
    
    examples = find_stat_examples(hitting_data, pitching_data)
    
    print("1. BATTING AVERAGE LIES - OBP TELLS TRUTH")
    print("Players with high AVG but mediocre OBP (probably don't walk):")
    print(examples['high_avg_low_obp'])
    print()
    
    print("2. ERA LIES - FIP TELLS TRUTH")
    print("Unlucky pitchers (ERA much higher than FIP):")
    print(examples['unlucky_pitchers'])
    print()
    
    print("3. RBI DEPENDS ON TEAMMATES")
    print("High RBI leaders - check their wRC+ for actual value:")
    print(examples['high_rbi'])
    print()


def tell_hitter_type_stories(hitting_data):
    """Analyze and tell stories about different hitter archetypes"""
    print("=== HITTER TYPE ANALYSIS ===\n")
    
    type_summary = analyze_hitter_types(hitting_data)
    print("Average stats by hitter type:")
    print(type_summary)
    print()
    
    for hitter_type in hitting_data['Hitter_Type'].unique():
        examples = get_hitter_examples_by_type(hitting_data, hitter_type, 2)
        print(f"{hitter_type} Examples:")
        print(examples[['Name', 'wRC+', 'K%', 'BB%', 'ISO', 'OPS']])
        print()


def show_context_matters(hitting_data):
    """Demonstrate position adjustments and context"""
    print("=== POSITION CONTEXT MATTERS ===\n")
    
    print("Key insight from the cheat sheet:")
    print("- Catcher with .750 OPS = Good (defense matters)")
    print("- First baseman with .750 OPS = Bad (must hit)")
    print("- Use wRC+ which adjusts for position context")
    print()
    
    elite_hitters = get_elite_hitters(hitting_data)
    print(f"Elite hitters (wRC+ > 140) - MVP candidates:")
    print(elite_hitters[['Name', 'Team', 'wRC+', 'OPS']].head())


def get_player_story(player_name, year=2025):
    """Get the 'story' a player's stats tell"""
    hitting_data = get_modern_hitting_stats(year)
    player = find_player_by_name(hitting_data, player_name)
    
    if player is None:
        print(f"Player containing '{player_name}' not found")
        return
    
    print(f"\n=== THE STORY FOR {player['Name']} ===")
    print(f"Team: {player['Team']}")
    print(f"wRC+: {player['wRC+']} ({get_player_performance_rating(player['wRC+'])})")
    print(f"Type: {player['Hitter_Type']}")
    print(f"K%: {player['K%']:.1f}% | BB%: {player['BB%']:.1f}% | ISO: {player['ISO']:.3f}")
    print(f"OBP: {player['OBP']:.3f} | OPS: {player['OPS']:.3f}")
    
    print("\nStory Analysis:")
    if player['K%'] > 25 and player['ISO'] > 0.200:
        print("Power hitter who strikes out but can change the game with one swing")
    elif player['K%'] < 15:
        print("Contact hitter who puts the ball in play consistently")
    elif player['BB%'] > 12:
        print("Patient hitter with good plate discipline")
    else:
        print("Balanced approach hitter")


def quality_contact_story(player_name, days=30):
    """Tell the quality of contact story for a player"""
    print(f"Analyzing quality of contact for {player_name} (last {days} days)")
    
    contact_metrics = analyze_player_contact_quality(player_name, days)
    
    if contact_metrics is None:
        print(f"Could not analyze contact quality for {player_name}")
        return
    
    print(f"\nQuality of Contact Analysis for {player_name}:")
    print(f"Average Exit Velocity: {contact_metrics['avg_exit_velo']:.1f} mph")
    print(f"Hard Hit Rate (95+ mph): {contact_metrics['hard_hit_rate']:.1f}%")
    print(f"Sweet Spot Rate (8-32°): {contact_metrics['sweet_spot_rate']:.1f}%")
    
    if contact_metrics['barrel_rate']:
        print(f"Barrel Rate: {contact_metrics['barrel_rate']:.1f}%")
    
    print("\nContact Quality Story:")
    if contact_metrics['hard_hit_rate'] > 45:
        print("Elite contact quality - consistently hitting the ball hard")
    elif contact_metrics['hard_hit_rate'] > 35:
        print("Good contact quality - making solid contact regularly")
    else:
        print("Needs to improve contact quality - focus on harder hit balls")


def quick_learning_examples(year=2025):
    """Run the key educational examples"""
    print("=== BASEBALL STATS LEARNING TOOL ===")
    print("Based on modern analytics priorities\n")
    
    hitting_data = get_modern_hitting_stats(year)
    pitching_data = get_modern_pitching_stats(year)
    
    print(f"Loaded {len(hitting_data)} qualified hitters and {len(pitching_data)} qualified pitchers\n")
    
    demonstrate_stat_lies(hitting_data, pitching_data)
    tell_hitter_type_stories(hitting_data)
    show_context_matters(hitting_data)
    
    return hitting_data, pitching_data


def tell_complete_story(player_name, year=2025):
    """Tell the complete analytical story for a player"""
    get_player_story(player_name, year)
    quality_contact_story(player_name)


def tell_modern_bat_tracking_story(player_name, year=2025):
    """Tell the bat tracking story for a player (2023+ data)"""
    print(f"\n=== BAT TRACKING ANALYSIS FOR {player_name} ===")
    
    # Get player lookup
    name_parts = player_name.split()
    if len(name_parts) != 2:
        print("Please provide first and last name")
        return
    
    first_name, last_name = name_parts
    player_lookup = get_player_lookup(first_name, last_name)
    
    if player_lookup.empty:
        print(f"Player {player_name} not found")
        return
    
    player_id = player_lookup.iloc[0]['key_mlbam']
    
    # Get bat tracking data for the season
    start_date = f"{year}-04-01"
    end_date = f"{year}-10-31"
    
    try:
        bat_data = get_bat_tracking_data(start_date, end_date)
        if bat_data is not None:
            player_data = bat_data[bat_data['batter'] == player_id]
            
            if not player_data.empty:
                metrics = analyze_bat_tracking_metrics(player_data)
                
                if metrics:
                    print(f"Bat Tracking Metrics for {player_name} ({year}):")
                    
                    if 'avg_bat_speed' in metrics:
                        print(f"Average Bat Speed: {metrics['avg_bat_speed']:.1f} mph")
                        print(f"Max Bat Speed: {metrics['max_bat_speed']:.1f} mph")
                        print(f"Fast Swing Rate (75+ mph): {metrics['fast_swing_rate']:.1f}%")
                    
                    if 'avg_swing_length' in metrics:
                        print(f"Average Swing Length: {metrics['avg_swing_length']:.1f} feet")
                        print(f"Efficient Swing Rate (≤7 ft): {metrics['efficient_swing_rate']:.1f}%")
                    
                    if 'squared_up_rate' in metrics:
                        print(f"Squared-up Rate: {metrics['squared_up_rate']:.1f}%")
                    
                    if 'blasts' in metrics:
                        print(f"Blasts (Elite Contact): {metrics['blasts']}")
                        print(f"Blast Rate: {metrics['blast_rate']:.1f}%")
                    
                    # Tell the story
                    print(f"\nBat Tracking Story for {player_name}:")
                    if metrics.get('avg_bat_speed', 0) >= 75 and metrics.get('squared_up_rate', 0) >= 25:
                        print("Elite combination of bat speed and contact quality - true power threat")
                    elif metrics.get('avg_bat_speed', 0) >= 75:
                        print("Elite bat speed but needs to improve contact quality")
                    elif metrics.get('squared_up_rate', 0) >= 25:
                        print("Excellent contact quality with room to add bat speed")
                    else:
                        print("Focus on improving both bat speed and contact quality")
                else:
                    print("Bat tracking data available but no metrics calculated")
            else:
                print(f"No bat tracking data found for {player_name} in {year}")
        else:
            print(f"Bat tracking data not available for {year}")
    except Exception as e:
        print(f"Error analyzing bat tracking: {e}")


def tell_defensive_story(year=2025):
    """Tell stories about defensive excellence and struggles"""
    print(f"\n=== DEFENSIVE EXCELLENCE & STRUGGLES ({year}) ===")
    
    defensive_analysis = analyze_defensive_metrics(year)
    
    if defensive_analysis is None:
        print("Defensive metrics not available")
        return
    
    if 'oaa_leaders' in defensive_analysis:
        print("DEFENSIVE STARS (Outs Above Average Leaders):")
        print(defensive_analysis['oaa_leaders'])
        print()
        
        print("DEFENSIVE STRUGGLES (Lowest OAA):")
        print(defensive_analysis['oaa_worst'])
        print()
    
    if 'catch_prob_leaders' in defensive_analysis:
        print("OUTFIELD EXCELLENCE (Catch Probability Leaders):")
        print(defensive_analysis['catch_prob_leaders'])
        print()
    
    print("Defensive Story Insights:")
    print("- OAA measures defensive value above/below average")
    print("- Catch probability shows outfielder efficiency on makeable plays")
    print("- Elite defenders can save 10+ runs per season")


def tell_baserunning_story(year=2025):
    """Tell stories about speed and baserunning"""
    print(f"\n=== SPEED & BASERUNNING ANALYSIS ({year}) ===")
    
    baserunning_analysis = analyze_baserunning_advanced(year)
    
    if baserunning_analysis is None:
        print("Baserunning metrics not available")
        return
    
    if 'sprint_speed' in baserunning_analysis:
        sprint_data = baserunning_analysis['sprint_speed']
        print(f"League Average Sprint Speed: {sprint_data['avg_sprint_speed']:.1f} ft/sec")
        print(f"Elite Speed Players (28+ ft/sec): {sprint_data['elite_speed_players']}")
        print()
        
        print("SPEED DEMONS (Sprint Speed Leaders):")
        print(sprint_data['speed_leaders'])
        print()
    
    if 'home_to_first' in baserunning_analysis:
        h2f_data = baserunning_analysis['home_to_first']
        print(f"Average Home-to-First Time: {h2f_data['avg_time']:.2f} seconds")
        print("FASTEST HOME-TO-FIRST TIMES:")
        print(h2f_data['fastest_times'])
        print()
    
    print("Baserunning Story Insights:")
    print("- 28+ ft/sec sprint speed = Elite (top 10%)")
    print("- Sub-4.2 second home-to-first = Excellent")
    print("- Speed creates value through stolen bases, extra bases, and defense")


def tell_situational_story(player_name, year=2025):
    """Tell how a player performs in clutch situations"""
    print(f"\n=== SITUATIONAL PERFORMANCE FOR {player_name} ===")
    
    # Get player data
    name_parts = player_name.split()
    if len(name_parts) != 2:
        print("Please provide first and last name")
        return
    
    first_name, last_name = name_parts
    player_lookup = get_player_lookup(first_name, last_name)
    
    if player_lookup.empty:
        print(f"Player {player_name} not found")
        return
    
    player_id = player_lookup.iloc[0]['key_mlbam']
    
    try:
        player_data = get_player_statcast_data(player_id, days=365)  # Full season
        
        if player_data is not None and not player_data.empty:
            # Analyze different situations
            situations = ['high_leverage', 'two_strikes', 'risp']
            
            for situation in situations:
                sit_performance = get_situational_performance(player_data, situation)
                
                if sit_performance:
                    for key, metrics in sit_performance.items():
                        print(f"\n{key.upper().replace('_', ' ')}:")
                        print(f"Total Plate Appearances: {metrics['total_abs']}")
                        
                        if 'avg_exit_velo' in metrics and metrics['avg_exit_velo']:
                            print(f"Average Exit Velocity: {metrics['avg_exit_velo']:.1f} mph")
                        
                        if 'barrel_rate' in metrics and metrics['barrel_rate']:
                            print(f"Barrel Rate: {metrics['barrel_rate']:.1f}%")
                        
                        if 'hard_hit_rate' in metrics and metrics['hard_hit_rate']:
                            print(f"Hard Hit Rate: {metrics['hard_hit_rate']:.1f}%")
                        
                        if 'whiff_rate' in metrics:
                            print(f"Whiff Rate: {metrics['whiff_rate']:.1f}%")
                        
                        if 'contact_rate' in metrics:
                            print(f"Contact Rate: {metrics['contact_rate']:.1f}%")
            
            print(f"\nSituational Story for {player_name}:")
            print("- High leverage: Performance in crucial moments")
            print("- Two strikes: Ability to fight off tough pitches")
            print("- RISP: Clutch hitting with runners in scoring position")
        else:
            print(f"No situational data available for {player_name}")
    
    except Exception as e:
        print(f"Error analyzing situational performance: {e}")


def tell_comprehensive_modern_story(player_name, year=2025):
    """Tell the complete modern baseball story for a player"""
    print(f"=== COMPREHENSIVE MODERN ANALYSIS: {player_name} ({year}) ===\n")
    
    # Traditional story
    get_player_story(player_name, year)
    
    # Modern contact quality
    quality_contact_story(player_name)
    
    # Bat tracking (if available)
    tell_modern_bat_tracking_story(player_name, year)
    
    # Situational performance
    tell_situational_story(player_name, year)
    
    print(f"\n=== END COMPREHENSIVE ANALYSIS FOR {player_name} ===")


def demonstrate_modern_analytics_revolution():
    """Demonstrate the revolution in baseball analytics"""
    print("=== THE MODERN BASEBALL ANALYTICS REVOLUTION ===\n")
    
    print("2024-2025 CUTTING-EDGE METRICS:")
    print("1. BAT TRACKING (2023+)")
    print("   - Bat Speed: How fast the barrel moves through the zone")
    print("   - Swing Length: Efficiency of swing path")
    print("   - Squared-up Rate: Perfect barrel-to-ball contact")
    print("   - Blasts: Elite contact (squared-up + high bat speed)")
    print()
    
    print("2. DEFENSIVE REVOLUTION")
    print("   - Outs Above Average (OAA): Runs saved/cost on defense")
    print("   - Catch Probability: Expected success rate on each play")
    print("   - Positioning: Real-time defensive alignment optimization")
    print()
    
    print("3. SITUATIONAL ANALYTICS")
    print("   - Leverage Index: Importance of each plate appearance")
    print("   - WPA (Win Probability Added): Situational impact")
    print("   - Count-specific performance: Success in 2-strike counts")
    print()
    
    print("4. PREDICTIVE METRICS")
    print("   - Expected stats vs. actual: Luck vs. skill separation")
    print("   - Injury risk models: Biomechanical breakdown prediction")
    print("   - Performance sustainability: Regression forecasting")
    print()
    
    print("THE STORY THESE METRICS TELL:")
    print("Baseball has evolved from simple counting stats to comprehensive")
    print("player evaluation that considers every aspect of the game.")
    print("Modern analytics separate luck from skill, predict future")
    print("performance, and optimize every decision on the field.")
    print()
    
    print("This represents the biggest analytical revolution in baseball")
    print("since the original sabermetrics movement of the 1980s-2000s.")


def create_hitter_radar_chart():
    """Create a radar chart for different hitter types"""
    import plotly.graph_objects as go
    
    # Define categories and their typical ranges
    categories = ['Power', 'Contact', 'Patience', 'Speed', 'Consistency']
    
    # Example values for different hitter types (normalized 0-100)
    hitter_profiles = {
        'Three True Outcomes': [90, 40, 70, 30, 50],
        'Contact Artist': [30, 90, 60, 70, 80],
        'Elite Hitter': [80, 80, 85, 60, 90],
        'Average Hitter': [50, 50, 50, 50, 50]
    }
    
    fig = go.Figure()
    
    colors = ['red', 'blue', 'green', 'orange']
    for i, (hitter_type, values) in enumerate(hitter_profiles.items()):
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=hitter_type,
            line_color=colors[i]
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title="Hitter Type Characteristics Profile",
        width=800,
        height=600
    )
    
    return fig


def create_metrics_explanation():
    """Create educational visualization explaining advanced metrics"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Define metric categories and their importance
    metric_categories = {
        'Swing Mechanics': {
            'Bat Speed': 'Raw power potential - faster bat = more power',
            'Swing Length': 'Efficiency - shorter swing = better contact',
            'Attack Angle': 'Launch angle optimization for power'
        },
        'Plate Discipline': {
            'Whiff Rate': 'Ability to make contact - lower is better',
            'Chase Rate': 'Pitch recognition - chasing bad pitches',
            'Zone Contact': 'Contact on hittable pitches'
        },
        'Expected Performance': {
            'xwOBA': 'Expected weighted on-base average',
            'xBA': 'Expected batting average based on contact quality',
            'xSLG': 'Expected slugging based on exit velocity/angle'
        },
        'Contact Quality': {
            'Barrel Rate': 'Percentage of ideal contact (98+ mph, 26-30°)',
            'Hard Hit Rate': 'Percentage of 95+ mph exit velocity',
            'Sweet Spot %': 'Percentage of 8-32° launch angle'
        }
    }
    
    # Create metric importance visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Advanced Metrics Educational Breakdown', fontsize=16, fontweight='bold')
    
    # Metric importance radar chart data
    categories = list(metric_categories.keys())
    importance_scores = [85, 90, 95, 88]  # Subjective importance scores
    
    # Create pie chart for metric category importance
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    axes[0, 0].pie(importance_scores, labels=categories, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Relative Importance of Metric Categories')
    
    # Create metric timeline (when they became important)
    timeline_data = {
        'Traditional Era (1900-1990)': ['BA', 'RBI', 'HR', 'Wins', 'ERA'],
        'Sabermetric Era (1990-2010)': ['OBP', 'SLG', 'OPS', 'WHIP', 'FIP'],
        'Statcast Era (2010-Present)': ['xwOBA', 'Barrel%', 'Exit Velocity', 'Launch Angle']
    }
    
    y_pos = 0
    colors_timeline = ['lightcoral', 'lightblue', 'lightgreen']
    for i, (era, metrics) in enumerate(timeline_data.items()):
        axes[0, 1].barh(y_pos, len(metrics), color=colors_timeline[i], alpha=0.7, 
                       label=era, height=0.8)
        axes[0, 1].text(len(metrics)/2, y_pos, f'{len(metrics)} metrics', 
                       ha='center', va='center', fontweight='bold')
        y_pos += 1
    
    axes[0, 1].set_yticks(range(len(timeline_data)))
    axes[0, 1].set_yticklabels(timeline_data.keys())
    axes[0, 1].set_xlabel('Number of Key Metrics')
    axes[0, 1].set_title('Evolution of Baseball Analytics')
    axes[0, 1].legend()
    
    # Create correlation strength visualization
    metric_correlations = {
        'BA vs wRC+': 0.65,
        'OPS vs wRC+': 0.92,
        'ERA vs FIP': 0.78,
        'xwOBA vs wOBA': 0.85,
        'Barrel% vs SLG': 0.81,
        'K% vs Whiff%': 0.89
    }
    
    metric_names = list(metric_correlations.keys())
    correlation_values = list(metric_correlations.values())
    
    bars = axes[1, 0].bar(range(len(metric_names)), correlation_values, 
                         color=['red' if x < 0.7 else 'yellow' if x < 0.85 else 'green' 
                               for x in correlation_values])
    axes[1, 0].set_xticks(range(len(metric_names)))
    axes[1, 0].set_xticklabels(metric_names, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Correlation Coefficient')
    axes[1, 0].set_title('Metric Correlation Strengths')
    axes[1, 0].set_ylim(0, 1)
    
    # Add correlation strength indicators
    for i, (bar, value) in enumerate(zip(bars, correlation_values)):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Create predictive power comparison
    predictive_power = {
        'Traditional': [0.45, 0.52, 0.38, 0.41],  # BA, RBI, Wins, ERA
        'Modern': [0.78, 0.85, 0.82, 0.79]       # wRC+, wOBA, FIP, xwOBA
    }
    
    x = np.arange(4)
    width = 0.35
    
    bars1 = axes[1, 1].bar(x - width/2, predictive_power['Traditional'], width, 
                          label='Traditional', color='lightcoral', alpha=0.7)
    bars2 = axes[1, 1].bar(x + width/2, predictive_power['Modern'], width, 
                          label='Modern', color='lightgreen', alpha=0.7)
    
    axes[1, 1].set_xlabel('Metric Type')
    axes[1, 1].set_ylabel('Predictive R²')
    axes[1, 1].set_title('Predictive Power: Traditional vs Modern')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Hitting', 'Overall Off.', 'Pitching', 'Expected'])
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()
    
    return metric_categories


def create_advanced_metrics_analysis():
    """Create analysis of cutting-edge baseball metrics concepts"""
    import numpy as np
    import pandas as pd
    
    # Generate synthetic advanced metrics data for demonstration
    n_players = 150  # Example size
    
    advanced_metrics = pd.DataFrame({
        'Bat_Speed': np.random.normal(72, 3, n_players),
        'Swing_Length': np.random.normal(7.2, 0.8, n_players),
        'xwOBA': np.random.normal(0.320, 0.040, n_players),
        'Whiff_Rate': np.random.normal(25, 5, n_players),
        'Chase_Rate': np.random.normal(30, 6, n_players),
        'Zone_Contact_Rate': np.random.normal(85, 4, n_players),
        'Swing_Aggression': np.random.normal(45, 8, n_players)
    })
    
    # Ensure realistic ranges
    advanced_metrics['Bat_Speed'] = np.clip(advanced_metrics['Bat_Speed'], 65, 80)
    advanced_metrics['Swing_Length'] = np.clip(advanced_metrics['Swing_Length'], 5.5, 9.0)
    advanced_metrics['xwOBA'] = np.clip(advanced_metrics['xwOBA'], 0.250, 0.420)
    advanced_metrics['Whiff_Rate'] = np.clip(advanced_metrics['Whiff_Rate'], 15, 40)
    advanced_metrics['Chase_Rate'] = np.clip(advanced_metrics['Chase_Rate'], 18, 45)
    advanced_metrics['Zone_Contact_Rate'] = np.clip(advanced_metrics['Zone_Contact_Rate'], 75, 95)
    advanced_metrics['Swing_Aggression'] = np.clip(advanced_metrics['Swing_Aggression'], 25, 65)
    
    return advanced_metrics


def create_situational_analysis(n_players=150):
    """Create situational performance analysis
    
    Args:
        n_players: Number of players to generate data for (default: 150)
    """
    import numpy as np
    import pandas as pd
    
    # Generate synthetic situational data for demonstration
    situational_data = pd.DataFrame({
        'RISP_OPS': np.random.normal(0.750, 0.100, n_players),
        'High_Leverage_OPS': np.random.normal(0.740, 0.120, n_players),
        'Two_Strike_BA': np.random.normal(0.180, 0.030, n_players),
        'Late_Inning_OPS': np.random.normal(0.730, 0.110, n_players),
        'Clutch_Score': np.random.normal(0.0, 1.0, n_players)
    })
    
    # Ensure realistic ranges
    situational_data['RISP_OPS'] = np.clip(situational_data['RISP_OPS'], 0.500, 1.200)
    situational_data['High_Leverage_OPS'] = np.clip(situational_data['High_Leverage_OPS'], 0.400, 1.300)
    situational_data['Two_Strike_BA'] = np.clip(situational_data['Two_Strike_BA'], 0.100, 0.280)
    situational_data['Late_Inning_OPS'] = np.clip(situational_data['Late_Inning_OPS'], 0.400, 1.200)
    situational_data['Clutch_Score'] = np.clip(situational_data['Clutch_Score'], -3.0, 3.0)
    
    return situational_data