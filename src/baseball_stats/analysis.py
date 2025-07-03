"""
Analysis and classification module for baseball statistics.
Handles statistical calculations, player classification, and data processing.
"""

import pandas as pd
import numpy as np
from .data import (
    get_qualified_hitters, get_qualified_pitchers, get_player_lookup, 
    get_player_statcast_data, get_bat_tracking_data, get_fielding_metrics,
    get_sprint_speed_data, get_baserunning_metrics, get_fangraphs_hitting_data,
    get_fangraphs_pitching_data, get_modern_statcast_data
)


def get_key_hitting_columns():
    """Return the key columns for hitting analysis"""
    return [
        'Name', 'Team', 'G', 'PA', 'AB',
        'AVG', 'OBP', 'SLG', 'OPS', 'wRC+',
        'K%', 'BB%', 'ISO', 'BABIP', 'wOBA',
        'Barrel%', 'HardHit%', 'Pull%', 'Cent%', 'Oppo%',
        'R', 'RBI', 'HR', '2B', '3B'
    ]


def get_modern_hitting_columns():
    """Return columns for modern hitting analysis including advanced metrics"""
    base_cols = get_key_hitting_columns()
    modern_cols = [
        'EV', 'LA', 'MaxEV', 'Sweet_Spot%', 'Flare/Burner%',
        'Solid_Contact%', 'Barrel%', 'HardHit%', 'xBA', 'xSLG', 'xwOBA',
        'xOBA_minus_OBA', 'Sprint_Speed', 'Baserunning', 'SB_Rate'
    ]
    return base_cols + modern_cols


def get_key_pitching_columns():
    """Return the key columns for pitching analysis"""
    return [
        'Name', 'Team', 'G', 'GS', 'IP',
        'ERA', 'FIP', 'WHIP', 'K/9', 'BB/9',
        'K%', 'BB%', 'K-BB%', 'HR/9', 'HR/FB',
        'BABIP', 'GB%', 'FB%', 'LOB%', 'ERA-',
        'W', 'L', 'SV'
    ]


def classify_hitter_type(row):
    """Classify hitters based on statistical patterns"""
    k_rate = row['K%']
    bb_rate = row['BB%']
    iso = row['ISO']
    
    if k_rate > 28 and bb_rate > 10 and iso > 0.200:
        return "Three True Outcomes"
    elif k_rate < 15 and iso < 0.150:
        return "Contact Artist"
    elif row['wRC+'] > 120:
        return "Elite Hitter"
    else:
        return "Average Hitter"


def classify_pitcher_type(row):
    """Classify pitchers based on statistical patterns"""
    k_9 = row['K/9']
    bb_9 = row['BB/9']
    
    if k_9 > 10:
        return "Flamethrower" if bb_9 > 3 else "Power Pitcher"
    elif k_9 < 7 and bb_9 < 2:
        return "Crafty Veteran"
    elif row['FIP'] < 3.50:
        return "Effective Pitcher"
    else:
        return "Average Pitcher"


def get_modern_hitting_stats(year=2025):
    """Get hitting data focused on meaningful stats"""
    hitting_data = get_qualified_hitters(year)
    key_columns = get_key_hitting_columns()
    
    hitting_data['Hitter_Type'] = hitting_data.apply(classify_hitter_type, axis=1)
    
    return hitting_data[key_columns + ['Hitter_Type']]


def get_modern_pitching_stats(year=2025):
    """Get pitching data focused on meaningful stats"""
    pitching_data = get_qualified_pitchers(year)
    key_columns = get_key_pitching_columns()
    
    pitching_data['Pitcher_Type'] = pitching_data.apply(classify_pitcher_type, axis=1)
    pitching_data['ERA_FIP_Diff'] = pitching_data['ERA'] - pitching_data['FIP']
    
    return pitching_data[key_columns + ['Pitcher_Type', 'ERA_FIP_Diff']]


def analyze_hitter_types(hitting_data):
    """Analyze different hitter archetypes"""
    return hitting_data.groupby('Hitter_Type').agg({
        'wRC+': 'mean',
        'K%': 'mean', 
        'BB%': 'mean',
        'ISO': 'mean',
        'OPS': 'mean'
    }).round(3)


def get_hitter_examples_by_type(hitting_data, hitter_type, n=2):
    """Get top examples of a specific hitter type"""
    return hitting_data[hitting_data['Hitter_Type'] == hitter_type].nlargest(n, 'wRC+')


def find_stat_examples(hitting_data, pitching_data):
    """Find examples of misleading vs meaningful stats"""
    examples = {}
    
    # High AVG but low OBP
    examples['high_avg_low_obp'] = hitting_data[
        (hitting_data['AVG'] > 0.300) & (hitting_data['OBP'] < 0.340)
    ][['Name', 'AVG', 'OBP', 'BB%']].head(3)
    
    # Unlucky pitchers (ERA much higher than FIP)
    examples['unlucky_pitchers'] = pitching_data[
        pitching_data['ERA_FIP_Diff'] > 0.75
    ][['Name', 'ERA', 'FIP', 'BABIP', 'LOB%']].head(3)
    
    # High RBI context
    examples['high_rbi'] = hitting_data.nlargest(5, 'RBI')[['Name', 'RBI', 'wRC+', 'Team']]
    
    return examples


def get_elite_hitters(hitting_data, wrc_threshold=140):
    """Get elite hitters above wRC+ threshold"""
    return hitting_data[hitting_data['wRC+'] > wrc_threshold]


def calculate_quality_contact_metrics(statcast_data):
    """Calculate quality of contact metrics from Statcast data"""
    if statcast_data is None or statcast_data.empty:
        return None
    
    return {
        'total_bbevents': len(statcast_data),
        'avg_exit_velo': statcast_data['launch_speed'].mean(),
        'hard_hit_rate': (statcast_data['launch_speed'] >= 95).mean() * 100,
        'barrel_rate': (statcast_data['barrel'] == 1).mean() * 100 if 'barrel' in statcast_data.columns else None,
        'sweet_spot_rate': ((statcast_data['launch_angle'] >= 8) & 
                          (statcast_data['launch_angle'] <= 32)).mean() * 100
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
    
    player_id = player_lookup.iloc[0]['key_mlbam']
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
    player_stats = hitting_data[hitting_data['Name'].str.contains(player_name, case=False)]
    return player_stats.iloc[0] if not player_stats.empty else None


def analyze_bat_tracking_metrics(statcast_data):
    """Analyze modern bat tracking metrics (2023+)"""
    if statcast_data is None or statcast_data.empty:
        return None
    
    bat_tracking_metrics = {}
    
    # Check for bat tracking columns
    if 'bat_speed' in statcast_data.columns:
        bat_tracking_metrics['avg_bat_speed'] = statcast_data['bat_speed'].mean()
        bat_tracking_metrics['max_bat_speed'] = statcast_data['bat_speed'].max()
        bat_tracking_metrics['fast_swing_rate'] = (statcast_data['bat_speed'] >= 75).mean() * 100
    
    if 'swing_length' in statcast_data.columns:
        bat_tracking_metrics['avg_swing_length'] = statcast_data['swing_length'].mean()
        bat_tracking_metrics['efficient_swing_rate'] = (statcast_data['swing_length'] <= 7).mean() * 100
    
    if 'squared_up' in statcast_data.columns:
        bat_tracking_metrics['squared_up_rate'] = (statcast_data['squared_up'] == 1).mean() * 100
    
    # Calculate "blasts" - elite squared-up contact with high bat speed
    if 'bat_speed' in statcast_data.columns and 'squared_up' in statcast_data.columns:
        blasts = ((statcast_data['bat_speed'] >= 75) & (statcast_data['squared_up'] == 1)).sum()
        bat_tracking_metrics['blasts'] = blasts
        bat_tracking_metrics['blast_rate'] = (blasts / len(statcast_data)) * 100
    
    return bat_tracking_metrics


def analyze_baserunning_advanced(year=2024):
    """Analyze advanced baserunning metrics"""
    sprint_data = get_sprint_speed_data(year)
    baserunning_data = get_baserunning_metrics(year)
    
    if sprint_data is None and baserunning_data is None:
        return None
    
    analysis = {}
    
    if sprint_data is not None:
        analysis['sprint_speed'] = {
            'avg_sprint_speed': sprint_data['sprint_speed'].mean(),
            'elite_speed_players': sprint_data[sprint_data['sprint_speed'] >= 28].shape[0],
            'speed_leaders': sprint_data.nlargest(10, 'sprint_speed')[['name', 'sprint_speed']]
        }
    
    if baserunning_data is not None:
        # Calculate baserunning value metrics if available
        if 'hp_to_1b' in baserunning_data.columns:
            analysis['home_to_first'] = {
                'avg_time': baserunning_data['hp_to_1b'].mean(),
                'fastest_times': baserunning_data.nsmallest(10, 'hp_to_1b')[['name', 'hp_to_1b']]
            }
    
    return analysis


def classify_modern_hitter_type(row):
    """Enhanced hitter classification with modern metrics"""
    # Use traditional classification as base
    base_type = classify_hitter_type(row)
    
    # Enhance with modern metrics if available
    enhancements = []
    
    # Bat tracking enhancements
    if 'avg_bat_speed' in row and pd.notna(row['avg_bat_speed']):
        if row['avg_bat_speed'] >= 75:
            enhancements.append("Elite Bat Speed")
        elif row['avg_bat_speed'] <= 70:
            enhancements.append("Controlled Swing")
    
    # Contact quality enhancements
    if 'Barrel%' in row and pd.notna(row['Barrel%']):
        if row['Barrel%'] >= 15:
            enhancements.append("Elite Contact")
        elif row['Barrel%'] >= 10:
            enhancements.append("Quality Contact")
    
    # Speed enhancements
    if 'Sprint_Speed' in row and pd.notna(row['Sprint_Speed']):
        if row['Sprint_Speed'] >= 28:
            enhancements.append("Elite Speed")
        elif row['Sprint_Speed'] >= 27:
            enhancements.append("Above Avg Speed")
    
    if enhancements:
        return f"{base_type} ({', '.join(enhancements)})"
    return base_type


def analyze_defensive_metrics(year=2024):
    """Analyze modern defensive metrics including OAA"""
    fielding_data = get_fielding_metrics(year)
    
    if fielding_data is None:
        return None
    
    analysis = {}
    
    # Outs Above Average analysis
    if 'outs_above_average' in fielding_data.columns:
        analysis['oaa_leaders'] = fielding_data.nlargest(10, 'outs_above_average')[
            ['name_display_first_last', 'primary_pos_txt', 'outs_above_average']
        ]
        analysis['oaa_worst'] = fielding_data.nsmallest(10, 'outs_above_average')[
            ['name_display_first_last', 'primary_pos_txt', 'outs_above_average']
        ]
    
    # Catch probability analysis for outfielders
    if 'catch_probability' in fielding_data.columns:
        of_data = fielding_data[fielding_data['primary_pos_txt'].isin(['OF', 'LF', 'CF', 'RF'])]
        if not of_data.empty:
            analysis['catch_prob_leaders'] = of_data.nlargest(10, 'catch_probability')[
                ['name_display_first_last', 'primary_pos_txt', 'catch_probability']
            ]
    
    return analysis


def calculate_advanced_pitcher_metrics(pitcher_data):
    """Calculate advanced pitcher metrics beyond traditional stats"""
    if pitcher_data is None or pitcher_data.empty:
        return None
    
    advanced_metrics = {}
    
    # Stuff metrics (velocity and movement)
    if 'release_speed' in pitcher_data.columns:
        advanced_metrics['avg_velocity'] = pitcher_data['release_speed'].mean()
        advanced_metrics['max_velocity'] = pitcher_data['release_speed'].max()
        advanced_metrics['velocity_consistency'] = pitcher_data['release_speed'].std()
    
    # Movement metrics
    movement_cols = ['pfx_x', 'pfx_z']
    available_movement = [col for col in movement_cols if col in pitcher_data.columns]
    if available_movement:
        # Calculate total movement
        if len(available_movement) == 2:
            pitcher_data['total_movement'] = np.sqrt(
                pitcher_data['pfx_x']**2 + pitcher_data['pfx_z']**2
            )
            advanced_metrics['avg_movement'] = pitcher_data['total_movement'].mean()
    
    # Command metrics
    if 'zone' in pitcher_data.columns:
        advanced_metrics['zone_rate'] = (pitcher_data['zone'] == 1).mean() * 100
        advanced_metrics['chase_rate'] = (
            (pitcher_data['zone'] == 0) & (pitcher_data['description'] == 'swinging_strike')
        ).mean() * 100
    
    # Expected stats vs actual
    expected_cols = ['estimated_ba_using_speedangle', 'estimated_slg_using_speedangle']
    available_expected = [col for col in expected_cols if col in pitcher_data.columns]
    if available_expected:
        advanced_metrics['expected_metrics'] = {
            col: pitcher_data[col].mean() for col in available_expected
        }
    
    return advanced_metrics


def get_situational_performance(statcast_data, situation='high_leverage'):
    """Analyze performance in specific game situations"""
    if statcast_data is None or statcast_data.empty:
        return None
    
    situational_stats = {}
    
    # High leverage situations (based on inning and score)
    if situation == 'high_leverage':
        # Define high leverage as late innings (7+) or close games
        if 'inning' in statcast_data.columns:
            late_innings = statcast_data['inning'] >= 7
            situational_data = statcast_data[late_innings]
            
            if not situational_data.empty:
                situational_stats['late_inning_performance'] = {
                    'total_abs': len(situational_data),
                    'avg_exit_velo': situational_data['launch_speed'].mean() if 'launch_speed' in situational_data.columns else None,
                    'barrel_rate': (situational_data['barrel'] == 1).mean() * 100 if 'barrel' in situational_data.columns else None
                }
    
    # Two-strike counts
    elif situation == 'two_strikes':
        if 'strikes' in statcast_data.columns:
            two_strike_data = statcast_data[statcast_data['strikes'] == 2]
            if not two_strike_data.empty:
                situational_stats['two_strike_performance'] = {
                    'total_abs': len(two_strike_data),
                    'whiff_rate': (two_strike_data['description'] == 'swinging_strike').mean() * 100,
                    'contact_rate': (~two_strike_data['description'].str.contains('strike', na=False)).mean() * 100
                }
    
    # Runners in scoring position
    elif situation == 'risp':
        if 'on_2b' in statcast_data.columns or 'on_3b' in statcast_data.columns:
            risp_filter = (
                (statcast_data.get('on_2b', 0).fillna(0) > 0) |
                (statcast_data.get('on_3b', 0).fillna(0) > 0)
            )
            risp_data = statcast_data[risp_filter]
            
            if not risp_data.empty:
                situational_stats['risp_performance'] = {
                    'total_abs': len(risp_data),
                    'avg_exit_velo': risp_data['launch_speed'].mean() if 'launch_speed' in risp_data.columns else None,
                    'hard_hit_rate': (risp_data['launch_speed'] >= 95).mean() * 100 if 'launch_speed' in risp_data.columns else None
                }
    
    return situational_stats