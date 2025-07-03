"""
Data extraction module for baseball statistics.
Handles all pybaseball data fetching and basic filtering.
"""

from pybaseball import (
    batting_stats, pitching_stats, statcast,
    playerid_lookup, statcast_pitcher, statcast_batter,
    standings, team_batting, team_pitching,
    fg_batting_data, fg_pitching_data, statcast_sprint_speed
)

# Try to import optional functions that may not be available
try:
    from pybaseball import statcast_fielding
    HAS_FIELDING = True
except ImportError:
    HAS_FIELDING = False

try:
    from pybaseball import statcast_running
    HAS_RUNNING = True
except ImportError:
    HAS_RUNNING = False
import pybaseball.cache
from datetime import datetime, timedelta

pybaseball.cache.enable()


def get_raw_hitting_stats(year=2025):
    """Fetch raw hitting data from pybaseball"""
    print(f"Fetching {year} hitting data...")
    return batting_stats(year)


def get_raw_pitching_stats(year=2025):
    """Fetch raw pitching data from pybaseball"""
    print(f"Fetching {year} pitching data...")
    return pitching_stats(year)


def get_qualified_hitters(year=2025, min_pa=100):
    """Get qualified hitters with minimum plate appearances"""
    batting_data = get_raw_hitting_stats(year)
    return batting_data[batting_data['PA'] >= min_pa].copy()


def get_qualified_pitchers(year=2025, min_ip=20):
    """Get qualified pitchers with minimum innings pitched"""
    pitching_data = get_raw_pitching_stats(year)
    return pitching_data[pitching_data['IP'] >= min_ip].copy()


def get_player_lookup(first_name, last_name):
    """Look up player ID for Statcast data"""
    return playerid_lookup(last_name, first_name)


def get_player_statcast_data(player_id, days=30):
    """Get Statcast data for a specific player"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        return statcast_batter(
            start_dt=start_date.strftime('%Y-%m-%d'),
            end_dt=end_date.strftime('%Y-%m-%d'),
            player_id=player_id
        )
    except Exception as e:
        print(f"Error getting Statcast data: {e}")
        return None


def get_team_stats(year=2025):
    """Get team-level batting and pitching stats"""
    return {
        'batting': team_batting(year),
        'pitching': team_pitching(year)
    }


def get_standings(year=2025):
    """Get league standings"""
    return standings(year)


def get_modern_statcast_data(start_date, end_date):
    """Get comprehensive Statcast data including modern metrics"""
    print(f"Fetching Statcast data from {start_date} to {end_date}...")
    return statcast(start_dt=start_date, end_dt=end_date)


def get_bat_tracking_data(start_date, end_date):
    """Get bat tracking data (available from 2023+)"""
    print(f"Fetching bat tracking data from {start_date} to {end_date}...")
    data = statcast(start_dt=start_date, end_dt=end_date)
    
    # Filter for bat tracking columns (available from 2023+)
    bat_tracking_cols = ['bat_speed', 'swing_length', 'squared_up']
    available_cols = [col for col in bat_tracking_cols if col in data.columns]
    
    if available_cols:
        return data.dropna(subset=available_cols)
    else:
        print("Bat tracking data not available for this date range")
        return data


def get_fielding_metrics(year=2025):
    """Get advanced fielding metrics including OAA"""
    print(f"Fetching fielding metrics for {year}...")
    if not HAS_FIELDING:
        print("Fielding metrics not available in this pybaseball version")
        return None
    try:
        return statcast_fielding(year)
    except Exception as e:
        print(f"Error getting fielding data: {e}")
        return None


def get_sprint_speed_data(year=2025):
    """Get sprint speed and baserunning metrics"""
    print(f"Fetching sprint speed data for {year}...")
    try:
        return statcast_sprint_speed(year)
    except Exception as e:
        print(f"Error getting sprint speed data: {e}")
        return None


def get_baserunning_metrics(year=2025):
    """Get comprehensive baserunning metrics"""
    print(f"Fetching baserunning metrics for {year}...")
    if not HAS_RUNNING:
        print("Baserunning metrics not available in this pybaseball version")
        return None
    try:
        return statcast_running(year)
    except Exception as e:
        print(f"Error getting baserunning data: {e}")
        return None


def get_fangraphs_hitting_data(year=2025, qual=100):
    """Get FanGraphs hitting data with advanced metrics"""
    print(f"Fetching FanGraphs hitting data for {year}...")
    try:
        return fg_batting_data(year, year, qual=qual)
    except Exception as e:
        print(f"Error getting FanGraphs hitting data: {e}")
        return None


def get_fangraphs_pitching_data(year=2025, qual=20):
    """Get FanGraphs pitching data with advanced metrics"""
    print(f"Fetching FanGraphs pitching data for {year}...")
    try:
        return fg_pitching_data(year, year, qual=qual)
    except Exception as e:
        print(f"Error getting FanGraphs pitching data: {e}")
        return None


def get_pitcher_statcast_data(player_id, days=30):
    """Get Statcast data for a specific pitcher"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        return statcast_pitcher(
            start_dt=start_date.strftime('%Y-%m-%d'),
            end_dt=end_date.strftime('%Y-%m-%d'),
            player_id=player_id
        )
    except Exception as e:
        print(f"Error getting pitcher Statcast data: {e}")
        return None


def get_comprehensive_player_data(player_id, year=2024):
    """Get comprehensive modern data for a single player"""
    print(f"Fetching comprehensive data for player {player_id} in {year}...")
    
    # Get basic Statcast data
    start_date = f"{year}-04-01"
    end_date = f"{year}-10-31"
    
    try:
        batter_data = statcast_batter(
            start_dt=start_date,
            end_dt=end_date,
            player_id=player_id
        )
        
        pitcher_data = statcast_pitcher(
            start_dt=start_date,
            end_dt=end_date,
            player_id=player_id
        )
        
        return {
            'batting': batter_data,
            'pitching': pitcher_data
        }
    except Exception as e:
        print(f"Error getting comprehensive player data: {e}")
        return None


def get_situational_data(start_date, end_date, situation_filter=None):
    """Get Statcast data filtered by specific situations"""
    print(f"Fetching situational data from {start_date} to {end_date}...")
    
    data = statcast(start_dt=start_date, end_dt=end_date)
    
    if situation_filter:
        # Apply situational filters
        if 'leverage' in situation_filter:
            # Filter by leverage index if available
            pass
        if 'count' in situation_filter:
            # Filter by specific counts
            pass
        if 'inning' in situation_filter:
            # Filter by specific innings
            pass
    
    return data