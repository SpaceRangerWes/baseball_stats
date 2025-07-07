"""Data extraction module for baseball statistics.
Handles all pybaseball data fetching organized by domain classes."""

from pybaseball import (
    batting_stats,
    pitching_stats,
    statcast,
    playerid_lookup,
    statcast_pitcher,
    statcast_batter,
    standings,
    team_batting,
    team_pitching,
    fg_batting_data,
    fg_pitching_data,
    statcast_sprint_speed,
)
from pybaseball import statcast_fielding
from pybaseball import statcast_running

import pybaseball.cache
from datetime import datetime, timedelta
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Optional, Any, List, Tuple

pybaseball.cache.enable()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CacheManager:
    """Manages caching for baseball statistics data."""

    def __init__(
        self,
        cache_dir: str = ".baseball_stats_cache",
        current_year_expire_hours: int = 1,
        historical_expire_hours: Optional[int] = None,
    ):
        """
        Initialize cache manager.

        Args:
            cache_dir: Directory to store cache files
            current_year_expire_hours: Hours before current year data expires
            historical_expire_hours: Hours before historical data expires (None = never)
        """
        self.cache_dir = Path(cache_dir)
        self.current_year_expire_hours = current_year_expire_hours
        self.historical_expire_hours = historical_expire_hours
        self.current_year = datetime.now().year

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        logger.debug(f"Cache directory initialized: {self.cache_dir}")

    def _generate_cache_key(self, data_type: str, **kwargs) -> str:
        """Generate a descriptive cache key from parameters."""
        # Sort kwargs for consistent hashing
        sorted_params = sorted(kwargs.items())
        param_str = "_".join([f"{k}_{v}" for k, v in sorted_params])

        # Create hash of parameters to avoid extremely long filenames
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]

        # Create descriptive filename
        cache_key = f"{data_type}_{param_str}_{param_hash}.pkl"

        # Replace problematic characters for filesystem
        cache_key = cache_key.replace(":", "_").replace("/", "_").replace(" ", "_")

        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get full path to cache file."""
        return self.cache_dir / cache_key

    def _get_metadata_path(self, cache_key: str) -> Path:
        """Get full path to metadata file."""
        return self.cache_dir / f"{cache_key}.meta"

    def _is_cache_expired(self, cache_path: Path, year: Optional[int] = None) -> bool:
        """Check if cache file is expired."""
        if not cache_path.exists():
            return True

        file_time = datetime.fromtimestamp(cache_path.stat().st_mtime)
        current_time = datetime.now()

        # Determine expiration time based on year
        if year == self.current_year:
            expire_hours: Optional[int] = self.current_year_expire_hours
        else:
            expire_hours = self.historical_expire_hours

        if expire_hours is None:
            # Never expires
            logger.debug(f"Cache file {cache_path} never expires")
            return False

        time_diff = current_time - file_time
        is_expired = time_diff.total_seconds() > (expire_hours * 3600)

        logger.debug(
            f"Cache file {cache_path} - Age: {time_diff}, Expired: {is_expired}"
        )
        return is_expired

    def get_cached_data(
        self, data_type: str, year: Optional[int] = None, **kwargs
    ) -> Optional[Any]:
        """Retrieve cached data if available and not expired."""
        cache_key = self._generate_cache_key(data_type, year=year, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        if self._is_cache_expired(cache_path, year):
            logger.debug(f"Cache miss/expired for {cache_key}")
            return None

        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit for {cache_key}")
            return data
        except Exception as e:
            logger.debug(f"Error loading cache {cache_key}: {e}")
            return None

    def save_cached_data(
        self, data: Any, data_type: str, year: Optional[int] = None, **kwargs
    ) -> None:
        """Save data to cache."""
        cache_key = self._generate_cache_key(data_type, year=year, **kwargs)
        cache_path = self._get_cache_path(cache_key)

        # Remove old cache file if it exists
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Removed old cache file: {cache_path}")

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(data, f)

            # Save metadata
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "parameters": kwargs,
                "year": year,
                "data_type": data_type,
            }
            metadata_path = self._get_metadata_path(cache_key)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.debug(f"Saved cache for {cache_key}")
        except Exception as e:
            logger.error(f"Error saving cache {cache_key}: {e}")

    def clear_cache(self, pattern: Optional[str] = None) -> None:
        """Clear cache files matching pattern or all if pattern is None."""
        if pattern:
            files_to_remove = list(self.cache_dir.glob(f"*{pattern}*"))
        else:
            files_to_remove = list(self.cache_dir.glob("*"))

        for file_path in files_to_remove:
            file_path.unlink()
            logger.debug(f"Removed cache file: {file_path}")

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime object."""
        return datetime.strptime(date_str, "%Y-%m-%d")

    def _dates_overlap(self, start1: str, end1: str, start2: str, end2: str) -> bool:
        """Check if two date ranges overlap."""
        s1, e1 = self._parse_date(start1), self._parse_date(end1)
        s2, e2 = self._parse_date(start2), self._parse_date(end2)
        return s1 <= e2 and s2 <= e1

    def _merge_date_ranges(
        self, start1: str, end1: str, start2: str, end2: str
    ) -> Tuple[str, str]:
        """Merge two overlapping date ranges."""
        s1, e1 = self._parse_date(start1), self._parse_date(end1)
        s2, e2 = self._parse_date(start2), self._parse_date(end2)
        merged_start = min(s1, s2).strftime("%Y-%m-%d")
        merged_end = max(e1, e2).strftime("%Y-%m-%d")
        return merged_start, merged_end

    def find_overlapping_statcast_caches(
        self, data_type: str, start_date: str, end_date: str, **kwargs
    ) -> List[Tuple[str, Any]]:
        """Find cached Statcast data that overlaps with the requested date range."""
        overlapping_caches = []

        # Look for cache files with same data type and kwargs
        pattern = f"{data_type}_*"
        cache_files = list(self.cache_dir.glob(pattern))

        for cache_file in cache_files:
            if cache_file.suffix != ".pkl" or cache_file.name.endswith(".meta"):
                continue

            try:
                # Load metadata to check parameters and dates
                metadata_path = self._get_metadata_path(cache_file.name)
                if not metadata_path.exists():
                    continue

                with open(metadata_path, "rb") as f:
                    metadata = pickle.load(f)

                # Check if parameters match (excluding dates)
                cache_params = metadata["parameters"].copy()
                cache_start = cache_params.pop("start_date", None)
                cache_end = cache_params.pop("end_date", None)

                if not cache_start or not cache_end:
                    continue

                # Compare parameters (excluding dates)
                request_params = kwargs.copy()
                request_params.pop("start_date", None)
                request_params.pop("end_date", None)

                if cache_params != request_params:
                    continue

                # Check if date ranges overlap
                if self._dates_overlap(start_date, end_date, cache_start, cache_end):
                    # Load the cached data
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                    overlapping_caches.append(
                        (f"{cache_start}_{cache_end}", cached_data)
                    )
                    logger.debug(
                        f"Found overlapping cache: {cache_start} to {cache_end}"
                    )

            except Exception as e:
                logger.debug(f"Error checking cache file {cache_file}: {e}")
                continue

        return overlapping_caches


# Global cache manager instance
cache_manager = CacheManager()


class HittingData:
    """Handles all batting/hitting data extraction and processing."""

    @staticmethod
    def get_raw_stats(year=2025):
        """Fetch raw hitting data from pybaseball"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("hitting_raw", year=year)
        if cached_data is not None:
            return cached_data

        print(f"Fetching {year} hitting data...")
        data = batting_stats(year)

        # Cache the data
        cache_manager.save_cached_data(data, "hitting_raw", year=year)
        return data

    @staticmethod
    def get_qualified_hitters(year=2025, min_pa=100):
        """Get qualified hitters with minimum plate appearances"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "hitting_qualified", year=year, min_pa=min_pa
        )
        if cached_data is not None:
            return cached_data

        batting_data = HittingData.get_raw_stats(year)
        qualified_data = batting_data[batting_data["PA"] >= min_pa].copy()

        # Cache the data
        cache_manager.save_cached_data(
            qualified_data, "hitting_qualified", year=year, min_pa=min_pa
        )
        return qualified_data

    @staticmethod
    def get_fangraphs_data(year=2025, qual=100):
        """Get FanGraphs hitting data with advanced metrics"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "fangraphs_hitting", year=year, qual=qual
        )
        if cached_data is not None:
            return cached_data

        print(f"Fetching FanGraphs hitting data for {year}...")
        try:
            data = fg_batting_data(year, year, qual=qual)
            # Cache the data
            cache_manager.save_cached_data(
                data, "fangraphs_hitting", year=year, qual=qual
            )
            return data
        except Exception as e:
            print(f"Error getting FanGraphs hitting data: {e}")
            return None

    @staticmethod
    def get_bat_tracking_data(start_date, end_date):
        """Get bat tracking data (available from 2023+)"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "bat_tracking", start_date=start_date, end_date=end_date
        )
        if cached_data is not None:
            return cached_data

        print(f"Fetching bat tracking data from {start_date} to {end_date}...")
        data = statcast(start_dt=start_date, end_dt=end_date)

        # Filter for bat tracking columns (available from 2023+)
        bat_tracking_cols = ["bat_speed", "swing_length", "squared_up"]
        available_cols = [col for col in bat_tracking_cols if col in data.columns]

        if available_cols:
            filtered_data = data.dropna(subset=available_cols)
        else:
            print("Bat tracking data not available for this date range")
            filtered_data = data

        # Cache the data
        cache_manager.save_cached_data(
            filtered_data, "bat_tracking", start_date=start_date, end_date=end_date
        )
        return filtered_data

    @staticmethod
    def get_sprint_speed_data(year=2025):
        """Get sprint speed and baserunning metrics"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("sprint_speed", year=year)
        if cached_data is not None:
            return cached_data

        print(f"Fetching sprint speed data for {year}...")
        try:
            data = statcast_sprint_speed(year)
            # Cache the data
            cache_manager.save_cached_data(data, "sprint_speed", year=year)
            return data
        except Exception as e:
            print(f"Error getting sprint speed data: {e}")
            return None

    @staticmethod
    def get_baserunning_metrics(year=2025):
        """Get comprehensive baserunning metrics"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("baserunning", year=year)
        if cached_data is not None:
            return cached_data

        print(f"Fetching baserunning metrics for {year}...")
        try:
            data = statcast_running(year)
            # Cache the data
            cache_manager.save_cached_data(data, "baserunning", year=year)
            return data
        except Exception as e:
            print(f"Error getting baserunning data: {e}")
            return None


class PitchingData:
    """Handles all pitching data extraction and processing."""

    @staticmethod
    def get_raw_stats(year=2025):
        """Fetch raw pitching data from pybaseball"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("pitching_raw", year=year)
        if cached_data is not None:
            return cached_data

        print(f"Fetching {year} pitching data...")
        data = pitching_stats(year)

        # Cache the data
        cache_manager.save_cached_data(data, "pitching_raw", year=year)
        return data

    @staticmethod
    def get_qualified_pitchers(year=2025, min_ip=20):
        """Get qualified pitchers with minimum innings pitched"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "pitching_qualified", year=year, min_ip=min_ip
        )
        if cached_data is not None:
            return cached_data

        pitching_data = PitchingData.get_raw_stats(year)
        qualified_data = pitching_data[pitching_data["IP"] >= min_ip].copy()

        # Cache the data
        cache_manager.save_cached_data(
            qualified_data, "pitching_qualified", year=year, min_ip=min_ip
        )
        return qualified_data

    @staticmethod
    def get_fangraphs_data(year=2025, qual=20):
        """Get FanGraphs pitching data with advanced metrics"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "fangraphs_pitching", year=year, qual=qual
        )
        if cached_data is not None:
            return cached_data

        print(f"Fetching FanGraphs pitching data for {year}...")
        try:
            data = fg_pitching_data(year, year, qual=qual)
            # Cache the data
            cache_manager.save_cached_data(
                data, "fangraphs_pitching", year=year, qual=qual
            )
            return data
        except Exception as e:
            print(f"Error getting FanGraphs pitching data: {e}")
            return None

    @staticmethod
    def get_fielding_metrics(year=2025):
        """Get advanced fielding metrics including OAA"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("fielding", year=year)
        if cached_data is not None:
            return cached_data

        print(f"Fetching fielding metrics for {year}...")
        try:
            data = statcast_fielding(year)
            # Cache the data
            cache_manager.save_cached_data(data, "fielding", year=year)
            return data
        except Exception as e:
            print(f"Error getting fielding data: {e}")
            return None


class TeamData:
    """Handles team-level statistics and standings."""

    @staticmethod
    def get_team_stats(year=2025):
        """Get team-level batting and pitching stats"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("team_stats", year=year)
        if cached_data is not None:
            return cached_data

        data = {"batting": team_batting(year), "pitching": team_pitching(year)}

        # Cache the data
        cache_manager.save_cached_data(data, "team_stats", year=year)
        return data

    @staticmethod
    def get_standings(year=2025):
        """Get league standings"""
        # Check cache first
        cached_data = cache_manager.get_cached_data("standings", year=year)
        if cached_data is not None:
            return cached_data

        data = standings(year)

        # Cache the data
        cache_manager.save_cached_data(data, "standings", year=year)
        return data


class PlayerData:
    """Handles individual player data and lookups."""

    @staticmethod
    def get_player_lookup(first_name, last_name):
        """Look up player ID for Statcast data"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "player_lookup", first_name=first_name, last_name=last_name
        )
        if cached_data is not None:
            return cached_data

        data = playerid_lookup(last_name, first_name)

        # Cache the data
        cache_manager.save_cached_data(
            data, "player_lookup", first_name=first_name, last_name=last_name
        )
        return data

    @staticmethod
    def get_batter_statcast_data(player_id, days=30):
        """Get Statcast data for a specific batter"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "batter_statcast", player_id=player_id, days=days
        )
        if cached_data is not None:
            return cached_data

        try:
            data = statcast_batter(
                start_dt=start_date.strftime("%Y-%m-%d"),
                end_dt=end_date.strftime("%Y-%m-%d"),
                player_id=player_id,
            )

            # Cache the data
            cache_manager.save_cached_data(
                data, "batter_statcast", player_id=player_id, days=days
            )
            return data
        except Exception as e:
            print(f"Error getting batter Statcast data: {e}")
            return None

    @staticmethod
    def get_pitcher_statcast_data(player_id, days=30):
        """Get Statcast data for a specific pitcher"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "pitcher_statcast", player_id=player_id, days=days
        )
        if cached_data is not None:
            return cached_data

        try:
            data = statcast_pitcher(
                start_dt=start_date.strftime("%Y-%m-%d"),
                end_dt=end_date.strftime("%Y-%m-%d"),
                player_id=player_id,
            )

            # Cache the data
            cache_manager.save_cached_data(
                data, "pitcher_statcast", player_id=player_id, days=days
            )
            return data
        except Exception as e:
            print(f"Error getting pitcher Statcast data: {e}")
            return None

    @staticmethod
    def get_comprehensive_data(player_id, year=2024):
        """Get comprehensive modern data for a single player"""
        # Check cache first
        cached_data = cache_manager.get_cached_data(
            "comprehensive_player", player_id=player_id, year=year
        )
        if cached_data is not None:
            return cached_data

        print(f"Fetching comprehensive data for player {player_id} in {year}...")

        # Get basic Statcast data
        start_date = f"{year}-04-01"
        end_date = f"{year}-10-31"

        try:
            batter_data = statcast_batter(
                start_dt=start_date, end_dt=end_date, player_id=player_id
            )

            pitcher_data = statcast_pitcher(
                start_dt=start_date, end_dt=end_date, player_id=player_id
            )

            data = {"batting": batter_data, "pitching": pitcher_data}

            # Cache the data
            cache_manager.save_cached_data(
                data, "comprehensive_player", player_id=player_id, year=year
            )
            return data
        except Exception as e:
            print(f"Error getting comprehensive player data: {e}")
            return None


class StatcastData:
    """Handles general Statcast data extraction."""

    @staticmethod
    def get_modern_data(start_date, end_date):
        """Get comprehensive Statcast data including modern metrics"""
        # Check for exact cache match first
        cached_data = cache_manager.get_cached_data(
            "statcast_modern", start_date=start_date, end_date=end_date
        )
        if cached_data is not None:
            return cached_data

        # Check for overlapping cached data
        overlapping_caches = cache_manager.find_overlapping_statcast_caches(
            "statcast_modern", start_date, end_date
        )

        if overlapping_caches:
            logger.debug(
                f"Found {len(overlapping_caches)} overlapping caches, attempting to merge"
            )
            # For simplicity, if we have overlapping data, just use the first one
            # that covers our range
            # In a more sophisticated implementation, we could merge the data
            for date_range, data in overlapping_caches:
                cache_start, cache_end = date_range.split("_")
                if cache_manager._parse_date(cache_start) <= cache_manager._parse_date(
                    start_date
                ) and cache_manager._parse_date(cache_end) >= cache_manager._parse_date(
                    end_date
                ):
                    logger.debug(
                        "Using overlapping cache that fully covers requested range"
                    )
                    # Filter data to requested date range
                    if hasattr(data, "game_date"):
                        filtered_data = data[
                            (data["game_date"] >= start_date)
                            & (data["game_date"] <= end_date)
                        ].copy()
                        return filtered_data
                    return data

        print(f"Fetching Statcast data from {start_date} to {end_date}...")
        data = statcast(start_dt=start_date, end_dt=end_date)

        # Cache the data
        cache_manager.save_cached_data(
            data, "statcast_modern", start_date=start_date, end_date=end_date
        )
        return data

    @staticmethod
    def get_situational_data(start_date, end_date, situation_filter=None):
        """Get Statcast data filtered by specific situations"""
        # Check cache first
        filter_key = str(situation_filter) if situation_filter else "no_filter"
        cached_data = cache_manager.get_cached_data(
            "statcast_situational",
            start_date=start_date,
            end_date=end_date,
            filter=filter_key,
        )
        if cached_data is not None:
            return cached_data

        print(f"Fetching situational data from {start_date} to {end_date}...")

        data = statcast(start_dt=start_date, end_dt=end_date)

        if situation_filter:
            # Apply situational filters
            if "leverage" in situation_filter:
                # Filter by leverage index if available
                pass
            if "count" in situation_filter:
                # Filter by specific counts
                pass
            if "inning" in situation_filter:
                # Filter by specific innings
                pass

        # Cache the data
        cache_manager.save_cached_data(
            data,
            "statcast_situational",
            start_date=start_date,
            end_date=end_date,
            filter=filter_key,
        )
        return data


# Backward compatibility - maintain original function names
def get_raw_hitting_stats(year=2025):
    """Fetch raw hitting data from pybaseball"""
    return HittingData.get_raw_stats(year)


def get_raw_pitching_stats(year=2025):
    """Fetch raw pitching data from pybaseball"""
    return PitchingData.get_raw_stats(year)


def get_qualified_hitters(year=2025, min_pa=100):
    """Get qualified hitters with minimum plate appearances"""
    return HittingData.get_qualified_hitters(year, min_pa)


def get_qualified_pitchers(year=2025, min_ip=20):
    """Get qualified pitchers with minimum innings pitched"""
    return PitchingData.get_qualified_pitchers(year, min_ip)


def get_player_lookup(first_name, last_name):
    """Look up player ID for Statcast data"""
    return PlayerData.get_player_lookup(first_name, last_name)


def get_player_statcast_data(player_id, days=30):
    """Get Statcast data for a specific player"""
    return PlayerData.get_batter_statcast_data(player_id, days)


def get_team_stats(year=2025):
    """Get team-level batting and pitching stats"""
    return TeamData.get_team_stats(year)


def get_standings(year=2025):
    """Get league standings"""
    return TeamData.get_standings(year)


def get_modern_statcast_data(start_date, end_date):
    """Get comprehensive Statcast data including modern metrics"""
    return StatcastData.get_modern_data(start_date, end_date)


def get_bat_tracking_data(start_date, end_date):
    """Get bat tracking data (available from 2023+)"""
    return HittingData.get_bat_tracking_data(start_date, end_date)


def get_fielding_metrics(year=2025):
    """Get advanced fielding metrics including OAA"""
    return PitchingData.get_fielding_metrics(year)


def get_sprint_speed_data(year=2025):
    """Get sprint speed and baserunning metrics"""
    return HittingData.get_sprint_speed_data(year)


def get_baserunning_metrics(year=2025):
    """Get comprehensive baserunning metrics"""
    return HittingData.get_baserunning_metrics(year)


def get_fangraphs_hitting_data(year=2025, qual=100):
    """Get FanGraphs hitting data with advanced metrics"""
    return HittingData.get_fangraphs_data(year, qual)


def get_fangraphs_pitching_data(year=2025, qual=20):
    """Get FanGraphs pitching data with advanced metrics"""
    return PitchingData.get_fangraphs_data(year, qual)


def get_pitcher_statcast_data(player_id, days=30):
    """Get Statcast data for a specific pitcher"""
    return PlayerData.get_pitcher_statcast_data(player_id, days)


def get_comprehensive_player_data(player_id, year=2024):
    """Get comprehensive modern data for a single player"""
    return PlayerData.get_comprehensive_data(player_id, year)


def get_situational_data(start_date, end_date, situation_filter=None):
    """Get Statcast data filtered by specific situations"""
    return StatcastData.get_situational_data(start_date, end_date, situation_filter)
