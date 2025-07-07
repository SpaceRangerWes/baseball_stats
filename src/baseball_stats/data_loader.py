"""Unified data loader for baseball statistics.
Provides a single interface to load and access all baseball data.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

from .data import (
    HittingData,
    PitchingData,
    TeamData,
    PlayerData,
    StatcastData,
    cache_manager,
)

logger = logging.getLogger(__name__)


class BaseballDataLoader:
    """Unified loader for all baseball statistics data."""

    def __init__(
        self,
        year: int = 2025,
        min_pa: int = 100,
        min_ip: int = 20,
        statcast_days: int = 30,
    ):
        """
        Initialize the data loader with default parameters.

        Args:
            year: The year to load data for
            min_pa: Minimum plate appearances for qualified hitters
            min_ip: Minimum innings pitched for qualified pitchers
            statcast_days: Number of days to look back for Statcast data
        """
        self.year = year
        self.min_pa = min_pa
        self.min_ip = min_ip
        self.statcast_days = statcast_days

        # Data storage
        self._hitting_data: Dict[str, pd.DataFrame] = {}
        self._pitching_data: Dict[str, pd.DataFrame] = {}
        self._team_data: Dict[str, Any] = {}
        self._statcast_data: Dict[str, pd.DataFrame] = {}
        self._fielding_data: Optional[pd.DataFrame] = None
        self._is_loaded = False

        logger.debug(f"Initialized BaseballDataLoader for year {year}")

    def load_all(self, verbose: bool = True) -> None:
        """
        Load all available baseball data.

        Args:
            verbose: Whether to print loading progress
        """
        if verbose:
            print(f"🚀 Loading all baseball data for {self.year}...")

        # Load hitting data
        self._load_hitting_data(verbose)

        # Load pitching data
        self._load_pitching_data(verbose)

        # Load team data
        self._load_team_data(verbose)

        # Load Statcast data
        self._load_statcast_data(verbose)

        # Load fielding data
        self._load_fielding_data(verbose)

        self._is_loaded = True

        if verbose:
            print("✅ All data loaded successfully!")
            self._print_data_summary()

    def _load_hitting_data(self, verbose: bool) -> None:
        """Load all hitting-related data."""
        if verbose:
            print("📊 Loading hitting data...")

        try:
            # Raw hitting stats
            self._hitting_data["raw"] = HittingData.get_raw_stats(self.year)
            logger.debug(f"Loaded {len(self._hitting_data['raw'])} raw hitting records")

            # Qualified hitters
            self._hitting_data["qualified"] = HittingData.get_qualified_hitters(
                self.year, self.min_pa
            )
            logger.debug(
                f"Loaded {len(self._hitting_data['qualified'])} qualified hitters"
            )

            # FanGraphs data
            fg_data = HittingData.get_fangraphs_data(self.year, self.min_pa)
            if fg_data is not None:
                self._hitting_data["fangraphs"] = fg_data
                logger.debug(f"Loaded {len(fg_data)} FanGraphs hitting records")

            # Sprint speed data
            sprint_data = HittingData.get_sprint_speed_data(self.year)
            if sprint_data is not None:
                self._hitting_data["sprint_speed"] = sprint_data
                logger.debug(f"Loaded {len(sprint_data)} sprint speed records")

            # Baserunning metrics
            baserunning_data = HittingData.get_baserunning_metrics(self.year)
            if baserunning_data is not None:
                self._hitting_data["baserunning"] = baserunning_data
                logger.debug(f"Loaded {len(baserunning_data)} baserunning records")

        except Exception as e:
            logger.error(f"Error loading hitting data: {e}")
            if verbose:
                print(f"⚠️ Error loading some hitting data: {e}")

    def _load_pitching_data(self, verbose: bool) -> None:
        """Load all pitching-related data."""
        if verbose:
            print("⚾ Loading pitching data...")

        try:
            # Raw pitching stats
            self._pitching_data["raw"] = PitchingData.get_raw_stats(self.year)
            logger.debug(
                f"Loaded {len(self._pitching_data['raw'])} raw pitching records"
            )

            # Qualified pitchers
            self._pitching_data["qualified"] = PitchingData.get_qualified_pitchers(
                self.year, self.min_ip
            )
            logger.debug(
                f"Loaded {len(self._pitching_data['qualified'])} qualified pitchers"
            )

            # FanGraphs data
            fg_data = PitchingData.get_fangraphs_data(self.year, self.min_ip)
            if fg_data is not None:
                self._pitching_data["fangraphs"] = fg_data
                logger.debug(f"Loaded {len(fg_data)} FanGraphs pitching records")

        except Exception as e:
            logger.error(f"Error loading pitching data: {e}")
            if verbose:
                print(f"⚠️ Error loading some pitching data: {e}")

    def _load_team_data(self, verbose: bool) -> None:
        """Load all team-related data."""
        if verbose:
            print("🏟️ Loading team data...")

        try:
            # Team stats
            team_stats = TeamData.get_team_stats(self.year)
            if team_stats is not None:
                self._team_data["stats"] = team_stats
                logger.debug("Loaded team batting and pitching stats")

            # Standings
            standings = TeamData.get_standings(self.year)
            if standings is not None:
                self._team_data["standings"] = standings
                logger.debug("Loaded team standings")

        except Exception as e:
            logger.error(f"Error loading team data: {e}")
            if verbose:
                print(f"⚠️ Error loading some team data: {e}")

    def _load_statcast_data(self, verbose: bool) -> None:
        """Load Statcast data for recent games."""
        if verbose:
            print("📡 Loading Statcast data...")

        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.statcast_days)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            # Modern Statcast data
            statcast_data = StatcastData.get_modern_data(start_str, end_str)
            if statcast_data is not None and len(statcast_data) > 0:
                self._statcast_data["recent"] = statcast_data
                logger.debug(f"Loaded {len(statcast_data)} recent Statcast records")

            # Bat tracking data (2023+)
            if self.year >= 2023:
                bat_tracking = HittingData.get_bat_tracking_data(start_str, end_str)
                if bat_tracking is not None and len(bat_tracking) > 0:
                    self._statcast_data["bat_tracking"] = bat_tracking
                    logger.debug(f"Loaded {len(bat_tracking)} bat tracking records")

        except Exception as e:
            logger.error(f"Error loading Statcast data: {e}")
            if verbose:
                print(f"⚠️ Error loading some Statcast data: {e}")

    def _load_fielding_data(self, verbose: bool) -> None:
        """Load fielding data."""
        if verbose:
            print("🧤 Loading fielding data...")

        try:
            fielding_data = PitchingData.get_fielding_metrics(self.year)
            if fielding_data is not None:
                self._fielding_data = fielding_data
                logger.debug(f"Loaded {len(fielding_data)} fielding records")

        except Exception as e:
            logger.error(f"Error loading fielding data: {e}")
            if verbose:
                print(f"⚠️ Error loading fielding data: {e}")

    def _print_data_summary(self) -> None:
        """Print a summary of loaded data."""
        print("\n📊 Data Summary:")
        print(f"Year: {self.year}")

        if self._hitting_data:
            print("\nHitting Data:")
            for key, df in self._hitting_data.items():
                if isinstance(df, pd.DataFrame):
                    print(f"  - {key}: {len(df)} records")

        if self._pitching_data:
            print("\nPitching Data:")
            for key, df in self._pitching_data.items():
                if isinstance(df, pd.DataFrame):
                    print(f"  - {key}: {len(df)} records")

        if self._team_data:
            print("\nTeam Data:")
            for key in self._team_data:
                print(f"  - {key}: loaded")

        if self._statcast_data:
            print("\nStatcast Data:")
            for key, df in self._statcast_data.items():
                if isinstance(df, pd.DataFrame):
                    print(f"  - {key}: {len(df)} records")

        if self._fielding_data is not None:
            print(f"\nFielding Data: {len(self._fielding_data)} records")

    # Hitting data getters
    @property
    def hitting_raw(self) -> pd.DataFrame:
        """Get raw hitting statistics."""
        if "raw" not in self._hitting_data:
            self._hitting_data["raw"] = HittingData.get_raw_stats(self.year)
        return self._hitting_data["raw"]

    @property
    def hitting_qualified(self) -> pd.DataFrame:
        """Get qualified hitters with minimum plate appearances."""
        if "qualified" not in self._hitting_data:
            self._hitting_data["qualified"] = HittingData.get_qualified_hitters(
                self.year, self.min_pa
            )
        return self._hitting_data["qualified"]

    @property
    def hitting_fangraphs(self) -> Optional[pd.DataFrame]:
        """Get FanGraphs hitting data with advanced metrics."""
        if "fangraphs" not in self._hitting_data:
            self._hitting_data["fangraphs"] = HittingData.get_fangraphs_data(
                self.year, self.min_pa
            )
        return self._hitting_data.get("fangraphs")

    @property
    def sprint_speed(self) -> Optional[pd.DataFrame]:
        """Get sprint speed data."""
        if "sprint_speed" not in self._hitting_data:
            self._hitting_data["sprint_speed"] = HittingData.get_sprint_speed_data(
                self.year
            )
        return self._hitting_data.get("sprint_speed")

    @property
    def baserunning(self) -> Optional[pd.DataFrame]:
        """Get baserunning metrics."""
        if "baserunning" not in self._hitting_data:
            self._hitting_data["baserunning"] = HittingData.get_baserunning_metrics(
                self.year
            )
        return self._hitting_data.get("baserunning")

    # Pitching data getters
    @property
    def pitching_raw(self) -> pd.DataFrame:
        """Get raw pitching statistics."""
        if "raw" not in self._pitching_data:
            self._pitching_data["raw"] = PitchingData.get_raw_stats(self.year)
        return self._pitching_data["raw"]

    @property
    def pitching_qualified(self) -> pd.DataFrame:
        """Get qualified pitchers with minimum innings pitched."""
        if "qualified" not in self._pitching_data:
            self._pitching_data["qualified"] = PitchingData.get_qualified_pitchers(
                self.year, self.min_ip
            )
        return self._pitching_data["qualified"]

    @property
    def pitching_fangraphs(self) -> Optional[pd.DataFrame]:
        """Get FanGraphs pitching data with advanced metrics."""
        if "fangraphs" not in self._pitching_data:
            self._pitching_data["fangraphs"] = PitchingData.get_fangraphs_data(
                self.year, self.min_ip
            )
        return self._pitching_data.get("fangraphs")

    # Team data getters
    @property
    def team_batting(self) -> Optional[pd.DataFrame]:
        """Get team batting statistics."""
        if "stats" not in self._team_data:
            self._team_data["stats"] = TeamData.get_team_stats(self.year)
        return self._team_data.get("stats", {}).get("batting")

    @property
    def team_pitching(self) -> Optional[pd.DataFrame]:
        """Get team pitching statistics."""
        if "stats" not in self._team_data:
            self._team_data["stats"] = TeamData.get_team_stats(self.year)
        return self._team_data.get("stats", {}).get("pitching")

    @property
    def standings(self) -> Optional[pd.DataFrame]:
        """Get team standings."""
        if "standings" not in self._team_data:
            self._team_data["standings"] = TeamData.get_standings(self.year)
        return self._team_data.get("standings")

    # Statcast data getters
    @property
    def statcast_recent(self) -> Optional[pd.DataFrame]:
        """Get recent Statcast data."""
        if "recent" not in self._statcast_data:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.statcast_days)
            self._statcast_data["recent"] = StatcastData.get_modern_data(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
        return self._statcast_data.get("recent")

    @property
    def bat_tracking(self) -> Optional[pd.DataFrame]:
        """Get bat tracking data (2023+)."""
        if self.year < 2023:
            logger.warning("Bat tracking data only available from 2023 onwards")
            return None

        if "bat_tracking" not in self._statcast_data:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.statcast_days)
            self._statcast_data["bat_tracking"] = HittingData.get_bat_tracking_data(
                start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
            )
        return self._statcast_data.get("bat_tracking")

    # Fielding data getter
    @property
    def fielding(self) -> Optional[pd.DataFrame]:
        """Get fielding metrics including OAA."""
        if self._fielding_data is None:
            self._fielding_data = PitchingData.get_fielding_metrics(self.year)
        return self._fielding_data

    # Utility methods
    def refresh_all(self) -> None:
        """Clear all cached data and reload."""
        logger.info("Refreshing all data...")

        # Clear internal storage
        self._hitting_data.clear()
        self._pitching_data.clear()
        self._team_data.clear()
        self._statcast_data.clear()
        self._fielding_data = None
        self._is_loaded = False

        # Optionally clear disk cache for current year
        if self.year == datetime.now().year:
            cache_manager.clear_cache(pattern=f"*year_{self.year}*")

        # Reload all data
        self.load_all()

    def get_player_data(
        self, first_name: str, last_name: str, statcast_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive data for a specific player.

        Args:
            first_name: Player's first name
            last_name: Player's last name
            statcast_days: Number of days to look back for Statcast data

        Returns:
            Dictionary containing player lookup, batting, and pitching data
        """
        if statcast_days is None:
            statcast_days = self.statcast_days

        # Look up player ID
        player_lookup = PlayerData.get_player_lookup(first_name, last_name)

        if player_lookup.empty:
            logger.warning(f"No player found for {first_name} {last_name}")
            return {}

        # Get the most recent player ID
        player_id = player_lookup.iloc[0]["key_mlbam"]

        # Get player data
        result = {"lookup": player_lookup, "player_id": player_id}

        # Get batting data
        batting_data = PlayerData.get_batter_statcast_data(player_id, statcast_days)
        if batting_data is not None and not batting_data.empty:
            result["batting"] = batting_data

        # Get pitching data
        pitching_data = PlayerData.get_pitcher_statcast_data(player_id, statcast_days)
        if pitching_data is not None and not pitching_data.empty:
            result["pitching"] = pitching_data

        return result

    def get_all_data_dict(self) -> Dict[str, Any]:
        """
        Get all loaded data as a dictionary.

        Returns:
            Dictionary containing all loaded data
        """
        return {
            "hitting": self._hitting_data,
            "pitching": self._pitching_data,
            "team": self._team_data,
            "statcast": self._statcast_data,
            "fielding": self._fielding_data,
            "year": self.year,
            "is_loaded": self._is_loaded,
        }
