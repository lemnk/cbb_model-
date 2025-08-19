"""
Player Features for NCAA CBB Betting ML System.

This module handles player-level feature engineering including:
- Injury flags and availability
- Minutes distribution and bench depth
- Star player impact ratings
- Foul trouble indicators
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta


class PlayerFeatures:
    """Engineers player-level features for CBB betting analysis."""
    
    def __init__(self, db_engine):
        """Initialize the player feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        self.engine = db_engine
    
    def transform(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Player-level features:
        - Injury flags
        - Minutes distribution (bench depth)
        - Star player impact rating
        
        Args:
            games_df: DataFrame with games data
            
        Returns:
            DataFrame with player features prefixed with 'player_'
        """
        if games_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        features_df = games_df.copy()
        
        # Add player availability features
        features_df = self._add_availability_features(features_df)
        
        # Add injury and health features
        features_df = self._add_injury_features(features_df)
        
        # Add lineup and rotation features
        features_df = self._add_lineup_features(features_df)
        
        # Add foul trouble features
        features_df = self._add_foul_features(features_df)
        
        # Add bench utilization features
        features_df = self._add_bench_features(features_df)
        
        # Add star player impact features
        features_df = self._add_star_player_features(features_df)
        
        return features_df
    
    def _add_availability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic player availability features."""
        # Simulated player availability data (replace with real injury database)
        np.random.seed(42)  # For reproducible demo
        
        # Home team player availability
        df['player_home_available_count'] = np.random.randint(10, 15, len(df))
        df['player_home_injured_count'] = np.random.randint(0, 3, len(df))
        df['player_home_suspended_count'] = np.random.randint(0, 2, len(df))
        
        # Away team player availability
        df['player_away_available_count'] = np.random.randint(10, 15, len(df))
        df['player_away_injured_count'] = np.random.randint(0, 3, len(df))
        df['player_away_suspended_count'] = np.random.randint(0, 2, len(df))
        
        # Total roster counts
        df['player_home_total_roster'] = df['player_home_available_count'] + df['player_home_injured_count'] + df['player_home_suspended_count']
        df['player_away_total_roster'] = df['player_away_available_count'] + df['player_away_injured_count'] + df['player_away_suspended_count']
        
        # Availability percentages
        df['player_home_availability_pct'] = df['player_home_available_count'] / df['player_home_total_roster']
        df['player_away_availability_pct'] = df['player_away_available_count'] / df['player_away_total_roster']
        
        # Availability differential
        df['player_availability_diff'] = df['player_home_availability_pct'] - df['player_away_availability_pct']
        
        # Critical availability indicators
        df['player_home_critical_shortage'] = (df['player_home_available_count'] <= 8).astype(int)
        df['player_away_critical_shortage'] = (df['player_away_available_count'] <= 8).astype(int)
        
        return df
    
    def _add_injury_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add injury and health-related features."""
        # Injury severity indicators
        df['player_home_minor_injuries'] = np.random.randint(0, 2, len(df))
        df['player_home_major_injuries'] = np.random.randint(0, 1, len(df))
        df['player_away_minor_injuries'] = np.random.randint(0, 2, len(df))
        df['player_away_major_injuries'] = np.random.randint(0, 1, len(df))
        
        # Injury impact scores (weighted by severity)
        df['player_home_injury_impact'] = (
            df['player_home_minor_injuries'] * 0.3 + 
            df['player_home_major_injuries'] * 0.7
        )
        df['player_away_injury_impact'] = (
            df['player_away_minor_injuries'] * 0.3 + 
            df['player_away_major_injuries'] * 0.7
        )
        
        # Injury differential
        df['player_injury_impact_diff'] = df['player_home_injury_impact'] - df['player_away_injury_impact']
        
        # Return from injury indicators
        df['player_home_returning_players'] = np.random.randint(0, 2, len(df))
        df['player_away_returning_players'] = np.random.randint(0, 2, len(df))
        
        # Fresh injury indicators
        df['player_home_fresh_injuries'] = np.random.randint(0, 2, len(df))
        df['player_away_fresh_injuries'] = np.random.randint(0, 2, len(df))
        
        return df
    
    def _add_lineup_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lineup and rotation features."""
        # Starting lineup consistency
        df['player_home_starters_available'] = np.random.randint(4, 6, len(df))
        df['player_away_starters_available'] = np.random.randint(4, 6, len(df))
        
        # Starting lineup availability percentage
        df['player_home_starters_pct'] = df['player_home_starters_available'] / 5.0
        df['player_away_starters_pct'] = df['player_away_starters_available'] / 5.0
        
        # Starting lineup differential
        df['player_starters_diff'] = df['player_home_starters_pct'] - df['player_away_starters_pct']
        
        # Rotation depth indicators
        df['player_home_rotation_depth'] = np.random.randint(7, 12, len(df))
        df['player_away_rotation_depth'] = np.random.randint(7, 12, len(df))
        
        # Rotation depth differential
        df['player_rotation_depth_diff'] = df['player_home_rotation_depth'] - df['player_away_rotation_depth']
        
        # Experience level indicators
        df['player_home_experience_years'] = np.random.normal(2.5, 0.8, len(df))
        df['player_away_experience_years'] = np.random.normal(2.5, 0.8, len(df))
        
        # Experience differential
        df['player_experience_diff'] = df['player_home_experience_years'] - df['player_away_experience_years']
        
        return df
    
    def _add_foul_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add foul trouble and discipline features."""
        # Foul trouble indicators
        df['player_home_foul_trouble_count'] = np.random.randint(0, 3, len(df))
        df['player_away_foul_trouble_count'] = np.random.randint(0, 3, len(df))
        
        # Foul trouble differential
        df['player_foul_trouble_diff'] = df['player_home_foul_trouble_count'] - df['player_away_foul_trouble_count']
        
        # Players in foul trouble percentage
        df['player_home_foul_trouble_pct'] = df['player_home_foul_trouble_count'] / df['player_home_available_count']
        df['player_away_foul_trouble_pct'] = df['player_away_foul_trouble_count'] / df['player_away_available_count']
        
        # Foul trouble percentage differential
        df['player_foul_trouble_pct_diff'] = df['player_home_foul_trouble_pct'] - df['player_away_foul_trouble_pct']
        
        # Disciplinary issues
        df['player_home_technical_fouls'] = np.random.randint(0, 2, len(df))
        df['player_away_technical_fouls'] = np.random.randint(0, 2, len(df))
        
        # Technical foul differential
        df['player_technical_foul_diff'] = df['player_home_technical_fouls'] - df['player_away_technical_fouls']
        
        return df
    
    def _add_bench_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bench utilization and depth features."""
        # Bench player availability
        df['player_home_bench_available'] = df['player_home_available_count'] - df['player_home_starters_available']
        df['player_away_bench_available'] = df['player_away_available_count'] - df['player_away_starters_available']
        
        # Bench utilization percentage
        df['player_home_bench_utilization'] = df['player_home_bench_available'] / df['player_home_available_count']
        df['player_away_bench_utilization'] = df['player_away_bench_available'] / df['player_away_available_count']
        
        # Bench utilization differential
        df['player_bench_utilization_diff'] = df['player_home_bench_utilization'] - df['player_away_bench_utilization']
        
        # Bench quality indicators
        df['player_home_bench_experience'] = np.random.normal(1.8, 0.6, len(df))
        df['player_away_bench_experience'] = np.random.normal(1.8, 0.6, len(df))
        
        # Bench experience differential
        df['player_bench_experience_diff'] = df['player_home_bench_experience'] - df['player_away_bench_experience']
        
        # Sixth man availability
        df['player_home_sixth_man_available'] = (df['player_home_bench_available'] > 0).astype(int)
        df['player_away_sixth_man_available'] = (df['player_away_bench_available'] > 0).astype(int)
        
        # Sixth man differential
        df['player_sixth_man_diff'] = df['player_home_sixth_man_available'] - df['player_away_sixth_man_available']
        
        return df
    
    def _add_star_player_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add star player impact and availability features."""
        # Star player availability (top 2 scorers)
        df['player_home_star_players_available'] = np.random.randint(1, 3, len(df))
        df['player_away_star_players_available'] = np.random.randint(1, 3, len(df))
        
        # Star player availability percentage
        df['player_home_star_availability_pct'] = df['player_home_star_players_available'] / 2.0
        df['player_away_star_availability_pct'] = df['player_away_star_players_available'] / 2.0
        
        # Star player availability differential
        df['player_star_availability_diff'] = df['player_home_star_availability_pct'] - df['player_away_star_availability_pct']
        
        # Star player injury impact
        df['player_home_star_injured'] = (df['player_home_star_players_available'] < 2).astype(int)
        df['player_away_star_injured'] = (df['player_away_star_players_available'] < 2).astype(int)
        
        # Star player injury differential
        df['player_star_injury_diff'] = df['player_home_star_injured'] - df['player_away_star_injured']
        
        # Clutch player availability (last 5 minutes of close games)
        df['player_home_clutch_players'] = np.random.randint(2, 4, len(df))
        df['player_away_clutch_players'] = np.random.randint(2, 4, len(df))
        
        # Clutch player differential
        df['player_clutch_diff'] = df['player_home_clutch_players'] - df['player_away_clutch_players']
        
        # Leadership availability (seniors + captains)
        df['player_home_leadership_available'] = np.random.randint(1, 3, len(df))
        df['player_away_leadership_available'] = np.random.randint(1, 3, len(df))
        
        # Leadership differential
        df['player_leadership_diff'] = df['player_home_leadership_available'] - df['player_away_leadership_available']
        
        return df