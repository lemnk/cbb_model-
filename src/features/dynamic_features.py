"""
Dynamic Features for NCAA CBB Betting ML System.

This module handles dynamic situational feature engineering including:
- Travel distance and fatigue
- Rest days and back-to-back indicators
- Altitude adjustments
- Game timing and situational factors
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta


class DynamicFeatures:
    """Engineers dynamic situational features for CBB betting analysis."""
    
    def __init__(self, db_engine):
        """Initialize the dynamic feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        self.engine = db_engine
    
    def transform(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Dynamic situational features:
        - Travel distance
        - Rest days
        - Back-to-back indicator
        - Altitude adjustment
        
        Args:
            games_df: DataFrame with games data
            
        Returns:
            DataFrame with dynamic features prefixed with 'dynamic_'
        """
        if games_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        features_df = games_df.copy()
        
        # Add travel and distance features
        features_df = self._add_travel_features(features_df)
        
        # Add rest and fatigue features
        features_df = self._add_rest_features(features_df)
        
        # Add altitude features
        features_df = self._add_altitude_features(features_df)
        
        # Add timing features
        features_df = self._add_timing_features(features_df)
        
        # Add situational features
        features_df = self._add_situational_features(features_df)
        
        # Add momentum features
        features_df = self._add_momentum_features(features_df)
        
        return features_df
    
    def _add_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add travel distance and fatigue features."""
        # Simulated travel distances (replace with real geocoding data)
        np.random.seed(42)
        
        # Home team travel distance (usually minimal)
        df['dynamic_home_travel_distance'] = np.random.exponential(50, len(df))  # miles
        
        # Away team travel distance (longer distances)
        df['dynamic_away_travel_distance'] = np.random.exponential(800, len(df))  # miles
        
        # Travel distance differential
        df['dynamic_travel_distance_diff'] = df['dynamic_away_travel_distance'] - df['dynamic_home_travel_distance']
        
        # Travel fatigue indicators
        df['dynamic_home_travel_fatigue'] = (df['dynamic_home_travel_distance'] > 100).astype(int)
        df['dynamic_away_travel_fatigue'] = (df['dynamic_away_travel_distance'] > 500).astype(int)
        
        # Travel fatigue differential
        df['dynamic_travel_fatigue_diff'] = df['dynamic_away_travel_fatigue'] - df['dynamic_home_travel_fatigue']
        
        # Long distance travel indicators
        df['dynamic_home_long_distance'] = (df['dynamic_home_travel_distance'] > 300).astype(int)
        df['dynamic_away_long_distance'] = (df['dynamic_away_travel_distance'] > 1000).astype(int)
        
        # Long distance differential
        df['dynamic_long_distance_diff'] = df['dynamic_away_long_distance'] - df['dynamic_home_long_distance']
        
        # Travel time estimates (assuming 60 mph average)
        df['dynamic_home_travel_time'] = df['dynamic_home_travel_distance'] / 60.0  # hours
        df['dynamic_away_travel_time'] = df['dynamic_away_travel_distance'] / 60.0  # hours
        
        # Travel time differential
        df['dynamic_travel_time_diff'] = df['dynamic_away_travel_time'] - df['dynamic_home_travel_time']
        
        # Travel efficiency (distance per day)
        df['dynamic_home_travel_efficiency'] = df['dynamic_home_travel_distance'] / 1.0  # miles per day
        df['dynamic_away_travel_efficiency'] = df['dynamic_away_travel_distance'] / 1.0  # miles per day
        
        return df
    
    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest days and fatigue features."""
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by team and date for rolling calculations
        df = df.sort_values(['home_team', 'date'])
        
        # Days since last game for home team
        df['dynamic_home_days_rest'] = df.groupby('home_team')['date'].diff().dt.days
        
        # Sort by away team and date
        df = df.sort_values(['away_team', 'date'])
        
        # Days since last game for away team
        df['dynamic_away_days_rest'] = df.groupby('away_team')['date'].diff().dt.days
        
        # Sort back to original order
        df = df.sort_index()
        
        # Rest advantage
        df['dynamic_rest_advantage'] = df['dynamic_home_days_rest'] - df['dynamic_away_days_rest']
        
        # Rest advantage categories
        df['dynamic_rest_advantage_category'] = pd.cut(
            df['dynamic_rest_advantage'],
            bins=[-10, -3, -1, 0, 1, 3, 10],
            labels=['large_disadvantage', 'disadvantage', 'slight_disadvantage', 'equal', 'slight_advantage', 'advantage', 'large_advantage']
        )
        
        # Back to back indicators
        df['dynamic_home_back_to_back'] = (df['dynamic_home_days_rest'] == 1).astype(int)
        df['dynamic_away_back_to_back'] = (df['dynamic_away_days_rest'] == 1).astype(int)
        
        # Back to back differential
        df['dynamic_back_to_back_diff'] = df['dynamic_away_back_to_back'] - df['dynamic_home_back_to_back']
        
        # Extended rest indicators
        df['dynamic_home_extended_rest'] = (df['dynamic_home_days_rest'] >= 5).astype(int)
        df['dynamic_away_extended_rest'] = (df['dynamic_away_days_rest'] >= 5).astype(int)
        
        # Extended rest differential
        df['dynamic_extended_rest_diff'] = df['dynamic_home_extended_rest'] - df['dynamic_away_extended_rest']
        
        # Rest efficiency (rest days vs travel distance)
        df['dynamic_home_rest_efficiency'] = df['dynamic_home_days_rest'] / (df['dynamic_home_travel_distance'] + 1)
        df['dynamic_away_rest_efficiency'] = df['dynamic_away_days_rest'] / (df['dynamic_away_travel_distance'] + 1)
        
        # Rest efficiency differential
        df['dynamic_rest_efficiency_diff'] = df['dynamic_home_rest_efficiency'] - df['dynamic_away_rest_efficiency']
        
        return df
    
    def _add_altitude_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add altitude and elevation features."""
        # Simulated altitude data (replace with real venue database)
        np.random.seed(42)
        
        # Home venue altitude
        df['dynamic_home_altitude'] = np.random.normal(1000, 800, len(df))  # feet
        df['dynamic_away_altitude'] = np.random.normal(1000, 800, len(df))  # feet
        
        # Altitude difference
        df['dynamic_altitude_difference'] = df['dynamic_home_altitude'] - df['dynamic_away_altitude']
        
        # High altitude indicators
        df['dynamic_home_high_altitude'] = (df['dynamic_home_altitude'] > 4000).astype(int)
        df['dynamic_away_high_altitude'] = (df['dynamic_away_altitude'] > 4000).astype(int)
        
        # High altitude differential
        df['dynamic_high_altitude_diff'] = df['dynamic_home_high_altitude'] - df['dynamic_away_high_altitude']
        
        # Altitude adjustment factors
        df['dynamic_home_altitude_adjustment'] = np.where(
            df['dynamic_home_altitude'] > 4000,
            (df['dynamic_home_altitude'] - 4000) / 1000,  # Adjustment factor
            0
        )
        
        df['dynamic_away_altitude_adjustment'] = np.where(
            df['dynamic_away_altitude'] > 4000,
            (df['dynamic_away_altitude'] - 4000) / 1000,  # Adjustment factor
            0
        )
        
        # Altitude adjustment differential
        df['dynamic_altitude_adjustment_diff'] = df['dynamic_home_altitude_adjustment'] - df['dynamic_away_altitude_adjustment']
        
        # Altitude categories
        df['dynamic_home_altitude_category'] = pd.cut(
            df['dynamic_home_altitude'],
            bins=[0, 1000, 3000, 5000, 8000],
            labels=['sea_level', 'low', 'moderate', 'high', 'very_high']
        )
        
        df['dynamic_away_altitude_category'] = pd.cut(
            df['dynamic_away_altitude'],
            bins=[0, 1000, 3000, 5000, 8000],
            labels=['sea_level', 'low', 'moderate', 'high', 'very_high']
        )
        
        return df
    
    def _add_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game timing and scheduling features."""
        # Ensure date column is datetime
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Day of week features
        df['dynamic_game_day_of_week'] = df['date'].dt.dayofweek
        
        # Weekend indicators
        df['dynamic_weekend_game'] = (df['dynamic_game_day_of_week'] >= 5).astype(int)
        
        # Day of week categories
        df['dynamic_day_category'] = pd.cut(
            df['dynamic_game_day_of_week'],
            bins=[0, 1, 2, 3, 4, 5, 6],
            labels=['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        )
        
        # Month features
        df['dynamic_game_month'] = df['date'].dt.month
        
        # Season timing indicators
        df['dynamic_early_season'] = (df['dynamic_game_month'].isin([11, 12])).astype(int)
        df['dynamic_mid_season'] = (df['dynamic_game_month'].isin([1, 2])).astype(int)
        df['dynamic_late_season'] = (df['dynamic_game_month'].isin([3, 4])).astype(int)
        
        # Conference tournament indicators
        df['dynamic_conference_tournament'] = (df['dynamic_game_month'] == 3).astype(int)
        
        # NCAA tournament indicators
        df['dynamic_ncaa_tournament'] = (df['dynamic_game_month'] == 3).astype(int)
        
        # Holiday game indicators
        holidays = [1, 2, 7, 11, 12]  # January, February, July, November, December
        df['dynamic_holiday_game'] = df['dynamic_game_month'].isin(holidays).astype(int)
        
        return df
    
    def _add_situational_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add situational and context features."""
        # Rivalry indicators (placeholder - would come from team database)
        np.random.seed(42)
        
        # Rivalry game indicators
        df['dynamic_rivalry_game'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
        
        # Conference game indicators (placeholder)
        df['dynamic_conference_game'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
        
        # Non-conference game indicators
        df['dynamic_non_conference_game'] = (df['dynamic_conference_game'] == 0).astype(int)
        
        # Homecoming indicators (placeholder)
        df['dynamic_homecoming_game'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        # Senior night indicators (placeholder)
        df['dynamic_senior_night'] = np.random.choice([0, 1], len(df), p=[0.95, 0.05])
        
        # Special event indicators
        df['dynamic_special_event'] = (
            df['dynamic_rivalry_game'] | 
            df['dynamic_homecoming_game'] | 
            df['dynamic_senior_night']
        ).astype(int)
        
        # Game importance indicators
        df['dynamic_high_importance'] = (
            df['dynamic_rivalry_game'] | 
            df['dynamic_conference_tournament'] | 
            df['dynamic_ncaa_tournament']
        ).astype(int)
        
        # Pressure situation indicators
        df['dynamic_pressure_situation'] = (
            df['dynamic_high_importance'] | 
            df['dynamic_conference_game']
        ).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum and recent performance features."""
        # Sort by team and date for rolling calculations
        df = df.sort_values(['home_team', 'date'])
        
        # Home team recent performance (last 3 games)
        df['dynamic_home_recent_momentum'] = df.groupby('home_team')['home_score'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Home team recent defensive performance
        df['dynamic_home_recent_defense'] = df.groupby('home_team')['away_score'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Sort by away team and date
        df = df.sort_values(['away_team', 'date'])
        
        # Away team recent performance (last 3 games)
        df['dynamic_away_recent_momentum'] = df.groupby('away_team')['away_score'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Away team recent defensive performance
        df['dynamic_away_recent_defense'] = df.groupby('away_team')['home_score'].rolling(
            window=3, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Sort back to original order
        df = df.sort_index()
        
        # Momentum differentials
        df['dynamic_momentum_diff'] = df['dynamic_home_recent_momentum'] - df['dynamic_away_recent_momentum']
        df['dynamic_defense_diff'] = df['dynamic_away_recent_defense'] - df['dynamic_home_recent_defense']
        
        # Combined momentum score
        df['dynamic_home_momentum_score'] = (
            df['dynamic_home_recent_momentum'] - df['dynamic_home_recent_defense']
        )
        
        df['dynamic_away_momentum_score'] = (
            df['dynamic_away_recent_momentum'] - df['dynamic_away_recent_defense']
        )
        
        # Overall momentum differential
        df['dynamic_overall_momentum_diff'] = df['dynamic_home_momentum_score'] - df['dynamic_away_momentum_score']
        
        # Momentum categories
        df['dynamic_momentum_category'] = pd.cut(
            df['dynamic_overall_momentum_diff'],
            bins=[-50, -20, -10, 0, 10, 20, 50],
            labels=['strong_away', 'moderate_away', 'slight_away', 'even', 'slight_home', 'moderate_home', 'strong_home']
        )
        
        return df