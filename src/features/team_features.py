"""
Team Features for NCAA CBB Betting ML System.

This module handles team-level feature engineering including:
- Offensive and defensive efficiency
- Pace and tempo metrics
- Win streaks and performance trends
- Strength of schedule calculations
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta


class TeamFeatures:
    """Engineers team-level features for CBB betting analysis."""
    
    def __init__(self, db_engine):
        """Initialize the team feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        self.engine = db_engine
    
    def transform(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute team-level features:
        - Offensive efficiency
        - Defensive efficiency
        - Pace
        - Win streaks
        - Strength of schedule (avg opp efficiency)
        
        Args:
            games_df: DataFrame with games data
            
        Returns:
            DataFrame with team features prefixed with 'team_'
        """
        if games_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        features_df = games_df.copy()
        
        # Basic team performance features
        features_df = self._add_basic_performance_features(features_df)
        
        # Efficiency ratings
        features_df = self._add_efficiency_features(features_df)
        
        # Win streaks and trends
        features_df = self._add_streak_features(features_df)
        
        # Strength of schedule
        features_df = self._add_strength_of_schedule(features_df)
        
        # Rolling performance metrics
        features_df = self._add_rolling_features(features_df)
        
        return features_df
    
    def _add_basic_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic team performance features."""
        # Score differential
        df['team_home_score_diff'] = df['home_score'] - df['away_score']
        df['team_away_score_diff'] = df['away_score'] - df['home_score']
        
        # Total score
        df['team_total_score'] = df['home_score'] + df['away_score']
        
        # Win/loss indicators
        df['team_home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['team_away_win'] = (df['away_score'] > df['home_score']).astype(int)
        
        # Game margin
        df['team_game_margin'] = abs(df['home_score'] - df['away_score'])
        
        # High scoring game indicator
        df['team_high_scoring_game'] = (df['team_total_score'] > 150).astype(int)
        
        # Close game indicator
        df['team_close_game'] = (df['team_game_margin'] <= 5).astype(int)
        
        return df
    
    def _add_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add offensive and defensive efficiency features."""
        # Simulated efficiency ratings (replace with real KenPom data)
        np.random.seed(42)  # For reproducible demo
        
        # Home team efficiency
        df['team_home_offensive_efficiency'] = np.random.normal(110, 10, len(df))
        df['team_home_defensive_efficiency'] = np.random.normal(100, 10, len(df))
        df['team_home_pace'] = np.random.normal(70, 5, len(df))
        
        # Away team efficiency
        df['team_away_offensive_efficiency'] = np.random.normal(110, 10, len(df))
        df['team_away_defensive_efficiency'] = np.random.normal(100, 10, len(df))
        df['team_away_pace'] = np.random.normal(70, 5, len(df))
        
        # Efficiency differentials
        df['team_offensive_efficiency_diff'] = (
            df['team_home_offensive_efficiency'] - df['team_away_offensive_efficiency']
        )
        df['team_defensive_efficiency_diff'] = (
            df['team_home_defensive_efficiency'] - df['team_away_defensive_efficiency']
        )
        df['team_pace_diff'] = df['team_home_pace'] - df['team_away_pace']
        
        # Combined efficiency metrics
        df['team_home_efficiency_rating'] = (
            df['team_home_offensive_efficiency'] - df['team_home_defensive_efficiency']
        )
        df['team_away_efficiency_rating'] = (
            df['team_away_offensive_efficiency'] - df['team_away_defensive_efficiency']
        )
        df['team_efficiency_rating_diff'] = (
            df['team_home_efficiency_rating'] - df['team_away_efficiency_rating']
        )
        
        # Predicted pace
        df['team_predicted_pace'] = (df['team_home_pace'] + df['team_away_pace']) / 2
        
        return df
    
    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add win streak and performance trend features."""
        # Sort by team and date for rolling calculations
        df = df.sort_values(['home_team', 'date'])
        
        # Home team win streaks
        df['team_home_win_streak'] = df.groupby('home_team')['team_home_win'].rolling(
            window=10, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # Away team win streaks
        df = df.sort_values(['away_team', 'date'])
        df['team_away_win_streak'] = df.groupby('away_team')['team_away_win'].rolling(
            window=10, min_periods=1
        ).sum().reset_index(0, drop=True)
        
        # Sort back to original order
        df = df.sort_index()
        
        # Win streak differential
        df['team_win_streak_diff'] = df['team_home_win_streak'] - df['team_away_win_streak']
        
        # Recent performance (last 5 games)
        df = df.sort_values(['home_team', 'date'])
        df['team_home_recent_performance'] = df.groupby('home_team')['home_score'].rolling(
            window=5, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        df = df.sort_values(['away_team', 'date'])
        df['team_away_recent_performance'] = df.groupby('away_team')['away_score'].rolling(
            window=5, min_periods=1
        ).mean().reset_index(0, drop=True)
        
        # Sort back to original order
        df = df.sort_index()
        
        # Recent performance differential
        df['team_recent_performance_diff'] = (
            df['team_home_recent_performance'] - df['team_away_recent_performance']
        )
        
        return df
    
    def _add_strength_of_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add strength of schedule features."""
        # Calculate average opponent efficiency for each team
        # This is a simplified version - in production you'd calculate this more accurately
        
        # Home team strength of schedule (avg away team efficiency)
        df['team_home_sos_offensive'] = df.groupby('home_team')['team_away_offensive_efficiency'].transform('mean')
        df['team_home_sos_defensive'] = df.groupby('home_team')['team_away_defensive_efficiency'].transform('mean')
        
        # Away team strength of schedule (avg home team efficiency)
        df['team_away_sos_offensive'] = df.groupby('away_team')['team_home_offensive_efficiency'].transform('mean')
        df['team_away_sos_defensive'] = df.groupby('away_team')['team_home_defensive_efficiency'].transform('mean')
        
        # SOS differentials
        df['team_sos_offensive_diff'] = df['team_home_sos_offensive'] - df['team_away_sos_offensive']
        df['team_sos_defensive_diff'] = df['team_home_sos_defensive'] - df['team_away_sos_defensive']
        
        # Overall SOS rating
        df['team_home_sos_rating'] = (df['team_home_sos_offensive'] + df['team_home_sos_defensive']) / 2
        df['team_away_sos_rating'] = (df['team_away_sos_offensive'] + df['team_away_sos_defensive']) / 2
        df['team_sos_rating_diff'] = df['team_home_sos_rating'] - df['team_away_sos_rating']
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance features."""
        # Rolling windows
        windows = [3, 5, 10]
        
        # Sort by team and date for rolling calculations
        df = df.sort_values(['home_team', 'date'])
        
        for window in windows:
            # Home team rolling features
            df[f'team_home_rolling_score_{window}'] = df.groupby('home_team')['home_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'team_home_rolling_conceded_{window}'] = df.groupby('home_team')['away_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'team_home_rolling_margin_{window}'] = df.groupby('home_team')['team_home_score_diff'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Sort by away team and date
        df = df.sort_values(['away_team', 'date'])
        
        for window in windows:
            # Away team rolling features
            df[f'team_away_rolling_score_{window}'] = df.groupby('away_team')['away_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'team_away_rolling_conceded_{window}'] = df.groupby('away_team')['home_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'team_away_rolling_margin_{window}'] = df.groupby('away_team')['team_away_score_diff'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Sort back to original order
        df = df.sort_index()
        
        # Rolling differentials
        for window in windows:
            df[f'team_rolling_score_diff_{window}'] = (
                df[f'team_home_rolling_score_{window}'] - df[f'team_away_rolling_score_{window}']
            )
            
            df[f'team_rolling_margin_diff_{window}'] = (
                df[f'team_home_rolling_margin_{window}'] - df[f'team_away_rolling_margin_{window}']
            )
        
        return df