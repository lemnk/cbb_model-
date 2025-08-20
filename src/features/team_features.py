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
from .feature_utils import ensure_time_order, safe_fill


class TeamFeatures:
    """Engineers team-level features for CBB betting analysis."""
    
    def __init__(self):
        """Initialize the team feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        pass
    
    def compute_team_efficiency(self, df):
        """
        Compute team-level efficiency features:
        - Offensive efficiency (points per possession)
        - Defensive efficiency (points allowed per possession)
        - Pace (possessions per game)
        """
        df = df.copy()
        
        # Ensure proper time ordering to prevent data leakage
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Simulate offensive/defensive efficiency (in production, use real data)
        np.random.seed(42)
        
        # Generate realistic efficiency metrics
        df['team_offensive_efficiency'] = np.random.normal(110, 15, len(df))
        df['team_defensive_efficiency'] = np.random.normal(100, 12, len(df))
        df['team_pace'] = np.random.normal(70, 8, len(df))
        
        # Ensure no negative values for efficiency metrics
        df['team_offensive_efficiency'] = df['team_offensive_efficiency'].clip(lower=80)
        df['team_defensive_efficiency'] = df['team_defensive_efficiency'].clip(lower=70)
        df['team_pace'] = df['team_pace'].clip(lower=50, upper=100)
        
        # Calculate efficiency differential
        df['team_efficiency_diff'] = df['team_offensive_efficiency'] - df['team_defensive_efficiency']
        
        # Calculate combined efficiency rating
        df['team_combined_efficiency'] = (df['team_offensive_efficiency'] + df['team_defensive_efficiency']) / 2
        
        return df
    
    def compute_home_away_splits(self, df):
        """
        Compute home/away performance splits:
        - Home win percentage
        - Away win percentage
        - Home/away scoring differentials
        """
        df = df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Simulate home/away performance (in production, use real data)
        np.random.seed(42)
        
        # Generate realistic home/away splits
        df['team_home_win_pct'] = np.random.uniform(0.4, 0.8, len(df))
        df['team_away_win_pct'] = np.random.uniform(0.3, 0.7, len(df))
        
        # Calculate home advantage
        df['team_home_advantage'] = df['team_home_win_pct'] - df['team_away_win_pct']
        
        # Home/away scoring differentials
        df['team_home_scoring_diff'] = np.random.normal(5, 8, len(df))
        df['team_away_scoring_diff'] = np.random.normal(-2, 8, len(df))
        
        # Home court advantage indicator
        df['team_strong_home'] = (df['team_home_win_pct'] > 0.7).astype(int)
        df['team_weak_away'] = (df['team_away_win_pct'] < 0.4).astype(int)
        
        return df
    
    def compute_consistency(self, df):
        """
        Compute team consistency metrics using rolling windows:
        - Scoring consistency (last 3, 5, 10 games)
        - Defensive consistency
        - Performance volatility
        """
        df = df.copy()
        
        # Ensure proper time ordering to prevent data leakage
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Simulate rolling averages for scoring consistency
        np.random.seed(42)
        
        # Generate base consistency metrics
        df['team_scoring_consistency_3g'] = np.random.normal(0, 1, len(df))
        df['team_scoring_consistency_5g'] = np.random.normal(0, 1, len(df))
        df['team_scoring_consistency_10g'] = np.random.normal(0, 1, len(df))
        
        # Generate defensive consistency metrics
        df['team_defensive_consistency_3g'] = np.random.normal(0, 1, len(df))
        df['team_defensive_consistency_5g'] = np.random.normal(0, 1, len(df))
        df['team_defensive_consistency_10g'] = np.random.normal(0, 1, len(df))
        
        # Calculate overall consistency score
        df['team_overall_consistency'] = (
            df['team_scoring_consistency_10g'] + 
            df['team_defensive_consistency_10g']
        ) / 2
        
        # Consistency categories
        df['team_consistency_category'] = pd.cut(
            df['team_overall_consistency'],
            bins=[-3, -1, 0, 1, 3],
            labels=['very_inconsistent', 'inconsistent', 'moderate', 'consistent', 'very_consistent']
        )
        
        # Performance volatility (inverse of consistency)
        df['team_volatility'] = 1 - (df['team_overall_consistency'] + 3) / 6  # Normalize to [0,1]
        
        return df
    
    def compute_strength_of_schedule(self, df):
        """
        Compute strength of schedule metrics:
        - Average opponent efficiency
        - SOS offensive/defensive components
        - Overall SOS rating
        """
        df = df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Simulate strength of schedule metrics
        np.random.seed(42)
        
        # Generate SOS components
        df['team_sos_offensive'] = np.random.normal(100, 10, len(df))
        df['team_sos_defensive'] = np.random.normal(100, 10, len(df))
        
        # Calculate overall SOS
        df['team_sos_overall'] = (df['team_sos_offensive'] + df['team_sos_defensive']) / 2
        
        # SOS difficulty categories
        df['team_sos_difficulty'] = pd.cut(
            df['team_sos_overall'],
            bins=[80, 95, 100, 105, 120],
            labels=['very_easy', 'easy', 'average', 'hard', 'very_hard']
        )
        
        # SOS impact on performance
        df['team_sos_adjusted_efficiency'] = df['team_combined_efficiency'] - (df['team_sos_overall'] - 100) * 0.1
        
        return df
    
    def transform(self, df):
        """
        Apply all team feature transformations with proper time ordering.
        """
        # Ensure we have required columns
        df = safe_fill(df, 'date', pd.Timestamp('2024-01-01'))
        df = safe_fill(df, 'team', 'unknown_team')
        
        # Apply transformations in order
        df = self.compute_team_efficiency(df)
        df = self.compute_home_away_splits(df)
        df = self.compute_consistency(df)
        df = self.compute_strength_of_schedule(df)
        
        # Final safety check - ensure no NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = safe_fill(df, col, 0)
        
        return df