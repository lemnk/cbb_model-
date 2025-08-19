"""
Dynamic Feature Engineering for CBB Betting ML System.

This module handles dynamic features derived from game flow and play-by-play data:
- Momentum index calculations
- Run-length encoding for streaks
- Game flow indicators
- Possession-based metrics
- Time-based features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..utils import get_logger, ConfigManager


class DynamicFeatureEngineer:
    """Engineers dynamic features from game flow and play-by-play data."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the dynamic feature engineer.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('dynamic_features')
        
        # Get feature configuration
        self.momentum_alpha = self.config.get('features.momentum.alpha', 0.7)
        self.momentum_beta = self.config.get('features.momentum.beta', 0.3)
        self.rolling_windows = self.config.get('features.rolling_windows', [3, 5, 10, 20])
    
    def compute_game_flow(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """Compute game flow features from play-by-play data.
        
        Args:
            pbp_df: DataFrame with play-by-play data
            
        Returns:
            DataFrame with game flow features
        """
        if pbp_df.empty:
            self.logger.warning("Empty play-by-play DataFrame provided")
            return pd.DataFrame()
        
        self.logger.info("Computing game flow features...")
        
        # Create a copy to avoid modifying original
        features_df = pbp_df.copy()
        
        # Basic play-by-play features
        features_df = self._add_basic_pbp_features(features_df)
        
        # Momentum index features
        features_df = self._add_momentum_features(features_df)
        
        # Run-length encoding for streaks
        features_df = self._add_streak_features(features_df)
        
        # Possession-based features
        features_df = self._add_possession_features(features_df)
        
        # Time-based features
        features_df = self._add_time_features(features_df)
        
        # Rolling game flow features
        features_df = self._add_rolling_flow_features(features_df)
        
        self.logger.info(f"Game flow features computed: {len(features_df)} rows")
        return features_df
    
    def _add_basic_pbp_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic play-by-play features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with basic PBP features added
        """
        # Ensure required columns exist (add placeholders if missing)
        required_columns = ['game_id', 'quarter', 'time_remaining', 'home_score', 'away_score']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'quarter':
                    df[col] = np.random.randint(1, 5, len(df))
                elif col == 'time_remaining':
                    df[col] = np.random.randint(0, 1200, len(df))  # seconds
                elif col == 'home_score':
                    df[col] = np.random.randint(0, 100, len(df))
                elif col == 'away_score':
                    df[col] = np.random.randint(0, 100, len(df))
        
        # Score differential at each play
        df['score_differential'] = df['home_score'] - df['away_score']
        
        # Lead change indicator
        df['lead_change'] = df['score_differential'].diff().abs() > 0
        
        # Quarter progression
        df['quarter_progression'] = df['quarter'] / 4.0
        
        # Time pressure (inverse of time remaining)
        df['time_pressure'] = 1.0 / (df['time_remaining'] + 1)
        
        # Game state indicators
        df['early_game'] = (df['quarter'] <= 2).astype(int)
        df['late_game'] = (df['quarter'] >= 3).astype(int)
        df['final_minutes'] = ((df['quarter'] == 4) & (df['time_remaining'] <= 300)).astype(int)
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum index features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with momentum features added
        """
        # Momentum Index: M_t = α * Δscore_t + β * Δpossessions_t
        
        # Score change (Δscore)
        df['score_change'] = df['score_differential'].diff().fillna(0)
        
        # Possession change (Δpossessions) - placeholder for real PBP data
        # In production, this would come from actual possession tracking
        np.random.seed(42)
        df['possession_change'] = np.random.choice([-1, 0, 1], len(df), p=[0.3, 0.4, 0.3])
        
        # Momentum index calculation
        df['momentum_index'] = (
            self.momentum_alpha * df['score_change'] + 
            self.momentum_beta * df['possession_change']
        )
        
        # Rolling momentum (last N plays)
        for window in [3, 5, 10]:
            df[f'momentum_rolling_{window}'] = df['momentum_index'].rolling(
                window=window, min_periods=1
            ).mean()
        
        # Momentum acceleration (change in momentum)
        df['momentum_acceleration'] = df['momentum_index'].diff().fillna(0)
        
        # Momentum volatility
        df['momentum_volatility'] = df['momentum_index'].rolling(
            window=10, min_periods=1
        ).std().fillna(0)
        
        # Momentum regime indicators
        df['high_momentum'] = (df['momentum_index'] > df['momentum_index'].quantile(0.75)).astype(int)
        df['low_momentum'] = (df['momentum_index'] < df['momentum_index'].quantile(0.25)).astype(int)
        
        return df
    
    def _add_streak_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add run-length encoding for streaks.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with streak features added
        """
        # Run-length encoding for score differential streaks
        df = df.sort_values(['game_id', 'quarter', 'time_remaining'])
        
        # Home team scoring streaks
        df['home_scoring_streak'] = self._encode_run_lengths(df['score_differential'] > 0)
        
        # Away team scoring streaks
        df['away_scoring_streak'] = self._encode_run_lengths(df['score_differential'] < 0)
        
        # Tied game streaks
        df['tied_streak'] = self._encode_run_lengths(df['score_differential'] == 0)
        
        # Lead change frequency
        df['lead_changes_rolling_10'] = df['lead_change'].rolling(
            window=10, min_periods=1
        ).sum()
        
        # Streak momentum
        df['home_streak_momentum'] = df['home_scoring_streak'] * df['momentum_index']
        df['away_streak_momentum'] = df['away_scoring_streak'] * df['momentum_index']
        
        # Streak break indicators
        df['home_streak_broken'] = (df['home_scoring_streak'] == 1).astype(int)
        df['away_streak_broken'] = (df['away_scoring_streak'] == 1).astype(int)
        
        return df
    
    def _add_possession_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add possession-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with possession features added
        """
        # Possession efficiency (placeholder for real data)
        np.random.seed(42)
        
        # Simulated possession data
        df['possession_duration'] = np.random.exponential(20, len(df))  # seconds
        df['possession_efficiency'] = np.random.uniform(0.3, 0.7, len(df))
        
        # Possession count
        df['possession_count'] = df.groupby('game_id').cumcount() + 1
        
        # Possession pace
        df['possession_pace'] = df['possession_duration'].rolling(
            window=5, min_periods=1
        ).mean()
        
        # Possession efficiency rolling
        for window in [3, 5, 10]:
            df[f'possession_efficiency_rolling_{window}'] = df['possession_efficiency'].rolling(
                window=window, min_periods=1
            ).mean()
        
        # Possession advantage
        df['possession_advantage'] = df['possession_efficiency'] - df['possession_efficiency'].rolling(
            window=10, min_periods=1
        ).mean()
        
        # Fast break indicators (placeholder)
        df['fast_break'] = (df['possession_duration'] < 10).astype(int)
        df['slow_break'] = (df['possession_duration'] > 30).astype(int)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features added
        """
        # Time remaining features
        df['time_remaining_minutes'] = df['time_remaining'] / 60.0
        
        # Quarter time progression
        df['quarter_time_progression'] = (1200 - df['time_remaining']) / 1200.0
        
        # Game time progression
        df['game_time_progression'] = ((4 - df['quarter']) * 1200 + (1200 - df['time_remaining'])) / 4800.0
        
        # Time pressure indicators
        df['high_time_pressure'] = (df['time_remaining'] <= 120).astype(int)  # Last 2 minutes
        df['medium_time_pressure'] = ((df['time_remaining'] > 120) & (df['time_remaining'] <= 300)).astype(int)  # Last 5 minutes
        
        # Quarter transition indicators
        df['quarter_start'] = (df['time_remaining'] >= 1180).astype(int)  # First 20 seconds
        df['quarter_end'] = (df['time_remaining'] <= 20).astype(int)      # Last 20 seconds
        
        # Halftime indicator
        df['halftime'] = ((df['quarter'] == 2) & (df['time_remaining'] <= 20)).astype(int)
        
        # Overtime indicators
        df['overtime_period'] = (df['quarter'] > 4).astype(int)
        
        return df
    
    def _add_rolling_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling game flow features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling flow features added
        """
        # Sort by game and time for rolling calculations
        df = df.sort_values(['game_id', 'quarter', 'time_remaining'])
        
        # Rolling score differential
        for window in [3, 5, 10]:
            df[f'score_diff_rolling_{window}'] = df['score_differential'].rolling(
                window=window, min_periods=1
            ).mean()
            
            df[f'score_diff_volatility_{window}'] = df['score_differential'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
        
        # Rolling momentum features
        for window in [3, 5, 10]:
            df[f'momentum_volatility_{window}'] = df['momentum_index'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
            
            df[f'momentum_trend_{window}'] = df['momentum_index'].rolling(
                window=window, min_periods=1
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        
        # Rolling possession features
        for window in [3, 5, 10]:
            df[f'possession_pace_rolling_{window}'] = df['possession_pace'].rolling(
                window=window, min_periods=1
            ).mean()
            
            df[f'possession_efficiency_volatility_{window}'] = df['possession_efficiency'].rolling(
                window=window, min_periods=1
            ).std().fillna(0)
        
        return df
    
    def _encode_run_lengths(self, series: pd.Series) -> pd.Series:
        """Encode run lengths for a boolean series.
        
        Args:
            series: Boolean series to encode
            
        Returns:
            Series with run lengths
        """
        # Group consecutive True values and count them
        groups = (series != series.shift()).cumsum()
        run_lengths = series.groupby(groups).cumsum()
        
        # Reset to 0 when False
        run_lengths = run_lengths.where(series, 0)
        
        return run_lengths
    
    def compute_momentum_index(self, score_change: float, possession_change: float) -> float:
        """Compute momentum index for given score and possession changes.
        
        Args:
            score_change: Change in score differential
            possession_change: Change in possession count
            
        Returns:
            Momentum index value
        """
        return self.momentum_alpha * score_change + self.momentum_beta * possession_change
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns generated by this engineer.
        
        Returns:
            List of feature column names
        """
        basic_features = [
            'score_differential', 'lead_change', 'quarter_progression', 'time_pressure',
            'early_game', 'late_game', 'final_minutes'
        ]
        
        momentum_features = [
            'score_change', 'possession_change', 'momentum_index', 'momentum_acceleration',
            'momentum_volatility', 'high_momentum', 'low_momentum'
        ]
        
        rolling_momentum_features = []
        for window in [3, 5, 10]:
            rolling_momentum_features.extend([
                f'momentum_rolling_{window}', f'momentum_volatility_{window}', f'momentum_trend_{window}'
            ])
        
        streak_features = [
            'home_scoring_streak', 'away_scoring_streak', 'tied_streak',
            'lead_changes_rolling_10', 'home_streak_momentum', 'away_streak_momentum',
            'home_streak_broken', 'away_streak_broken'
        ]
        
        possession_features = [
            'possession_duration', 'possession_efficiency', 'possession_count',
            'possession_pace', 'possession_advantage', 'fast_break', 'slow_break'
        ]
        
        rolling_possession_features = []
        for window in [3, 5, 10]:
            rolling_possession_features.extend([
                f'possession_efficiency_rolling_{window}', f'possession_pace_rolling_{window}',
                f'possession_efficiency_volatility_{window}'
            ])
        
        time_features = [
            'time_remaining_minutes', 'quarter_time_progression', 'game_time_progression',
            'high_time_pressure', 'medium_time_pressure', 'quarter_start', 'quarter_end',
            'halftime', 'overtime_period'
        ]
        
        rolling_flow_features = []
        for window in [3, 5, 10]:
            rolling_flow_features.extend([
                f'score_diff_rolling_{window}', f'score_diff_volatility_{window}'
            ])
        
        return (basic_features + momentum_features + rolling_momentum_features + 
                streak_features + possession_features + rolling_possession_features + 
                time_features + rolling_flow_features)


def create_dynamic_feature_engineer(config_path: str = "config.yaml") -> DynamicFeatureEngineer:
    """Create and return a dynamic feature engineer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DynamicFeatureEngineer instance
    """
    config = ConfigManager(config_path)
    return DynamicFeatureEngineer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the dynamic feature engineer
    try:
        engineer = create_dynamic_feature_engineer()
        
        # Create sample play-by-play data
        sample_pbp = pd.DataFrame({
            'game_id': ['game_1'] * 20,
            'quarter': [1] * 5 + [2] * 5 + [3] * 5 + [4] * 5,
            'time_remaining': list(range(1200, 0, -60)),
            'home_score': [0, 2, 5, 7, 10, 12, 15, 18, 20, 25, 28, 30, 35, 38, 40, 45, 48, 50, 55, 58],
            'away_score': [0, 0, 2, 5, 7, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45]
        })
        
        # Compute features
        features = engineer.compute_game_flow(sample_pbp)
        
        print(f"Generated {len(features)} feature rows")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Sample momentum features:\n{features[['momentum_index', 'momentum_rolling_5', 'home_scoring_streak']].head()}")
        
    except Exception as e:
        print(f"Error: {e}")