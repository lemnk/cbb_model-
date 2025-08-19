"""
Player Feature Engineering for CBB Betting ML System.

This module handles player-level feature engineering including:
- Player availability and injury probabilities
- Foul-out probabilities
- Bench utilization rates
- Player performance metrics
- Lineup composition features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..utils import get_logger, ConfigManager


class PlayerFeatureEngineer:
    """Engineers player-level features for CBB betting analysis."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the player feature engineer.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('player_features')
        
        # Get feature configuration
        self.injury_threshold = self.config.get('features.player.injury_threshold', 0.1)
        self.foul_threshold = self.config.get('features.player.foul_threshold', 4)
    
    def compute_player_availability(self, injury_df: pd.DataFrame, lineup_df: pd.DataFrame) -> pd.DataFrame:
        """Compute player availability features from injury and lineup data.
        
        Args:
            injury_df: DataFrame with player injury information
            lineup_df: DataFrame with lineup and playing time data
            
        Returns:
            DataFrame with player availability features
        """
        if injury_df.empty and lineup_df.empty:
            self.logger.warning("Both injury and lineup DataFrames are empty")
            return pd.DataFrame()
        
        self.logger.info("Computing player availability features...")
        
        # Create base features DataFrame
        if not injury_df.empty:
            features_df = injury_df.copy()
        elif not lineup_df.empty:
            features_df = lineup_df.copy()
        else:
            features_df = pd.DataFrame()
        
        # Add injury-related features
        if not injury_df.empty:
            features_df = self._add_injury_features(features_df, injury_df)
        
        # Add lineup and playing time features
        if not lineup_df.empty:
            features_df = self._add_lineup_features(features_df, lineup_df)
        
        # Add foul-related features
        features_df = self._add_foul_features(features_df)
        
        # Add bench utilization features
        features_df = self._add_bench_features(features_df)
        
        # Add player performance features
        features_df = self._add_performance_features(features_df)
        
        # Add availability summary features
        features_df = self._add_availability_summary(features_df)
        
        self.logger.info(f"Player availability features computed: {len(features_df)} rows")
        return features_df
    
    def _add_injury_features(self, df: pd.DataFrame, injury_df: pd.DataFrame) -> pd.DataFrame:
        """Add injury-related features.
        
        Args:
            df: Base features DataFrame
            injury_df: Injury data DataFrame
            
        Returns:
            DataFrame with injury features added
        """
        # Merge injury data if not already present
        if 'injury_status' not in df.columns:
            # Create sample injury data for demo
            np.random.seed(42)
            df['injury_status'] = np.random.choice(['healthy', 'questionable', 'out'], len(df), p=[0.8, 0.15, 0.05])
        
        # Injury probability features
        df['injury_probability'] = df['injury_status'].map({
            'healthy': 0.0,
            'questionable': 0.3,
            'out': 1.0
        })
        
        # Days since injury (placeholder)
        df['days_since_injury'] = np.random.randint(0, 30, len(df))
        
        # Injury severity (placeholder)
        df['injury_severity'] = np.random.uniform(0, 1, len(df))
        
        # Recovery probability
        df['recovery_probability'] = 1 - (df['days_since_injury'] * df['injury_severity'] / 30)
        df['recovery_probability'] = df['recovery_probability'].clip(0, 1)
        
        # Availability indicator
        df['player_available'] = (df['injury_probability'] < self.injury_threshold).astype(int)
        
        # Injury risk level
        df['injury_risk_level'] = pd.cut(
            df['injury_probability'], 
            bins=[0, 0.1, 0.3, 0.6, 1.0], 
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        return df
    
    def _add_lineup_features(self, df: pd.DataFrame, lineup_df: pd.DataFrame) -> pd.DataFrame:
        """Add lineup and playing time features.
        
        Args:
            df: Base features DataFrame
            lineup_df: Lineup data DataFrame
            
        Returns:
            DataFrame with lineup features added
        """
        # Merge lineup data if not already present
        if 'minutes_played' not in df.columns:
            # Create sample lineup data for demo
            np.random.seed(42)
            df['minutes_played'] = np.random.randint(0, 40, len(df))
            df['games_played'] = np.random.randint(1, 30, len(df))
            df['starter'] = np.random.choice([True, False], len(df), p=[0.2, 0.8])
        
        # Playing time features
        df['avg_minutes_per_game'] = df['minutes_played'] / df['games_played'].clip(1)
        
        # Starter indicator
        df['starter_indicator'] = df['starter'].astype(int)
        
        # Bench player indicator
        df['bench_player'] = (~df['starter']).astype(int)
        
        # Playing time categories
        df['playing_time_category'] = pd.cut(
            df['avg_minutes_per_game'],
            bins=[0, 10, 20, 30, 40],
            labels=['low', 'medium', 'high', 'very_high']
        )
        
        # Minutes efficiency (placeholder)
        df['minutes_efficiency'] = np.random.uniform(0.5, 1.0, len(df))
        
        # Rest days (placeholder)
        df['rest_days'] = np.random.randint(1, 7, len(df))
        
        return df
    
    def _add_foul_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add foul-related features.
        
        Args:
            df: Base features DataFrame
            
        Returns:
            DataFrame with foul features added
        """
        # Foul count (placeholder)
        if 'fouls' not in df.columns:
            np.random.seed(42)
            df['fouls'] = np.random.randint(0, 5, len(df))
        
        # Foul rate per game
        df['foul_rate_per_game'] = df['fouls'] / df['games_played'].clip(1)
        
        # Foul-out probability
        df['foul_out_probability'] = np.where(
            df['fouls'] >= self.foul_threshold,
            1.0,
            df['fouls'] / self.foul_threshold
        )
        
        # Foul trouble indicator
        df['foul_trouble'] = (df['fouls'] >= 3).astype(int)
        
        # Foul risk level
        df['foul_risk_level'] = pd.cut(
            df['fouls'],
            bins=[0, 1, 2, 3, 4, 5],
            labels=['very_low', 'low', 'medium', 'high', 'very_high', 'fouled_out']
        )
        
        # Minutes remaining (placeholder)
        df['minutes_remaining'] = np.random.randint(0, 40, len(df))
        
        # Foul efficiency (fouls per minute)
        df['foul_efficiency'] = df['fouls'] / df['minutes_played'].clip(1)
        
        return df
    
    def _add_bench_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add bench utilization features.
        
        Args:
            df: Base features DataFrame
            
        Returns:
            DataFrame with bench features added
        """
        # Bench depth (placeholder - would come from team roster)
        np.random.seed(42)
        df['bench_depth'] = np.random.randint(8, 15, len(df))
        
        # Bench utilization rate
        df['bench_utilization_rate'] = np.random.uniform(0.3, 0.7, len(df))
        
        # Bench scoring contribution (placeholder)
        df['bench_scoring_contribution'] = np.random.uniform(0.1, 0.4, len(df))
        
        # Bench minutes distribution
        df['bench_minutes_share'] = np.random.uniform(0.2, 0.5, len(df))
        
        # Bench efficiency (placeholder)
        df['bench_efficiency'] = np.random.uniform(0.4, 0.8, len(df))
        
        # Bench depth quality
        df['bench_depth_quality'] = pd.cut(
            df['bench_depth'],
            bins=[0, 10, 12, 15],
            labels=['shallow', 'adequate', 'deep']
        )
        
        return df
    
    def _add_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add player performance features.
        
        Args:
            df: Base features DataFrame
            
        Returns:
            DataFrame with performance features added
        """
        # Performance metrics (placeholders)
        np.random.seed(42)
        
        # Scoring features
        df['points_per_game'] = np.random.uniform(5, 25, len(df))
        df['field_goal_percentage'] = np.random.uniform(0.35, 0.55, len(df))
        df['three_point_percentage'] = np.random.uniform(0.25, 0.45, len(df))
        df['free_throw_percentage'] = np.random.uniform(0.65, 0.85, len(df))
        
        # Rebounding features
        df['rebounds_per_game'] = np.random.uniform(1, 10, len(df))
        df['offensive_rebounds_per_game'] = np.random.uniform(0.5, 3, len(df))
        df['defensive_rebounds_per_game'] = df['rebounds_per_game'] - df['offensive_rebounds_per_game']
        
        # Assists and ball handling
        df['assists_per_game'] = np.random.uniform(1, 8, len(df))
        df['turnovers_per_game'] = np.random.uniform(0.5, 4, len(df))
        df['assist_to_turnover_ratio'] = df['assists_per_game'] / df['turnovers_per_game'].clip(0.1)
        
        # Defensive features
        df['steals_per_game'] = np.random.uniform(0.5, 3, len(df))
        df['blocks_per_game'] = np.random.uniform(0.2, 2, len(df))
        
        # Efficiency metrics
        df['true_shooting_percentage'] = np.random.uniform(0.45, 0.65, len(df))
        df['player_efficiency_rating'] = np.random.uniform(8, 25, len(df))
        
        # Performance consistency (placeholder)
        df['performance_consistency'] = np.random.uniform(0.6, 0.95, len(df))
        
        return df
    
    def _add_availability_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add team-level availability summary features.
        
        Args:
            df: Base features DataFrame
            
        Returns:
            DataFrame with availability summary features added
        """
        # Group by team to compute team-level features
        if 'team_id' in df.columns:
            team_stats = df.groupby('team_id').agg({
                'player_available': ['sum', 'mean'],
                'injury_probability': 'mean',
                'foul_out_probability': 'mean',
                'avg_minutes_per_game': 'mean',
                'bench_utilization_rate': 'mean'
            }).reset_index()
            
            # Flatten column names
            team_stats.columns = ['team_id'] + [
                f'team_{col[0]}_{col[1]}' if col[1] else f'team_{col[0]}'
                for col in team_stats.columns[1:]
            ]
            
            # Merge back to main DataFrame
            df = df.merge(team_stats, on='team_id', how='left')
        
        # Overall availability score
        df['overall_availability_score'] = (
            (1 - df['injury_probability']) * 0.4 +
            (1 - df['foul_out_probability']) * 0.3 +
            (df['player_available']) * 0.3
        )
        
        # Availability risk level
        df['availability_risk_level'] = pd.cut(
            df['overall_availability_score'],
            bins=[0, 0.5, 0.7, 0.9, 1.0],
            labels=['very_high', 'high', 'medium', 'low']
        )
        
        # Minutes availability
        df['minutes_availability'] = df['avg_minutes_per_game'] * df['player_available']
        
        # Performance availability (combines availability with performance)
        df['performance_availability'] = df['overall_availability_score'] * df['player_efficiency_rating']
        
        return df
    
    def compute_team_availability_summary(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """Compute team-level availability summary.
        
        Args:
            player_df: DataFrame with player availability features
            
        Returns:
            DataFrame with team availability summary
        """
        if player_df.empty:
            return pd.DataFrame()
        
        self.logger.info("Computing team availability summary...")
        
        # Group by team and compute summary statistics
        team_summary = player_df.groupby('team_id').agg({
            'player_available': ['count', 'sum', 'mean'],
            'injury_probability': ['mean', 'std'],
            'foul_out_probability': ['mean', 'std'],
            'avg_minutes_per_game': ['mean', 'sum'],
            'bench_utilization_rate': 'mean',
            'overall_availability_score': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        team_summary.columns = ['team_id'] + [
            f'team_{col[0]}_{col[1]}' if col[1] else f'team_{col[0]}'
            for col in team_summary.columns[1:]
        ]
        
        # Add derived metrics
        team_summary['team_available_players'] = team_summary['team_player_available_sum']
        team_summary['team_total_players'] = team_summary['team_player_available_count']
        team_summary['team_availability_rate'] = team_summary['team_player_available_mean']
        team_summary['team_injury_risk'] = team_summary['team_injury_probability_mean']
        team_summary['team_foul_risk'] = team_summary['team_foul_out_probability_mean']
        team_summary['team_minutes_available'] = team_summary['team_avg_minutes_per_game_sum']
        
        return team_summary
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns generated by this engineer.
        
        Returns:
            List of feature column names
        """
        injury_features = [
            'injury_status', 'injury_probability', 'days_since_injury', 'injury_severity',
            'recovery_probability', 'player_available', 'injury_risk_level'
        ]
        
        lineup_features = [
            'minutes_played', 'games_played', 'starter', 'avg_minutes_per_game',
            'starter_indicator', 'bench_player', 'playing_time_category', 'minutes_efficiency',
            'rest_days'
        ]
        
        foul_features = [
            'fouls', 'foul_rate_per_game', 'foul_out_probability', 'foul_trouble',
            'foul_risk_level', 'minutes_remaining', 'foul_efficiency'
        ]
        
        bench_features = [
            'bench_depth', 'bench_utilization_rate', 'bench_scoring_contribution',
            'bench_minutes_share', 'bench_efficiency', 'bench_depth_quality'
        ]
        
        performance_features = [
            'points_per_game', 'field_goal_percentage', 'three_point_percentage',
            'free_throw_percentage', 'rebounds_per_game', 'offensive_rebounds_per_game',
            'defensive_rebounds_per_game', 'assists_per_game', 'turnovers_per_game',
            'assist_to_turnover_ratio', 'steals_per_game', 'blocks_per_game',
            'true_shooting_percentage', 'player_efficiency_rating', 'performance_consistency'
        ]
        
        summary_features = [
            'overall_availability_score', 'availability_risk_level', 'minutes_availability',
            'performance_availability'
        ]
        
        team_features = [
            'team_player_available_sum', 'team_player_available_mean', 'team_injury_probability_mean',
            'team_foul_out_probability_mean', 'team_avg_minutes_per_game_mean',
            'team_bench_utilization_rate_mean', 'team_overall_availability_score_mean'
        ]
        
        return (injury_features + lineup_features + foul_features + bench_features + 
                performance_features + summary_features + team_features)


def create_player_feature_engineer(config_path: str = "config.yaml") -> PlayerFeatureEngineer:
    """Create and return a player feature engineer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        PlayerFeatureEngineer instance
    """
    config = ConfigManager(config_path)
    return PlayerFeatureEngineer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the player feature engineer
    try:
        engineer = create_player_feature_engineer()
        
        # Create sample data
        sample_injury = pd.DataFrame({
            'player_id': [f'player_{i}' for i in range(10)],
            'team_id': ['team_1'] * 5 + ['team_2'] * 5,
            'injury_status': ['healthy', 'questionable', 'out', 'healthy', 'healthy'] * 2
        })
        
        sample_lineup = pd.DataFrame({
            'player_id': [f'player_{i}' for i in range(10)],
            'team_id': ['team_1'] * 5 + ['team_2'] * 5,
            'minutes_played': np.random.randint(0, 40, 10),
            'games_played': np.random.randint(1, 30, 10),
            'starter': [True, False, False, True, False] * 2
        })
        
        # Compute features
        features = engineer.compute_player_availability(sample_injury, sample_lineup)
        
        print(f"Generated {len(features)} feature rows")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Sample features:\n{features[['injury_probability', 'foul_out_probability', 'overall_availability_score']].head()}")
        
    except Exception as e:
        print(f"Error: {e}")