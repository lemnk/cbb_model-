"""
Team Feature Engineering for CBB Betting ML System.

This module handles team-level feature engineering including:
- Team context and performance metrics
- Efficiency ratings (AdjO, AdjD, Tempo)
- Rebounding and turnover percentages
- Travel fatigue indicators
- Conference and division features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..utils import get_logger, ConfigManager


class TeamFeatureEngineer:
    """Engineers team-level features for CBB betting analysis."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the team feature engineer.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('team_features')
        
        # Get feature configuration
        self.rolling_windows = self.config.get('features.rolling_windows', [3, 5, 10, 20])
        self.advanced_stats = self.config.get('features.advanced_stats', [])
    
    def compute_team_context(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Compute team context features from games data.
        
        Args:
            games_df: DataFrame with games data including scores, teams, dates
            
        Returns:
            DataFrame with team context features
        """
        if games_df.empty:
            self.logger.warning("Empty games DataFrame provided")
            return pd.DataFrame()
        
        self.logger.info("Computing team context features...")
        
        # Create a copy to avoid modifying original
        features_df = games_df.copy()
        
        # Basic team performance features
        features_df = self._add_basic_performance_features(features_df)
        
        # Efficiency and tempo features
        features_df = self._add_efficiency_features(features_df)
        
        # Travel and fatigue features
        features_df = self._add_travel_features(features_df)
        
        # Conference and division features
        features_df = self._add_conference_features(features_df)
        
        # Rolling performance features
        features_df = self._add_rolling_features(features_df)
        
        self.logger.info(f"Team context features computed: {len(features_df)} rows")
        return features_df
    
    def _add_basic_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic team performance features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with basic performance features added
        """
        # Score differential
        df['home_score_diff'] = df['home_score'] - df['away_score']
        df['away_score_diff'] = df['away_score'] - df['home_score']
        
        # Total score
        df['total_score'] = df['home_score'] + df['away_score']
        
        # Win/loss indicators
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
        df['away_win'] = (df['away_score'] > df['home_score']).astype(int)
        
        # Overtime indicator
        if 'overtime' in df.columns:
            df['overtime_indicator'] = df['overtime'].astype(int)
        else:
            df['overtime_indicator'] = 0
        
        # Game margin
        df['game_margin'] = abs(df['home_score'] - df['away_score'])
        
        # High scoring game indicator
        df['high_scoring_game'] = (df['total_score'] > 150).astype(int)
        
        # Close game indicator
        df['close_game'] = (df['game_margin'] <= 5).astype(int)
        
        return df
    
    def _add_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add efficiency and tempo features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with efficiency features added
        """
        # Placeholder for KenPom efficiency ratings
        # In production, these would come from KenPom API or database
        
        # Simulated efficiency ratings (replace with real data)
        np.random.seed(42)  # For reproducible demo
        
        # Home team efficiency
        df['home_adj_o'] = np.random.normal(110, 10, len(df))  # Offensive efficiency
        df['home_adj_d'] = np.random.normal(100, 10, len(df))  # Defensive efficiency
        df['home_tempo'] = np.random.normal(70, 5, len(df))    # Tempo
        
        # Away team efficiency
        df['away_adj_o'] = np.random.normal(110, 10, len(df))
        df['away_adj_d'] = np.random.normal(100, 10, len(df))
        df['away_tempo'] = np.random.normal(70, 5, len(df))
        
        # Efficiency differentials
        df['offensive_efficiency_diff'] = df['home_adj_o'] - df['away_adj_o']
        df['defensive_efficiency_diff'] = df['home_adj_d'] - df['away_adj_d']
        df['tempo_diff'] = df['home_tempo'] - df['away_tempo']
        
        # Combined efficiency metrics
        df['home_efficiency_rating'] = df['home_adj_o'] - df['home_adj_d']
        df['away_efficiency_rating'] = df['away_adj_o'] - df['away_adj_d']
        df['efficiency_rating_diff'] = df['home_efficiency_rating'] - df['away_efficiency_rating']
        
        # Pace prediction
        df['predicted_pace'] = (df['home_tempo'] + df['away_tempo']) / 2
        
        return df
    
    def _add_travel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add travel and fatigue features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with travel features added
        """
        # Convert date to datetime if needed
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by team and date for rolling calculations
        df = df.sort_values(['home_team', 'date'])
        
        # Days since last game for home team
        df['home_days_rest'] = df.groupby('home_team')['date'].diff().dt.days
        
        # Days since last game for away team
        df = df.sort_values(['away_team', 'date'])
        df['away_days_rest'] = df.groupby('away_team')['date'].diff().dt.days
        
        # Back to back indicators
        df['home_back_to_back'] = (df['home_days_rest'] == 1).astype(int)
        df['away_back_to_back'] = (df['away_days_rest'] == 1).astype(int)
        
        # Rest advantage
        df['rest_advantage'] = df['home_days_rest'] - df['away_days_rest']
        
        # Travel distance features (placeholder - would come from geocoding API)
        # Simulated travel distances
        np.random.seed(42)
        df['home_travel_distance'] = np.random.exponential(500, len(df))  # miles
        df['away_travel_distance'] = np.random.exponential(800, len(df))  # miles
        
        # Travel fatigue indicators
        df['home_travel_fatigue'] = (df['home_travel_distance'] > 1000).astype(int)
        df['away_travel_fatigue'] = (df['away_travel_distance'] > 1000).astype(int)
        
        # Altitude adjustment (placeholder - would come from venue database)
        df['home_altitude'] = np.random.normal(1000, 500, len(df))  # feet
        df['away_altitude'] = np.random.normal(1000, 500, len(df))
        df['altitude_difference'] = df['home_altitude'] - df['away_altitude']
        
        # High altitude game indicator
        df['high_altitude_game'] = ((df['home_altitude'] > 4000) | (df['away_altitude'] > 4000)).astype(int)
        
        return df
    
    def _add_conference_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add conference and division features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with conference features added
        """
        # Conference information (placeholder - would come from team database)
        # Simulated conference assignments
        np.random.seed(42)
        conferences = ['ACC', 'Big Ten', 'Big 12', 'SEC', 'Pac-12', 'Big East', 'American', 'Mountain West']
        
        # Assign random conferences to teams for demo
        unique_teams = pd.concat([df['home_team'], df['away_team']]).unique()
        team_conferences = {team: np.random.choice(conferences) for team in unique_teams}
        
        # Add conference columns
        df['home_conference'] = df['home_team'].map(team_conferences)
        df['away_conference'] = df['away_team'].map(team_conferences)
        
        # Conference game indicators
        df['conference_game'] = (df['home_conference'] == df['away_conference']).astype(int)
        
        # Power conference indicators
        power_conferences = ['ACC', 'Big Ten', 'Big 12', 'SEC', 'Pac-12', 'Big East']
        df['home_power_conference'] = df['home_conference'].isin(power_conferences).astype(int)
        df['away_power_conference'] = df['away_conference'].isin(power_conferences).astype(int)
        
        # Conference strength differential
        df['conference_strength_diff'] = df['home_power_conference'] - df['away_power_conference']
        
        # Division features (placeholder)
        df['home_division'] = np.random.choice(['East', 'West', 'North', 'South'], len(df))
        df['away_division'] = np.random.choice(['East', 'West', 'North', 'South'], len(df))
        df['division_game'] = (df['home_division'] == df['away_division']).astype(int)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling performance features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with rolling features added
        """
        # Sort by team and date
        df = df.sort_values(['home_team', 'date'])
        
        # Rolling averages for home team
        for window in self.rolling_windows:
            # Home team rolling features
            df[f'home_rolling_score_{window}'] = df.groupby('home_team')['home_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'home_rolling_conceded_{window}'] = df.groupby('home_team')['away_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'home_rolling_margin_{window}'] = df.groupby('home_team')['home_score_diff'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Home team rolling win rate
            df[f'home_rolling_win_rate_{window}'] = df.groupby('home_team')['home_win'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Sort by away team and date for away team features
        df = df.sort_values(['away_team', 'date'])
        
        # Rolling averages for away team
        for window in self.rolling_windows:
            # Away team rolling features
            df[f'away_rolling_score_{window}'] = df.groupby('away_team')['away_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'away_rolling_conceded_{window}'] = df.groupby('away_team')['home_score'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df[f'away_rolling_margin_{window}'] = df.groupby('away_team')['away_score_diff'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            # Away team rolling win rate
            df[f'away_rolling_win_rate_{window}'] = df.groupby('away_team')['away_win'].rolling(
                window=window, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Rolling performance differentials
        for window in self.rolling_windows:
            df[f'rolling_score_diff_{window}'] = (
                df[f'home_rolling_score_{window}'] - df[f'away_rolling_score_{window}']
            )
            
            df[f'rolling_conceded_diff_{window}'] = (
                df[f'away_rolling_conceded_{window}'] - df[f'home_rolling_conceded_{window}']
            )
            
            df[f'rolling_win_rate_diff_{window}'] = (
                df[f'home_rolling_win_rate_{window}'] - df[f'away_rolling_win_rate_{window}']
            )
        
        return df
    
    def compute_team_efficiency_ratings(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Compute team efficiency ratings from historical games.
        
        Args:
            games_df: DataFrame with historical games data
            
        Returns:
            DataFrame with team efficiency ratings
        """
        if games_df.empty:
            return pd.DataFrame()
        
        self.logger.info("Computing team efficiency ratings...")
        
        # Group by team and compute efficiency metrics
        team_stats = []
        
        for team in pd.concat([games_df['home_team'], games_df['away_team']]).unique():
            # Home games
            home_games = games_df[games_df['home_team'] == team]
            away_games = games_df[games_df['away_team'] == team]
            
            if len(home_games) > 0:
                home_offensive_efficiency = home_games['home_score'].mean()
                home_defensive_efficiency = home_games['away_score'].mean()
            else:
                home_offensive_efficiency = home_defensive_efficiency = 0
            
            if len(away_games) > 0:
                away_offensive_efficiency = away_games['away_score'].mean()
                away_defensive_efficiency = away_games['home_score'].mean()
            else:
                away_offensive_efficiency = away_defensive_efficiency = 0
            
            # Combined efficiency (weighted by number of games)
            total_home = len(home_games)
            total_away = len(away_games)
            total_games = total_home + total_away
            
            if total_games > 0:
                offensive_efficiency = (
                    (home_offensive_efficiency * total_home + away_offensive_efficiency * total_away) / total_games
                )
                defensive_efficiency = (
                    (home_defensive_efficiency * total_home + away_defensive_efficiency * total_away) / total_games
                )
                
                team_stats.append({
                    'team': team,
                    'games_played': total_games,
                    'offensive_efficiency': offensive_efficiency,
                    'defensive_efficiency': defensive_efficiency,
                    'efficiency_rating': offensive_efficiency - defensive_efficiency,
                    'home_games': total_home,
                    'away_games': total_away
                })
        
        return pd.DataFrame(team_stats)
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns generated by this engineer.
        
        Returns:
            List of feature column names
        """
        base_features = [
            'home_score_diff', 'away_score_diff', 'total_score', 'home_win', 'away_win',
            'overtime_indicator', 'game_margin', 'high_scoring_game', 'close_game'
        ]
        
        efficiency_features = [
            'home_adj_o', 'home_adj_d', 'home_tempo', 'away_adj_o', 'away_adj_d', 'away_tempo',
            'offensive_efficiency_diff', 'defensive_efficiency_diff', 'tempo_diff',
            'home_efficiency_rating', 'away_efficiency_rating', 'efficiency_rating_diff',
            'predicted_pace'
        ]
        
        travel_features = [
            'home_days_rest', 'away_days_rest', 'home_back_to_back', 'away_back_to_back',
            'rest_advantage', 'home_travel_distance', 'away_travel_distance',
            'home_travel_fatigue', 'away_travel_fatigue', 'home_altitude', 'away_altitude',
            'altitude_difference', 'high_altitude_game'
        ]
        
        conference_features = [
            'home_conference', 'away_conference', 'conference_game', 'home_power_conference',
            'away_power_conference', 'conference_strength_diff', 'home_division', 'away_division',
            'division_game'
        ]
        
        rolling_features = []
        for window in self.rolling_windows:
            rolling_features.extend([
                f'home_rolling_score_{window}', f'home_rolling_conceded_{window}',
                f'home_rolling_margin_{window}', f'home_rolling_win_rate_{window}',
                f'away_rolling_score_{window}', f'away_rolling_conceded_{window}',
                f'away_rolling_margin_{window}', f'away_rolling_win_rate_{window}',
                f'rolling_score_diff_{window}', f'rolling_conceded_diff_{window}',
                f'rolling_win_rate_diff_{window}'
            ])
        
        return base_features + efficiency_features + travel_features + conference_features + rolling_features


def create_team_feature_engineer(config_path: str = "config.yaml") -> TeamFeatureEngineer:
    """Create and return a team feature engineer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        TeamFeatureEngineer instance
    """
    config = ConfigManager(config_path)
    return TeamFeatureEngineer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the team feature engineer
    try:
        engineer = create_team_feature_engineer()
        
        # Create sample data
        sample_games = pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(10)],
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'home_team': ['Duke', 'Kentucky', 'Duke', 'Kansas', 'Michigan State'] * 2,
            'away_team': ['North Carolina', 'Kansas', 'Michigan State', 'Duke', 'Kentucky'] * 2,
            'home_score': [85, 78, 92, 88, 76, 82, 79, 91, 84, 87],
            'away_score': [78, 82, 88, 85, 79, 85, 82, 88, 81, 84],
            'overtime': [False, False, False, False, False, False, False, False, False, False]
        })
        
        # Compute features
        features = engineer.compute_team_context(sample_games)
        
        print(f"Generated {len(features)} feature rows")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Sample features:\n{features.head()}")
        
    except Exception as e:
        print(f"Error: {e}")