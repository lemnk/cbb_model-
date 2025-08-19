"""
Market Feature Engineering for CBB Betting ML System.

This module handles market-based feature engineering including:
- Line drift and movement analysis
- Implied probability calculations
- Market efficiency indicators
- CLV (Closing Line Value) analysis
- Sportsbook comparison features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..utils import get_logger, ConfigManager


class MarketFeatureEngineer:
    """Engineers market-based features for CBB betting analysis."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the market feature engineer.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('market_features')
        
        # Get feature configuration
        self.clv_threshold = self.config.get('features.market.clv_threshold', 0.05)
        self.line_movement_threshold = self.config.get('features.market.line_movement_threshold', 0.5)
    
    def compute_market_signals(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Compute market signals from odds data.
        
        Args:
            odds_df: DataFrame with betting odds data
            
        Returns:
            DataFrame with market signal features
        """
        if odds_df.empty:
            self.logger.warning("Empty odds DataFrame provided")
            return pd.DataFrame()
        
        self.logger.info("Computing market signals...")
        
        # Create a copy to avoid modifying original
        features_df = odds_df.copy()
        
        # Line movement features
        features_df = self._add_line_movement_features(features_df)
        
        # Implied probability features
        features_df = self._add_implied_probability_features(features_df)
        
        # Market efficiency features
        features_df = self._add_market_efficiency_features(features_df)
        
        # CLV analysis features
        features_df = self._add_clv_features(features_df)
        
        # Sportsbook comparison features
        features_df = self._add_sportsbook_features(features_df)
        
        # Market timing features
        features_df = self._add_timing_features(features_df)
        
        # Market volatility features
        features_df = self._add_volatility_features(features_df)
        
        self.logger.info(f"Market signals computed: {len(features_df)} rows")
        return features_df
    
    def _add_line_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add line movement features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with line movement features added
        """
        # Ensure required columns exist
        required_columns = [
            'open_moneyline_home', 'close_moneyline_home', 'open_moneyline_away', 'close_moneyline_away',
            'open_spread', 'close_spread', 'open_total', 'close_total'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                # Create placeholder data for demo
                np.random.seed(42)
                if 'moneyline' in col:
                    df[col] = np.random.choice([-110, -105, -100, +100, +105, +110], len(df))
                elif 'spread' in col:
                    df[col] = np.random.uniform(-15, 15, len(df))
                elif 'total' in col:
                    df[col] = np.random.uniform(120, 180, len(df))
        
        # Line drift calculations
        df['moneyline_home_drift'] = df['close_moneyline_home'] - df['open_moneyline_home']
        df['moneyline_away_drift'] = df['close_moneyline_away'] - df['open_moneyline_away']
        df['spread_drift'] = df['close_spread'] - df['open_spread']
        df['total_drift'] = df['close_total'] - df['open_total']
        
        # Absolute line movements
        df['moneyline_home_movement'] = abs(df['moneyline_home_drift'])
        df['moneyline_away_movement'] = abs(df['moneyline_away_drift'])
        df['spread_movement'] = abs(df['spread_drift'])
        df['total_movement'] = abs(df['total_drift'])
        
        # Line movement indicators
        df['significant_moneyline_movement'] = (df['moneyline_home_movement'] > self.line_movement_threshold).astype(int)
        df['significant_spread_movement'] = (df['spread_movement'] > self.line_movement_threshold).astype(int)
        df['significant_total_movement'] = (df['total_movement'] > self.line_movement_threshold).astype(int)
        
        # Line movement direction
        df['moneyline_home_direction'] = np.sign(df['moneyline_home_drift'])
        df['moneyline_away_direction'] = np.sign(df['moneyline_away_drift'])
        df['spread_direction'] = np.sign(df['spread_drift'])
        df['total_direction'] = np.sign(df['total_drift'])
        
        # Line movement speed (placeholder for time-based data)
        df['line_movement_speed'] = np.random.uniform(0.1, 2.0, len(df))
        
        return df
    
    def _add_implied_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add implied probability features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with implied probability features added
        """
        # Convert moneyline odds to implied probabilities
        df['open_prob_home'] = self._moneyline_to_probability(df['open_moneyline_home'])
        df['close_prob_home'] = self._moneyline_to_probability(df['close_moneyline_home'])
        df['open_prob_away'] = self._moneyline_to_probability(df['open_moneyline_away'])
        df['close_prob_away'] = self._moneyline_to_probability(df['close_moneyline_away'])
        
        # Probability drift
        df['prob_home_drift'] = df['close_prob_home'] - df['open_prob_home']
        df['prob_away_drift'] = df['close_prob_away'] - df['open_prob_away']
        
        # Implied probability edge (model vs market)
        # Placeholder for model probabilities - would come from ML model
        np.random.seed(42)
        df['model_prob_home'] = np.random.uniform(0.3, 0.7, len(df))
        df['model_prob_away'] = 1 - df['model_prob_home']
        
        # Probability edge calculations
        df['prob_edge_home'] = df['model_prob_home'] - df['close_prob_home']
        df['prob_edge_away'] = df['model_prob_away'] - df['close_prob_away']
        
        # Significant edge indicators
        df['significant_edge_home'] = (abs(df['prob_edge_home']) > self.clv_threshold).astype(int)
        df['significant_edge_away'] = (abs(df['prob_edge_away']) > self.clv_threshold).astype(int)
        
        # Edge direction
        df['edge_direction_home'] = np.sign(df['prob_edge_home'])
        df['edge_direction_away'] = np.sign(df['prob_edge_away'])
        
        # Market efficiency score
        df['market_efficiency_score'] = 1 - abs(df['prob_edge_home'] + df['prob_edge_away']) / 2
        
        return df
    
    def _add_market_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market efficiency features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with market efficiency features added
        """
        # Market efficiency indicators
        df['efficient_market'] = (df['market_efficiency_score'] > 0.9).astype(int)
        df['inefficient_market'] = (df['market_efficiency_score'] < 0.7).astype(int)
        
        # Line movement efficiency
        df['line_movement_efficiency'] = 1 - (
            abs(df['moneyline_home_drift']) + abs(df['spread_drift']) + abs(df['total_drift'])
        ) / (3 * 100)  # Normalized by typical movement ranges
        
        # Market consensus indicators
        df['market_consensus'] = (
            (df['moneyline_home_movement'] < 0.5) & 
            (df['spread_movement'] < 0.5) & 
            (df['total_movement'] < 0.5)
        ).astype(int)
        
        # Market disagreement indicators
        df['market_disagreement'] = (
            (df['moneyline_home_movement'] > 1.0) | 
            (df['spread_movement'] > 1.0) | 
            (df['total_movement'] > 1.0)
        ).astype(int)
        
        # Market timing efficiency
        df['market_timing_efficiency'] = np.random.uniform(0.6, 1.0, len(df))  # Placeholder
        
        return df
    
    def _add_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add CLV (Closing Line Value) features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with CLV features added
        """
        # CLV calculations
        df['clv_home'] = df['prob_edge_home']
        df['clv_away'] = df['prob_edge_away']
        
        # CLV magnitude
        df['clv_magnitude_home'] = abs(df['clv_home'])
        df['clv_magnitude_away'] = abs(df['clv_away'])
        
        # CLV threshold indicators
        df['clv_above_threshold_home'] = (df['clv_magnitude_home'] > self.clv_threshold).astype(int)
        df['clv_above_threshold_away'] = (df['clv_magnitude_away'] > self.clv_threshold).astype(int)
        
        # CLV quality score
        df['clv_quality_home'] = df['clv_magnitude_home'] * df['market_efficiency_score']
        df['clv_quality_away'] = df['clv_magnitude_away'] * df['market_efficiency_score']
        
        # CLV direction
        df['clv_direction_home'] = np.sign(df['clv_home'])
        df['clv_direction_away'] = np.sign(df['clv_away'])
        
        # CLV overlay opportunities
        df['clv_overlay_home'] = (df['clv_home'] > self.clv_threshold).astype(int)
        df['clv_overlay_away'] = (df['clv_away'] > self.clv_threshold).astype(int)
        
        # CLV underlay opportunities
        df['clv_underlay_home'] = (df['clv_home'] < -self.clv_threshold).astype(int)
        df['clv_underlay_away'] = (df['clv_away'] < -self.clv_threshold).astype(int)
        
        return df
    
    def _add_sportsbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sportsbook comparison features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sportsbook features added
        """
        # Ensure book column exists
        if 'book' not in df.columns:
            np.random.seed(42)
            df['book'] = np.random.choice(['pinnacle', 'draftkings', 'fanduel'], len(df))
        
        # Sportsbook-specific features
        df['pinnacle_odds'] = (df['book'] == 'pinnacle').astype(int)
        df['draftkings_odds'] = (df['book'] == 'draftkings').astype(int)
        df['fanduel_odds'] = (df['book'] == 'fanduel').astype(int)
        
        # Sportsbook comparison (placeholder for multi-book data)
        # In production, this would compare odds across different books
        df['odds_variance'] = np.random.uniform(0, 0.1, len(df))  # Placeholder
        
        # Sportsbook bias indicators
        df['home_team_bias'] = np.random.uniform(-0.05, 0.05, len(df))  # Placeholder
        df['away_team_bias'] = np.random.uniform(-0.05, 0.05, len(df))  # Placeholder
        
        # Sportsbook efficiency
        df['sportsbook_efficiency'] = 1 - df['odds_variance']
        
        return df
    
    def _add_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market timing features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with timing features added
        """
        # Ensure timing columns exist
        if 'open_time' not in df.columns:
            np.random.seed(42)
            df['open_time'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
        
        if 'close_time' not in df.columns:
            df['close_time'] = df['open_time'] + pd.Timedelta(hours=np.random.randint(1, 24, len(df)))
        
        # Convert to datetime if needed
        for col in ['open_time', 'close_time']:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        
        # Time-based features
        df['line_duration'] = (df['close_time'] - df['open_time']).dt.total_seconds() / 3600  # hours
        
        # Market timing indicators
        df['early_line'] = (df['line_duration'] > 24).astype(int)  # Line open more than 24 hours
        df['late_line'] = (df['line_duration'] < 6).astype(int)   # Line open less than 6 hours
        
        # Time of day features
        df['open_hour'] = df['open_time'].dt.hour
        df['close_hour'] = df['close_time'].dt.hour
        
        # Day of week features
        df['open_day_of_week'] = df['open_time'].dt.dayofweek
        df['close_day_of_week'] = df['close_time'].dt.dayofweek
        
        # Weekend indicators
        df['weekend_open'] = (df['open_day_of_week'] >= 5).astype(int)
        df['weekend_close'] = (df['close_day_of_week'] >= 5).astype(int)
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market volatility features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with volatility features added
        """
        # Market volatility indicators
        df['market_volatility'] = (
            df['moneyline_home_movement'] + 
            df['spread_movement'] + 
            df['total_movement']
        ) / 3
        
        # Volatility categories
        df['volatility_level'] = pd.cut(
            df['market_volatility'],
            bins=[0, 0.5, 1.0, 2.0, 10.0],
            labels=['very_low', 'low', 'medium', 'high']
        )
        
        # High volatility indicators
        df['high_volatility'] = (df['market_volatility'] > 1.0).astype(int)
        df['low_volatility'] = (df['market_volatility'] < 0.5).astype(int)
        
        # Volatility vs efficiency relationship
        df['volatility_efficiency_ratio'] = df['market_volatility'] / (df['market_efficiency_score'] + 0.01)
        
        # Market stability score
        df['market_stability_score'] = 1 - df['market_volatility'] / 10  # Normalized
        
        return df
    
    def _moneyline_to_probability(self, moneyline: pd.Series) -> pd.Series:
        """Convert moneyline odds to implied probability.
        
        Args:
            moneyline: Series of moneyline odds
            
        Returns:
            Series of implied probabilities
        """
        def convert_ml_to_prob(ml):
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        
        return moneyline.apply(convert_ml_to_prob)
    
    def calculate_line_drift(self, open_line: float, close_line: float) -> float:
        """Calculate line drift between open and close.
        
        Args:
            open_line: Opening line value
            close_line: Closing line value
            
        Returns:
            Line drift value
        """
        return close_line - open_line
    
    def compute_implied_probability_edge(self, model_prob: float, market_prob: float) -> float:
        """Compute implied probability edge between model and market.
        
        Args:
            model_prob: Model's predicted probability
            market_prob: Market's implied probability
            
        Returns:
            Probability edge (positive = overlay, negative = underlay)
        """
        return model_prob - market_prob
    
    def get_feature_columns(self) -> List[str]:
        """Get list of all feature columns generated by this engineer.
        
        Returns:
            List of feature column names
        """
        line_movement_features = [
            'moneyline_home_drift', 'moneyline_away_drift', 'spread_drift', 'total_drift',
            'moneyline_home_movement', 'moneyline_away_movement', 'spread_movement', 'total_movement',
            'significant_moneyline_movement', 'significant_spread_movement', 'significant_total_movement',
            'moneyline_home_direction', 'moneyline_away_direction', 'spread_direction', 'total_direction',
            'line_movement_speed'
        ]
        
        probability_features = [
            'open_prob_home', 'close_prob_home', 'open_prob_away', 'close_prob_away',
            'prob_home_drift', 'prob_away_drift', 'model_prob_home', 'model_prob_away',
            'prob_edge_home', 'prob_edge_away', 'significant_edge_home', 'significant_edge_away',
            'edge_direction_home', 'edge_direction_away', 'market_efficiency_score'
        ]
        
        efficiency_features = [
            'efficient_market', 'inefficient_market', 'line_movement_efficiency',
            'market_consensus', 'market_disagreement', 'market_timing_efficiency'
        ]
        
        clv_features = [
            'clv_home', 'clv_away', 'clv_magnitude_home', 'clv_magnitude_away',
            'clv_above_threshold_home', 'clv_above_threshold_away', 'clv_quality_home', 'clv_quality_away',
            'clv_direction_home', 'clv_direction_away', 'clv_overlay_home', 'clv_overlay_away',
            'clv_underlay_home', 'clv_underlay_away'
        ]
        
        sportsbook_features = [
            'pinnacle_odds', 'draftkings_odds', 'fanduel_odds', 'odds_variance',
            'home_team_bias', 'away_team_bias', 'sportsbook_efficiency'
        ]
        
        timing_features = [
            'line_duration', 'early_line', 'late_line', 'open_hour', 'close_hour',
            'open_day_of_week', 'close_day_of_week', 'weekend_open', 'weekend_close'
        ]
        
        volatility_features = [
            'market_volatility', 'volatility_level', 'high_volatility', 'low_volatility',
            'volatility_efficiency_ratio', 'market_stability_score'
        ]
        
        return (line_movement_features + probability_features + efficiency_features + 
                clv_features + sportsbook_features + timing_features + volatility_features)


def create_market_feature_engineer(config_path: str = "config.yaml") -> MarketFeatureEngineer:
    """Create and return a market feature engineer instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        MarketFeatureEngineer instance
    """
    config = ConfigManager(config_path)
    return MarketFeatureEngineer(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the market feature engineer
    try:
        engineer = create_market_feature_engineer()
        
        # Create sample odds data
        sample_odds = pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(10)],
            'book': ['pinnacle', 'draftkings'] * 5,
            'open_moneyline_home': [-110, -105, -100, +100, +105] * 2,
            'close_moneyline_home': [-105, -110, -95, +105, +100] * 2,
            'open_moneyline_away': [+100, +105, +110, -110, -105] * 2,
            'close_moneyline_away': [+105, +100, +115, -105, -110] * 2,
            'open_spread': [-2.5, -3.0, -1.5, +1.5, +2.0] * 2,
            'close_spread': [-3.0, -2.5, -2.0, +2.0, +1.5] * 2,
            'open_total': [145.5, 150.0, 148.5, 152.0, 147.5] * 2,
            'close_total': [146.0, 149.5, 149.0, 151.5, 148.0] * 2
        })
        
        # Compute features
        features = engineer.compute_market_signals(sample_odds)
        
        print(f"Generated {len(features)} feature rows")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Sample market features:\n{features[['moneyline_home_drift', 'spread_drift', 'market_efficiency_score']].head()}")
        
    except Exception as e:
        print(f"Error: {e}")