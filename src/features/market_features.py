"""
Market Features for NCAA CBB Betting ML System.

This module handles market-level feature engineering including:
- Opening vs closing line analysis
- Line movement magnitude and direction
- Closing line value (CLV) calculations
- Market efficiency indicators
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta


class MarketFeatures:
    """Engineers market-level features for CBB betting analysis."""
    
    def __init__(self, db_engine):
        """Initialize the market feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        self.engine = db_engine
    
    def transform(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """
        Market-level features:
        - Opening vs closing line
        - Line movement magnitude
        - Closing line value (team vs market close)
        
        Args:
            odds_df: DataFrame with odds data
            
        Returns:
            DataFrame with market features prefixed with 'market_'
        """
        if odds_df.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original
        features_df = odds_df.copy()
        
        # Add line movement features
        features_df = self._add_line_movement_features(features_df)
        
        # Add implied probability features
        features_df = self._add_implied_probability_features(features_df)
        
        # Add market efficiency features
        features_df = self._add_market_efficiency_features(features_df)
        
        # Add CLV features
        features_df = self._add_clv_features(features_df)
        
        # Add sportsbook comparison features
        features_df = self._add_sportsbook_features(features_df)
        
        # Add timing features
        features_df = self._add_timing_features(features_df)
        
        # Add volatility features
        features_df = self._add_volatility_features(features_df)
        
        return features_df
    
    def _add_line_movement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add line movement and drift features."""
        # Line movement calculations
        df['market_spread_movement'] = df['close_spread'] - df['open_spread']
        df['market_total_movement'] = df['close_total'] - df['open_total']
        
        # Line movement magnitude (absolute value)
        df['market_spread_movement_magnitude'] = abs(df['market_spread_movement'])
        df['market_total_movement_magnitude'] = abs(df['market_total_movement'])
        
        # Line movement direction indicators
        df['market_spread_movement_direction'] = np.where(df['market_spread_movement'] > 0, 1, 
                                                        np.where(df['market_spread_movement'] < 0, -1, 0))
        df['market_total_movement_direction'] = np.where(df['market_total_movement'] > 0, 1,
                                                       np.where(df['market_total_movement'] < 0, -1, 0))
        
        # Line movement percentage
        df['market_spread_movement_pct'] = (df['market_spread_movement'] / df['open_spread'].abs()).clip(-2, 2)
        df['market_total_movement_pct'] = (df['market_total_movement'] / df['open_total']).clip(-0.5, 0.5)
        
        # Significant line movement indicators
        df['market_significant_spread_movement'] = (df['market_spread_movement_magnitude'] > 2).astype(int)
        df['market_significant_total_movement'] = (df['market_total_movement_magnitude'] > 5).astype(int)
        
        # Line movement categories
        df['market_spread_movement_category'] = pd.cut(
            df['market_spread_movement_magnitude'],
            bins=[0, 1, 2, 5, 10, 100],
            labels=['minimal', 'small', 'moderate', 'large', 'extreme']
        )
        
        df['market_total_movement_category'] = pd.cut(
            df['market_total_movement_magnitude'],
            bins=[0, 2, 5, 10, 20, 100],
            labels=['minimal', 'small', 'moderate', 'large', 'extreme']
        )
        
        return df
    
    def _add_implied_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add implied probability and edge features."""
        # Convert moneyline to implied probability
        df['market_home_open_prob'] = self._moneyline_to_probability(df['open_ml_home'])
        df['market_away_open_prob'] = self._moneyline_to_probability(df['open_ml_away'])
        df['market_home_close_prob'] = self._moneyline_to_probability(df['close_ml_home'])
        df['market_away_close_prob'] = self._moneyline_to_probability(df['close_ml_away'])
        
        # Implied probability movement
        df['market_home_prob_movement'] = df['market_home_close_prob'] - df['market_home_open_prob']
        df['market_away_prob_movement'] = df['market_away_close_prob'] - df['market_away_open_prob']
        
        # Implied probability differential
        df['market_open_prob_diff'] = df['market_home_open_prob'] - df['market_away_open_prob']
        df['market_close_prob_diff'] = df['market_home_close_prob'] - df['market_away_close_prob']
        
        # Probability movement magnitude
        df['market_home_prob_movement_magnitude'] = abs(df['market_home_prob_movement'])
        df['market_away_prob_movement_magnitude'] = abs(df['market_away_prob_movement'])
        
        # Probability movement direction
        df['market_home_prob_movement_direction'] = np.where(df['market_home_prob_movement'] > 0, 1,
                                                           np.where(df['market_home_prob_movement'] < 0, -1, 0))
        df['market_away_prob_movement_direction'] = np.where(df['market_away_prob_movement'] > 0, 1,
                                                           np.where(df['market_away_prob_movement'] < 0, -1, 0))
        
        # Market efficiency indicators
        df['market_prob_sum_open'] = df['market_home_open_prob'] + df['market_away_open_prob']
        df['market_prob_sum_close'] = df['market_home_close_prob'] + df['market_away_close_prob']
        
        # Vigorish (overround)
        df['market_vigorish_open'] = df['market_prob_sum_open'] - 1.0
        df['market_vigorish_close'] = df['market_prob_sum_close'] - 1.0
        
        # Vigorish change
        df['market_vigorish_change'] = df['market_vigorish_close'] - df['market_vigorish_open']
        
        return df
    
    def _add_market_efficiency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market efficiency and arbitrage features."""
        # Market efficiency indicators
        df['market_efficient_spread'] = (df['market_spread_movement_magnitude'] <= 1).astype(int)
        df['market_efficient_total'] = (df['market_total_movement_magnitude'] <= 3).astype(int)
        df['market_efficient_moneyline'] = (df['market_home_prob_movement_magnitude'] <= 0.02).astype(int)
        
        # Overall market efficiency score
        df['market_efficiency_score'] = (
            df['market_efficient_spread'] + 
            df['market_efficient_total'] + 
            df['market_efficient_moneyline']
        ) / 3.0
        
        # Market inefficiency indicators
        df['market_inefficient_spread'] = (df['market_spread_movement_magnitude'] > 3).astype(int)
        df['market_inefficient_total'] = (df['market_total_movement_magnitude'] > 8).astype(int)
        df['market_inefficient_moneyline'] = (df['market_home_prob_movement_magnitude'] > 0.05).astype(int)
        
        # Market manipulation indicators
        df['market_manipulation_risk'] = (
            (df['market_spread_movement_magnitude'] > 5) & 
            (df['market_total_movement_magnitude'] > 10)
        ).astype(int)
        
        # Market stability indicators
        df['market_stable_spread'] = (df['market_spread_movement_magnitude'] <= 0.5).astype(int)
        df['market_stable_total'] = (df['market_total_movement_magnitude'] <= 1).astype(int)
        df['market_stable_moneyline'] = (df['market_home_prob_movement_magnitude'] <= 0.01).astype(int)
        
        # Market stability score
        df['market_stability_score'] = (
            df['market_stable_spread'] + 
            df['market_stable_total'] + 
            df['market_stable_moneyline']
        ) / 3.0
        
        return df
    
    def _add_clv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Closing Line Value (CLV) features."""
        # CLV calculations (difference between opening and closing lines)
        df['market_clv_spread'] = df['open_spread'] - df['close_spread']
        df['market_clv_total'] = df['open_total'] - df['close_total']
        df['market_clv_home_prob'] = df['market_home_open_prob'] - df['market_home_close_prob']
        df['market_clv_away_prob'] = df['market_away_open_prob'] - df['market_away_close_prob']
        
        # CLV magnitude
        df['market_clv_spread_magnitude'] = abs(df['market_clv_spread'])
        df['market_clv_total_magnitude'] = abs(df['market_clv_total'])
        df['market_clv_home_prob_magnitude'] = abs(df['market_clv_home_prob'])
        df['market_clv_away_prob_magnitude'] = abs(df['market_clv_away_prob'])
        
        # CLV direction indicators
        df['market_clv_spread_direction'] = np.where(df['market_clv_spread'] > 0, 1,
                                                   np.where(df['market_clv_spread'] < 0, -1, 0))
        df['market_clv_total_direction'] = np.where(df['market_clv_total'] > 0, 1,
                                                  np.where(df['market_clv_total'] < 0, -1, 0))
        
        # CLV advantage indicators
        df['market_clv_home_advantage'] = (df['market_clv_home_prob'] > 0).astype(int)
        df['market_clv_away_advantage'] = (df['market_clv_away_prob'] > 0).astype(int)
        
        # CLV edge magnitude
        df['market_clv_edge_magnitude'] = np.maximum(
            df['market_clv_home_prob_magnitude'],
            df['market_clv_away_prob_magnitude']
        )
        
        # CLV edge direction
        df['market_clv_edge_direction'] = np.where(
            df['market_clv_home_prob_magnitude'] > df['market_clv_away_prob_magnitude'],
            np.where(df['market_clv_home_prob'] > 0, 1, -1),
            np.where(df['market_clv_away_prob'] > 0, -1, 1)
        )
        
        # Significant CLV indicators
        df['market_significant_clv'] = (df['market_clv_edge_magnitude'] > 0.03).astype(int)
        df['market_high_clv'] = (df['market_clv_edge_magnitude'] > 0.05).astype(int)
        
        return df
    
    def _add_sportsbook_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add sportsbook comparison and competition features."""
        # Sportsbook count (if multiple books available)
        df['market_sportsbook_count'] = np.random.randint(1, 5, len(df))  # Placeholder
        
        # Line consensus indicators
        df['market_consensus_spread'] = (df['market_spread_movement_magnitude'] <= 1).astype(int)
        df['market_consensus_total'] = (df['market_total_movement_magnitude'] <= 2).astype(int)
        df['market_consensus_moneyline'] = (df['market_home_prob_movement_magnitude'] <= 0.015).astype(int)
        
        # Line divergence indicators
        df['market_divergent_spread'] = (df['market_spread_movement_magnitude'] > 2).astype(int)
        df['market_divergent_total'] = (df['market_total_movement_magnitude'] > 5).astype(int)
        df['market_divergent_moneyline'] = (df['market_home_prob_movement_magnitude'] > 0.03).astype(int)
        
        # Market competition indicators
        df['market_competitive_spread'] = (df['market_spread_movement_magnitude'] <= 0.5).astype(int)
        df['market_competitive_total'] = (df['market_total_movement_magnitude'] <= 1).astype(int)
        df['market_competitive_moneyline'] = (df['market_home_prob_movement_magnitude'] <= 0.01).astype(int)
        
        # Market competition score
        df['market_competition_score'] = (
            df['market_competitive_spread'] + 
            df['market_competitive_total'] + 
            df['market_competitive_moneyline']
        ) / 3.0
        
        return df
    
    def _add_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add timing and market hours features."""
        # Market timing indicators (placeholder - would come from timestamp data)
        np.random.seed(42)
        
        # Market hours indicators
        df['market_early_betting'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        df['market_late_betting'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])
        df['market_game_day_betting'] = np.random.choice([0, 1], len(df), p=[0.5, 0.5])
        
        # Market timing patterns
        df['market_timing_pattern'] = np.random.choice(
            ['early', 'late', 'game_day', 'mixed'], 
            len(df), 
            p=[0.2, 0.3, 0.3, 0.2]
        )
        
        # Market hours score
        df['market_hours_score'] = (
            df['market_early_betting'] * 0.3 + 
            df['market_late_betting'] * 0.4 + 
            df['market_game_day_betting'] * 0.3
        )
        
        # Market timing efficiency
        df['market_timing_efficiency'] = np.random.uniform(0.5, 1.0, len(df))
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market volatility and risk features."""
        # Market volatility indicators
        df['market_volatile_spread'] = (df['market_spread_movement_magnitude'] > 3).astype(int)
        df['market_volatile_total'] = (df['market_total_movement_magnitude'] > 8).astype(int)
        df['market_volatile_moneyline'] = (df['market_home_prob_movement_magnitude'] > 0.04).astype(int)
        
        # Overall volatility score
        df['market_volatility_score'] = (
            df['market_volatile_spread'] + 
            df['market_volatile_total'] + 
            df['market_volatile_moneyline']
        ) / 3.0
        
        # Market risk indicators
        df['market_high_risk_spread'] = (df['market_spread_movement_magnitude'] > 5).astype(int)
        df['market_high_risk_total'] = (df['market_total_movement_magnitude'] > 12).astype(int)
        df['market_high_risk_moneyline'] = (df['market_home_prob_movement_magnitude'] > 0.06).astype(int)
        
        # Market risk score
        df['market_risk_score'] = (
            df['market_high_risk_spread'] + 
            df['market_high_risk_total'] + 
            df['market_high_risk_moneyline']
        ) / 3.0
        
        # Market stability vs volatility
        df['market_stability_volatility_ratio'] = df['market_stability_score'] / (df['market_volatility_score'] + 0.01)
        
        # Market confidence indicators
        df['market_high_confidence'] = (df['market_stability_score'] > 0.8).astype(int)
        df['market_low_confidence'] = (df['market_volatility_score'] > 0.6).astype(int)
        
        return df
    
    def _moneyline_to_probability(self, moneyline: pd.Series) -> pd.Series:
        """Convert moneyline odds to implied probability.
        
        Args:
            moneyline: Series of moneyline odds
            
        Returns:
            Series of implied probabilities
        """
        def ml_to_prob(ml):
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        
        return moneyline.apply(ml_to_prob)