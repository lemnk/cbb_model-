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
from datetime import datetime, timedelta
from .feature_utils import ensure_time_order, safe_fill


class MarketFeatures:
    def __init__(self):
        pass
    
    def compute_line_movement(self, df):
        """
        Compute line movement features:
        - Line Movement = CloseOdds - OpenOdds
        - Movement Magnitude and Direction
        """
        df = df.copy()
        
        # Ensure proper time ordering to prevent data leakage
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Ensure we have odds columns
        required_cols = ['open_spread', 'close_spread', 'open_total', 'close_total']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Generate sample odds data if missing (in production, use real data)
            np.random.seed(42)
            df['open_spread'] = np.random.normal(0, 10, len(df))
            df['close_spread'] = df['open_spread'] + np.random.normal(0, 2, len(df))
            df['open_total'] = np.random.normal(140, 20, len(df))
            df['close_total'] = df['open_total'] + np.random.normal(0, 3, len(df))
        
        # Validate that closing odds are not post-game (leakage prevention)
        if 'game_date' in df.columns and 'market_timestamp' in df.columns:
            # Ensure market data is from before game start
            df = df[df['market_timestamp'] <= df['game_date']]
        
        # Spread movement
        df['spread_movement'] = df['close_spread'] - df['open_spread']
        df['spread_movement_magnitude'] = np.abs(df['spread_movement'])
        df['spread_movement_direction'] = np.sign(df['spread_movement'])
        
        # Total movement
        df['total_movement'] = df['close_total'] - df['open_total']
        df['total_movement_magnitude'] = np.abs(df['total_movement'])
        df['total_movement_direction'] = np.sign(df['total_movement'])
        
        # Movement categories
        df['spread_movement_category'] = pd.cut(
            df['spread_movement_magnitude'],
            bins=[0, 1, 2, 3, 5, 10],
            labels=['no_movement', 'minimal', 'small', 'moderate', 'large', 'extreme']
        )
        
        df['total_movement_category'] = pd.cut(
            df['total_movement_magnitude'],
            bins=[0, 1, 2, 3, 5, 10],
            labels=['no_movement', 'minimal', 'small', 'moderate', 'large', 'extreme']
        )
        
        # Sharp money indicators (large movements)
        df['sharp_spread_movement'] = (df['spread_movement_magnitude'] > 3).astype(int)
        df['sharp_total_movement'] = (df['total_movement_magnitude'] > 3).astype(int)
        
        # Movement consistency
        df['movement_consistency'] = (
            (df['spread_movement_direction'] == df['total_movement_direction']).astype(int)
        )
        
        # Movement velocity (movement per hour if timestamp available)
        if 'market_timestamp' in df.columns and 'open_timestamp' in df.columns:
            df['open_timestamp'] = pd.to_datetime(df['open_timestamp'])
            df['market_timestamp'] = pd.to_datetime(df['market_timestamp'])
            time_diff = (df['market_timestamp'] - df['open_timestamp']).dt.total_seconds() / 3600
            df['movement_velocity'] = df['spread_movement_magnitude'] / time_diff.clip(lower=0.1)
        else:
            df['movement_velocity'] = 0
        
        return df
    
    def compute_implied_prob(self, df):
        """
        Compute implied probability features:
        - Implied Probability from moneyline odds
        - Market Edge and Value Indicators
        """
        df = df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Ensure we have moneyline columns
        if 'open_moneyline' not in df.columns or 'close_moneyline' not in df.columns:
            # Generate sample moneyline data (in production, use real data)
            np.random.seed(42)
            df['open_moneyline'] = np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], len(df))
            df['close_moneyline'] = df['open_moneyline'] + np.random.choice([-20, -10, 0, 10, 20], len(df))
        
        # Validate that closing odds are pre-game
        if 'game_date' in df.columns and 'market_timestamp' in df.columns:
            df = df[df['market_timestamp'] <= df['game_date']]
        
        # Convert moneyline to implied probability
        df['open_implied_prob'] = self._moneyline_to_probability(df['open_moneyline'])
        df['close_implied_prob'] = self._moneyline_to_probability(df['close_moneyline'])
        
        # Implied probability movement
        df['implied_prob_movement'] = df['close_implied_prob'] - df['open_implied_prob']
        df['implied_prob_movement_magnitude'] = np.abs(df['implied_prob_movement'])
        
        # Market edge (difference between open and close)
        df['market_edge'] = df['implied_prob_movement']
        
        # Edge categories
        df['edge_category'] = pd.cut(
            df['market_edge'],
            bins=[-0.1, -0.05, -0.02, 0, 0.02, 0.05, 0.1],
            labels=['large_negative', 'moderate_negative', 'small_negative', 'neutral', 'small_positive', 'moderate_positive', 'large_positive']
        )
        
        # Value bet indicators
        df['value_bet_positive'] = (df['market_edge'] > 0.02).astype(int)
        df['value_bet_negative'] = (df['market_edge'] < -0.02).astype(int)
        
        # Market efficiency indicators
        df['market_efficient_prob'] = (df['implied_prob_movement_magnitude'] < 0.01).astype(int)
        df['market_inefficient_prob'] = (df['implied_prob_movement_magnitude'] > 0.05).astype(int)
        
        return df
    
    def _moneyline_to_probability(self, moneyline):
        """Convert moneyline odds to implied probability."""
        if moneyline > 0:
            return 100 / (moneyline + 100)
        else:
            return abs(moneyline) / (abs(moneyline) + 100)
    
    def compute_market_efficiency(self, df):
        """
        Compute market efficiency features:
        - Market Efficiency Score
        - Line Stability and Confidence
        """
        df = df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Market efficiency score (lower = more efficient)
        df['market_efficiency_score'] = (
            df['spread_movement_magnitude'] + 
            df['total_movement_magnitude'] + 
            df['implied_prob_movement_magnitude'] * 100
        )
        
        # Line stability (inverse of movement)
        df['line_stability'] = 1 / (1 + df['market_efficiency_score'])
        
        # Public vs Sharp money indicators
        df['public_money_indicator'] = (
            (df['spread_movement_magnitude'] < 1) & 
            (df['total_movement_magnitude'] < 1)
        ).astype(int)
        
        df['sharp_money_indicator'] = (
            (df['spread_movement_magnitude'] > 2) | 
            (df['total_movement_magnitude'] > 2)
        ).astype(int)
        
        # Market confidence (inverse of movement)
        df['market_confidence'] = 1 - (df['market_efficiency_score'] / 20)  # Normalize to 0-1
        df['market_confidence'] = df['market_confidence'].clip(0, 1)
        
        # Efficiency categories
        df['efficiency_category'] = pd.cut(
            df['market_efficiency_score'],
            bins=[0, 5, 10, 15, 20, 50],
            labels=['very_efficient', 'efficient', 'moderate', 'inefficient', 'very_inefficient', 'extremely_inefficient']
        )
        
        # Market timing indicators
        df['early_market_movement'] = (
            (df['spread_movement_magnitude'] > 1) | 
            (df['total_movement_magnitude'] > 1)
        ).astype(int)
        
        df['late_market_movement'] = (
            (df['spread_movement_magnitude'] > 2) | 
            (df['total_movement_magnitude'] > 2)
        ).astype(int)
        
        # Market volatility score
        df['market_volatility'] = (
            df['spread_movement_magnitude'] + 
            df['total_movement_magnitude']
        ) / 20  # Normalize to 0-1
        
        return df
    
    def compute_closing_line_value(self, df):
        """
        Compute Closing Line Value (CLV) features:
        - CLV = Model Prediction - Market Close
        - CLV Overlay Thresholds and Confidence
        """
        df = df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Validate that we're only using pre-game data
        if 'game_date' in df.columns and 'market_timestamp' in df.columns:
            df = df[df['market_timestamp'] <= df['game_date']]
        
        # Simulate model predictions (in production, these come from ML models)
        # IMPORTANT: These predictions must only use pre-game information
        np.random.seed(42)
        df['model_spread_prediction'] = np.random.normal(0, 8, len(df))
        df['model_total_prediction'] = np.random.normal(140, 15, len(df))
        
        # Closing Line Value for spread
        df['clv_spread'] = df['model_spread_prediction'] - df['close_spread']
        df['clv_spread_magnitude'] = np.abs(df['clv_spread'])
        
        # Closing Line Value for total
        df['clv_total'] = df['model_total_prediction'] - df['close_total']
        df['clv_total_magnitude'] = np.abs(df['clv_total'])
        
        # CLV categories
        df['clv_spread_category'] = pd.cut(
            df['clv_spread_magnitude'],
            bins=[0, 1, 2, 3, 5, 10],
            labels=['no_edge', 'minimal_edge', 'small_edge', 'moderate_edge', 'large_edge', 'extreme_edge']
        )
        
        df['clv_total_category'] = pd.cut(
            df['clv_total_magnitude'],
            bins=[0, 1, 2, 3, 5, 10],
            labels=['no_edge', 'minimal_edge', 'small_edge', 'moderate_edge', 'large_edge', 'extreme_edge']
        )
        
        # CLV overlay thresholds
        df['clv_overlay_spread'] = (df['clv_spread_magnitude'] > 2).astype(int)
        df['clv_overlay_total'] = (df['clv_total_magnitude'] > 2).astype(int)
        
        # Combined CLV score
        df['clv_combined_score'] = (
            df['clv_spread_magnitude'] + 
            df['clv_total_magnitude']
        )
        
        # CLV confidence
        df['clv_confidence'] = pd.cut(
            df['clv_combined_score'],
            bins=[0, 2, 4, 6, 8, 20],
            labels=['no_confidence', 'low_confidence', 'moderate_confidence', 'high_confidence', 'very_high_confidence', 'extreme_confidence']
        )
        
        # CLV edge direction
        df['clv_edge_direction'] = np.where(
            df['clv_spread_magnitude'] > df['clv_total_magnitude'],
            np.where(df['clv_spread'] > 0, 'spread_positive', 'spread_negative'),
            np.where(df['clv_total'] > 0, 'total_positive', 'total_negative')
        )
        
        return df
    
    def transform(self, df):
        """
        Apply all market feature transformations with leakage prevention.
        """
        # Ensure we have required columns
        df = safe_fill(df, 'date', pd.Timestamp('2024-01-01'))
        df = safe_fill(df, 'team', 'unknown_team')
        df = safe_fill(df, 'game_date', pd.Timestamp('2024-01-01'))
        df = safe_fill(df, 'market_timestamp', pd.Timestamp('2024-01-01'))
        
        # Apply transformations in order
        df = self.compute_line_movement(df)
        df = self.compute_implied_prob(df)
        df = self.compute_market_efficiency(df)
        df = self.compute_closing_line_value(df)
        
        # Final safety check - ensure no NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = safe_fill(df, col, 0)
        
        return df