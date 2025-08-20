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


class MarketFeatures:
    def __init__(self):
        pass
    
    def compute_line_movement(self, df):
        """
        Compute line movement features:
        - Line Movement = CloseOdds - OpenOdds
        - Movement Magnitude
        - Movement Direction
        """
        df = df.copy()
        
        # Ensure we have odds columns
        required_cols = ['open_spread', 'close_spread', 'open_total', 'close_total']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            # Generate sample odds data if missing
            np.random.seed(42)
            df['open_spread'] = np.random.normal(0, 10, len(df))
            df['close_spread'] = df['open_spread'] + np.random.normal(0, 2, len(df))
            df['open_total'] = np.random.normal(140, 20, len(df))
            df['close_total'] = df['open_total'] + np.random.normal(0, 3, len(df))
        
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
        
        return df
    
    def compute_implied_prob(self, df):
        """
        Compute implied probability features:
        - Implied Probability from moneyline odds
        - Market Edge
        """
        df = df.copy()
        
        # Ensure we have moneyline columns
        if 'open_moneyline' not in df.columns or 'close_moneyline' not in df.columns:
            # Generate sample moneyline data
            np.random.seed(42)
            df['open_moneyline'] = np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], len(df))
            df['close_moneyline'] = df['open_moneyline'] + np.random.choice([-20, -10, 0, 10, 20], len(df))
        
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
        - Line Stability
        - Public vs Sharp Money
        """
        df = df.copy()
        
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
        
        return df
    
    def compute_closing_line_value(self, df):
        """
        Compute Closing Line Value (CLV) features:
        - CLV = Model Prediction - Market Close
        - CLV Overlay Thresholds
        """
        df = df.copy()
        
        # Simulate model predictions (in production, these come from ML models)
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
        
        return df
    
    def transform(self, df):
        """
        Apply all market feature transformations
        """
        df = self.compute_line_movement(df)
        df = self.compute_implied_prob(df)
        df = self.compute_market_efficiency(df)
        df = self.compute_closing_line_value(df)
        return df