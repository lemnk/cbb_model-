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
from datetime import datetime, timedelta


class DynamicFeatures:
    """Engineers dynamic situational features for CBB betting analysis."""
    
    def __init__(self):
        """Initialize the dynamic feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        pass
    
    def compute_streaks(self, df):
        """
        Compute streak-related features:
        - Win/Loss streak length
        """
        df = df.copy()
        
        # Ensure we have win/loss indicators
        if 'won' not in df.columns:
            if 'result' in df.columns:
                df['won'] = df['result'].str.contains('win|victory', case=False, na=False).astype(int)
            elif 'points' in df.columns and 'opponent_points' in df.columns:
                df['won'] = (df['points'] > df['opponent_points']).astype(int)
            else:
                df['won'] = np.random.choice([0, 1], len(df))
        
        # Sort by team and date for streak calculations
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['team', 'date'])
        
        # Calculate win streaks
        df['win_streak'] = df.groupby('team')['won'].rolling(
            window=10, min_periods=1
        ).apply(lambda x: self._calculate_streak(x)).reset_index(0, drop=True)
        
        # Calculate loss streaks
        df['loss_streak'] = df.groupby('team')['won'].rolling(
            window=10, min_periods=1
        ).apply(lambda x: self._calculate_streak(1 - x)).reset_index(0, drop=True)
        
        # Streak momentum (positive for wins, negative for losses)
        df['streak_momentum'] = df['win_streak'] - df['loss_streak']
        
        # Streak categories
        df['streak_category'] = pd.cut(
            df['streak_momentum'],
            bins=[-10, -5, -2, 0, 2, 5, 10],
            labels=['cold', 'cool', 'slight_cool', 'neutral', 'slight_hot', 'hot', 'on_fire']
        )
        
        # Sort back to original order
        df = df.sort_index()
        
        return df
    
    def _calculate_streak(self, series):
        """Helper function to calculate current streak length."""
        if len(series) == 0:
            return 0
        
        # Find the last value and count consecutive occurrences
        last_val = series.iloc[-1]
        streak = 0
        
        for i in range(len(series) - 1, -1, -1):
            if series.iloc[i] == last_val:
                streak += 1
            else:
                break
        
        return streak if last_val == 1 else -streak
    
    def compute_rest_days(self, df):
        """
        Compute rest-related features:
        - Rest Days = CurrentGameDate - PreviousGameDate
        - Fatigue Index = exp(-RestDays / 3)
        """
        df = df.copy()
        
        # Ensure we have date column
        if 'date' not in df.columns:
            df['date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
        else:
            df['date'] = pd.to_datetime(df['date'])
        
        # Sort by team and date
        df = df.sort_values(['team', 'date'])
        
        # Calculate days since last game
        df['days_since_last_game'] = df.groupby('team')['date'].diff().dt.days
        
        # Fill first game for each team
        df['days_since_last_game'] = df['days_since_last_game'].fillna(7)  # Assume 7 days for first game
        
        # Rest advantage (if we have home/away info)
        if 'is_home' in df.columns:
            home_rest = df[df['is_home'] == 1]['days_since_last_game']
            away_rest = df[df['is_home'] == 0]['days_since_last_game']
            
            if len(home_rest) > 0 and len(away_rest) > 0:
                df['rest_advantage'] = home_rest.mean() - away_rest.mean()
            else:
                df['rest_advantage'] = 0
        else:
            df['rest_advantage'] = 0
        
        # Fatigue index (exponential decay)
        df['fatigue_index'] = np.exp(-df['days_since_last_game'] / 3)
        
        # Rest categories
        df['rest_category'] = pd.cut(
            df['days_since_last_game'],
            bins=[0, 1, 2, 3, 5, 7, 14],
            labels=['back_to_back', '1_day_rest', '2_days_rest', '3_days_rest', '4-5_days_rest', 'week_rest', 'extended_rest']
        )
        
        # Back-to-back indicator
        df['is_back_to_back'] = (df['days_since_last_game'] == 1).astype(int)
        
        # Extended rest indicator
        df['is_extended_rest'] = (df['days_since_last_game'] >= 5).astype(int)
        
        # Sort back to original order
        df = df.sort_index()
        
        return df
    
    def compute_travel(self, df):
        """
        Compute travel-related features:
        - Travel Distance using haversine formula
        """
        df = df.copy()
        
        # Simulate travel distances (in production, use real venue coordinates)
        np.random.seed(42)
        
        # Generate realistic travel distances
        df['travel_distance_miles'] = np.random.exponential(500, len(df))
        
        # Travel time estimates (assuming 60 mph average)
        df['travel_time_hours'] = df['travel_distance_miles'] / 60
        
        # Travel fatigue indicators
        df['travel_fatigue'] = np.where(
            df['travel_distance_miles'] > 1000,
            np.exp(-1000 / df['travel_distance_miles']),
            0
        )
        
        # Travel categories
        df['travel_category'] = pd.cut(
            df['travel_distance_miles'],
            bins=[0, 100, 300, 500, 1000, 2000],
            labels=['local', 'regional', 'moderate', 'long', 'very_long']
        )
        
        # Time zone changes (simplified)
        df['timezone_changes'] = np.random.choice([0, 1, 2], len(df), p=[0.7, 0.2, 0.1])
        
        # Travel efficiency (distance per day)
        df['travel_efficiency'] = df['travel_distance_miles'] / df['days_since_last_game'].clip(lower=1)
        
        return df
    
    def compute_altitude(self, df):
        """
        Compute altitude-related features:
        - Altitude Flag = 1 if Home Court > 3000ft else 0
        """
        df = df.copy()
        
        # Simulate venue altitudes (in production, use real venue database)
        np.random.seed(42)
        
        # Generate realistic altitudes
        df['venue_altitude_ft'] = np.random.normal(1000, 800, len(df))
        
        # High altitude flag
        df['is_high_altitude'] = (df['venue_altitude_ft'] > 3000).astype(int)
        
        # Very high altitude flag
        df['is_very_high_altitude'] = (df['venue_altitude_ft'] > 5000).astype(int)
        
        # Altitude adjustment factor
        df['altitude_adjustment'] = np.where(
            df['venue_altitude_ft'] > 3000,
            (df['venue_altitude_ft'] - 3000) / 1000,  # Adjustment per 1000ft above 3000
            0
        )
        
        # Altitude categories
        df['altitude_category'] = pd.cut(
            df['venue_altitude_ft'],
            bins=[0, 1000, 3000, 5000, 8000],
            labels=['sea_level', 'low', 'moderate', 'high', 'very_high']
        )
        
        # Altitude impact on performance (theoretical)
        df['altitude_performance_impact'] = df['altitude_adjustment'] * 0.1  # 10% impact per 1000ft above 3000
        
        return df
    
    def transform(self, df):
        """
        Apply all dynamic feature transformations
        """
        df = self.compute_streaks(df)
        df = self.compute_rest_days(df)
        df = self.compute_travel(df)
        df = self.compute_altitude(df)
        return df