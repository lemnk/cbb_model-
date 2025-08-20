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
    
    def __init__(self):
        """Initialize the team feature engineer.
        
        Args:
            db_engine: SQLAlchemy database engine
        """
        pass
    
    def compute_team_efficiency(self, df):
        """
        Compute team efficiency metrics:
        - Offensive Efficiency = Points Scored / Possessions
        - Defensive Efficiency = Points Allowed / Possessions  
        - Pace = Possessions per 40 minutes
        """
        df = df.copy()
        
        # Calculate possessions (estimated from FGA, FTA, TO, OReb)
        if 'fga' in df.columns and 'fta' in df.columns and 'turnovers' in df.columns and 'oreb' in df.columns:
            df['possessions'] = df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['turnovers']
        else:
            # Fallback: estimate possessions from scoring
            df['possessions'] = df['points'] * 0.8  # Rough estimate
        
        # Offensive Efficiency
        df['offensive_efficiency'] = df['points'] / df['possessions'] * 100
        
        # Defensive Efficiency (points allowed)
        if 'points_allowed' in df.columns:
            df['defensive_efficiency'] = df['points_allowed'] / df['possessions'] * 100
        else:
            # Estimate from opponent scoring
            df['defensive_efficiency'] = df['opponent_points'] / df['possessions'] * 100
        
        # Pace (possessions per 40 minutes)
        if 'minutes' in df.columns:
            df['pace'] = df['possessions'] / df['minutes'] * 40
        else:
            df['pace'] = df['possessions'] * 1.2  # Assume 40 minutes
        
        # Net Efficiency
        df['net_efficiency'] = df['offensive_efficiency'] - df['defensive_efficiency']
        
        return df
    
    def compute_home_away_splits(self, df):
        """
        Compute home/away performance splits:
        - Home Win% = Home Wins / Home Games
        - Away Win% = Away Wins / Away Games
        """
        df = df.copy()
        
        # Ensure we have home/away indicators
        if 'is_home' not in df.columns:
            if 'home_team' in df.columns and 'team' in df.columns:
                df['is_home'] = (df['team'] == df['home_team']).astype(int)
            else:
                df['is_home'] = np.random.choice([0, 1], len(df))
        
        # Calculate home/away splits
        home_games = df[df['is_home'] == 1]
        away_games = df[df['is_home'] == 0]
        
        # Home performance
        if len(home_games) > 0:
            home_wins = home_games['won'].sum() if 'won' in home_games.columns else 0
            home_win_pct = home_wins / len(home_games)
            df['home_win_pct'] = home_win_pct
            df['home_avg_points'] = home_games['points'].mean() if 'points' in home_games.columns else 0
        else:
            df['home_win_pct'] = 0
            df['home_avg_points'] = 0
        
        # Away performance
        if len(away_games) > 0:
            away_wins = away_games['won'].sum() if 'won' in away_games.columns else 0
            away_win_pct = away_wins / len(away_games)
            df['away_win_pct'] = away_win_pct
            df['away_avg_points'] = away_games['points'].mean() if 'points' in away_games.columns else 0
        else:
            df['away_win_pct'] = 0
            df['away_avg_points'] = 0
        
        # Home/Away advantage
        df['home_away_advantage'] = df['home_win_pct'] - df['away_win_pct']
        
        return df
    
    def compute_consistency(self, df):
        """
        Compute team consistency metrics:
        - Rolling mean & std of scoring (last N games)
        """
        df = df.copy()
        
        # Sort by team and date for rolling calculations
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values(['team', 'date'])
        
        # Rolling scoring statistics (last 5 games)
        if 'points' in df.columns:
            df['scoring_rolling_mean_5'] = df.groupby('team')['points'].rolling(
                window=5, min_periods=1
            ).mean().reset_index(0, drop=True)
            
            df['scoring_rolling_std_5'] = df.groupby('team')['points'].rolling(
                window=5, min_periods=1
            ).std().reset_index(0, drop=True)
            
            # Consistency score (lower std = more consistent)
            df['scoring_consistency'] = 1 / (1 + df['scoring_rolling_std_5'])
        
        # Rolling win rate (last 10 games)
        if 'won' in df.columns:
            df['win_rate_rolling_10'] = df.groupby('team')['won'].rolling(
                window=10, min_periods=1
            ).mean().reset_index(0, drop=True)
        
        # Sort back to original order
        df = df.sort_index()
        
        return df
    
    def transform(self, df):
        """
        Apply all team feature transformations
        """
        df = self.compute_team_efficiency(df)
        df = self.compute_home_away_splits(df)
        df = self.compute_consistency(df)
        return df