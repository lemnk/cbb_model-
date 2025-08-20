"""
Player Features for NCAA CBB Betting ML System.

This module handles player-level feature engineering including:
- Injury flags and availability
- Minutes distribution and bench depth
- Star player impact ratings
- Foul trouble indicators
"""

import pandas as pd
import numpy as np
from .feature_utils import ensure_time_order, safe_fill

class PlayerFeatures:
    def __init__(self):
        pass
    
    def compute_injury_flags(self, players_df):
        """
        Compute injury-related features:
        - Binary indicator for injured/missing
        - Injury severity and recency
        """
        df = players_df.copy()
        
        # Ensure proper time ordering to prevent data leakage
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Ensure we have injury indicators
        if 'injured' not in df.columns:
            if 'status' in df.columns:
                df['injured'] = df['status'].str.contains('injured|out|questionable', case=False, na=False).astype(int)
            else:
                # Simulate injury data (in production, use real injury database)
                np.random.seed(42)
                df['injured'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        # Injury severity (if available)
        if 'injury_severity' in df.columns:
            df['injury_severity_score'] = df['injury_severity'].map({
                'questionable': 0.3,
                'doubtful': 0.6,
                'out': 1.0
            }).fillna(0)
        else:
            df['injury_severity_score'] = df['injured']
        
        # Days since injury (if available)
        if 'days_since_injury' in df.columns:
            df['injury_recency'] = np.exp(-df['days_since_injury'] / 7)  # Decay factor
        else:
            df['injury_recency'] = df['injured'] * 0.5
        
        # Overall injury impact
        df['injury_impact'] = df['injury_severity_score'] * df['injury_recency']
        
        # Injury categories
        df['injury_category'] = pd.cut(
            df['injury_impact'],
            bins=[0, 0.1, 0.3, 0.6, 1.0],
            labels=['healthy', 'minor', 'moderate', 'severe', 'out']
        )
        
        return df
    
    def compute_foul_trouble(self, players_df):
        """
        Compute foul trouble metrics:
        - Foul Rate = Fouls / Minutes
        - Projected Minutes Lost = FoulRate * Avg Minutes
        """
        df = players_df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Calculate foul rate
        if 'fouls' in df.columns and 'minutes' in df.columns:
            df['foul_rate'] = df['fouls'] / df['minutes'].clip(lower=1)
        elif 'fouls' in df.columns:
            df['foul_rate'] = df['fouls'] / 40  # Assume 40 minutes
        else:
            # Simulate foul data (in production, use real data)
            np.random.seed(42)
            df['foul_rate'] = np.random.exponential(0.1, len(df))
        
        # Foul trouble indicators
        df['foul_trouble_3'] = (df['foul_rate'] > 0.075).astype(int)  # 3+ fouls in 40 min
        df['foul_trouble_4'] = (df['foul_rate'] > 0.1).astype(int)    # 4+ fouls in 40 min
        df['foul_trouble_5'] = (df['foul_rate'] > 0.125).astype(int)  # 5+ fouls in 40 min
        
        # Projected minutes lost due to fouls
        if 'avg_minutes' in df.columns:
            df['projected_minutes_lost'] = df['foul_rate'] * df['avg_minutes']
        else:
            df['projected_minutes_lost'] = df['foul_rate'] * 30  # Assume 30 avg minutes
        
        # Foul efficiency (lower is better)
        df['foul_efficiency'] = 1 / (1 + df['foul_rate'])
        
        # Foul trouble risk level
        df['foul_risk_level'] = pd.cut(
            df['foul_rate'],
            bins=[0, 0.05, 0.075, 0.1, 0.15, 1.0],
            labels=['low_risk', 'moderate_risk', 'high_risk', 'very_high_risk', 'extreme_risk']
        )
        
        return df
    
    def compute_bench_depth(self, players_df):
        """
        Compute bench depth metrics:
        - Bench Contribution % = Bench Points / Total Points
        - Bench depth quality and utilization
        """
        df = players_df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Determine if player is starter or bench
        if 'is_starter' not in df.columns:
            if 'minutes' in df.columns:
                df['is_starter'] = (df['minutes'] > 25).astype(int)
            elif 'role' in df.columns:
                df['is_starter'] = df['role'].str.contains('starter|starting', case=False, na=False).astype(int)
            else:
                # Simulate starter/bench data (in production, use real data)
                np.random.seed(42)
                df['is_starter'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
        # Calculate bench contribution
        if 'points' in df.columns:
            # Total team points for each game
            df['team_total_points'] = df.groupby('game_id')['points'].transform('sum')
            
            # Bench points for each game
            df['bench_points'] = df.groupby('game_id').apply(
                lambda x: x.loc[x['is_starter'] == 0, 'points'].sum()
            ).reset_index(0, drop=True)
            
            # Bench contribution percentage
            df['bench_contribution_pct'] = df['bench_points'] / df['team_total_points'].clip(lower=1)
            
            # Individual bench player contribution
            df['player_bench_contribution'] = np.where(
                df['is_starter'] == 0,
                df['points'] / df['team_total_points'].clip(lower=1),
                0
            )
        else:
            # Simulate bench contribution (in production, use real data)
            np.random.seed(42)
            df['bench_contribution_pct'] = np.random.uniform(0.2, 0.4, len(df))
            df['player_bench_contribution'] = np.where(
                df['is_starter'] == 0,
                df['bench_contribution_pct'] * np.random.uniform(0.5, 1.5, len(df)),
                0
            )
        
        # Bench depth quality
        df['bench_depth_quality'] = pd.cut(
            df['bench_contribution_pct'],
            bins=[0, 0.2, 0.35, 0.5, 1.0],
            labels=['poor', 'below_average', 'average', 'good']
        )
        
        # Sixth man indicator (highest scoring bench player)
        if 'points' in df.columns:
            df['is_sixth_man'] = df.groupby('game_id').apply(
                lambda x: (x['is_starter'] == 0) & (x['points'] == x.loc[x['is_starter'] == 0, 'points'].max())
            ).reset_index(0, drop=True).astype(int)
        else:
            df['is_sixth_man'] = 0
        
        # Bench utilization rate
        df['bench_utilization_rate'] = np.where(
            df['is_starter'] == 0,
            np.random.uniform(0.6, 1.0, len(df)),  # Simulate bench utilization
            0
        )
        
        return df
    
    def compute_availability_metrics(self, players_df):
        """
        Compute player availability metrics:
        - Minutes availability
        - Rotation depth
        - Substitution patterns
        """
        df = players_df.copy()
        
        # Ensure proper time ordering
        df = ensure_time_order(df, date_col="date", team_col="team")
        
        # Minutes availability
        if 'minutes' in df.columns:
            df['minutes_availability'] = df['minutes'] / 40  # Normalize to 40-minute game
            df['high_minutes_player'] = (df['minutes'] > 30).astype(int)
            df['rotation_player'] = (df['minutes'] > 15).astype(int)
        else:
            # Simulate minutes data (in production, use real data)
            np.random.seed(42)
            df['minutes_availability'] = np.random.uniform(0.2, 1.0, len(df))
            df['high_minutes_player'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
            df['rotation_player'] = np.random.choice([0, 1], len(df), p=[0.4, 0.6])
        
        # Substitution frequency (if available)
        if 'substitutions' in df.columns:
            df['substitution_rate'] = df['substitutions'] / df['minutes'].clip(lower=1)
        else:
            # Simulate substitution data
            np.random.seed(42)
            df['substitution_rate'] = np.random.exponential(0.1, len(df))
        
        # Player role classification
        df['player_role'] = np.where(
            df['is_starter'] == 1,
            'starter',
            np.where(
                df['rotation_player'] == 1,
                'rotation',
                'deep_bench'
            )
        )
        
        return df
    
    def transform(self, df):
        """
        Apply all player feature transformations with proper time ordering.
        """
        # Ensure we have required columns
        df = safe_fill(df, 'date', pd.Timestamp('2024-01-01'))
        df = safe_fill(df, 'team', 'unknown_team')
        df = safe_fill(df, 'game_id', range(1, len(df) + 1))
        
        # Apply transformations in order
        df = self.compute_injury_flags(df)
        df = self.compute_foul_trouble(df)
        df = self.compute_bench_depth(df)
        df = self.compute_availability_metrics(df)
        
        # Final safety check - ensure no NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df = safe_fill(df, col, 0)
        
        return df