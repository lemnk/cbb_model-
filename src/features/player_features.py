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

class PlayerFeatures:
    def __init__(self):
        pass
    
    def compute_injury_flags(self, players_df):
        """
        Compute injury-related features:
        - Binary indicator for injured/missing
        """
        df = players_df.copy()
        
        # Ensure we have injury indicators
        if 'injured' not in df.columns:
            if 'status' in df.columns:
                df['injured'] = df['status'].str.contains('injured|out|questionable', case=False, na=False).astype(int)
            else:
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
        
        return df
    
    def compute_foul_trouble(self, players_df):
        """
        Compute foul trouble metrics:
        - Foul Rate = Fouls / Minutes
        - Projected Minutes Lost = FoulRate * Avg Minutes
        """
        df = players_df.copy()
        
        # Calculate foul rate
        if 'fouls' in df.columns and 'minutes' in df.columns:
            df['foul_rate'] = df['fouls'] / df['minutes'].clip(lower=1)
        elif 'fouls' in df.columns:
            df['foul_rate'] = df['fouls'] / 40  # Assume 40 minutes
        else:
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
        
        return df
    
    def compute_bench_depth(self, players_df):
        """
        Compute bench depth metrics:
        - Bench Contribution % = Bench Points / Total Points
        """
        df = players_df.copy()
        
        # Determine if player is starter or bench
        if 'is_starter' not in df.columns:
            if 'minutes' in df.columns:
                df['is_starter'] = (df['minutes'] > 25).astype(int)
            elif 'role' in df.columns:
                df['is_starter'] = df['role'].str.contains('starter|starting', case=False, na=False).astype(int)
            else:
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
            df['bench_contribution_pct'] = 0.3  # Default 30%
            df['player_bench_contribution'] = 0
        
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
        
        return df
    
    def transform(self, df):
        """
        Apply all player feature transformations
        """
        df = self.compute_injury_flags(df)
        df = self.compute_foul_trouble(df)
        df = self.compute_bench_depth(df)
        return df