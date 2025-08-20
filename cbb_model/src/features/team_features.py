import pandas as pd
import numpy as np

class TeamFeatures:
    def __init__(self):
        pass
    
    def compute_team_efficiency(self, df):
        df = df.copy()
        
        # Simulate offensive/defensive efficiency
        np.random.seed(42)
        df['team_offensive_efficiency'] = np.random.normal(110, 15, len(df))
        df['team_defensive_efficiency'] = np.random.normal(100, 12, len(df))
        df['team_pace'] = np.random.normal(70, 8, len(df))
        
        return df
    
    def compute_home_away_splits(self, df):
        df = df.copy()
        
        # Simulate home/away performance
        np.random.seed(42)
        df['team_home_win_pct'] = np.random.uniform(0.4, 0.8, len(df))
        df['team_away_win_pct'] = np.random.uniform(0.3, 0.7, len(df))
        
        return df
    
    def compute_consistency(self, df):
        df = df.copy()
        
        # Simulate rolling averages
        np.random.seed(42)
        df['team_scoring_consistency_3g'] = np.random.normal(0, 1, len(df))
        df['team_scoring_consistency_10g'] = np.random.normal(0, 1, len(df))
        
        return df
    
    def transform(self, df):
        df = self.compute_team_efficiency(df)
        df = self.compute_home_away_splits(df)
        df = self.compute_consistency(df)
        return df
