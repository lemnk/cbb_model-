"""
Feature Pipeline for NCAA CBB Betting ML System.

This module orchestrates the complete feature engineering pipeline by integrating
all individual feature engineers and creating a unified feature dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

from .team_features import TeamFeatures
from .player_features import PlayerFeatures
from .market_features import MarketFeatures
from .dynamic_features import DynamicFeatures
from .feature_utils import normalize, handle_missing

class FeaturePipeline:
    """
    Orchestrates the complete feature engineering pipeline for CBB betting analysis.
    """
    
    def __init__(self):
        """Initialize the feature pipeline."""
        self.team_features = TeamFeatures()
        self.player_features = PlayerFeatures()
        self.market_features = MarketFeatures()
        self.dynamic_features = DynamicFeatures()
        
        # Feature weights for Game Strength Index
        self.feature_weights = {
            'team_efficiency': 0.35,
            'player_availability': 0.25,
            'dynamic_factors': 0.20,
            'market_signals': 0.20
        }
    
    def load_sample_data(self):
        """
        Load sample data for demonstration purposes.
        In production, this would load from the Phase 1 database.
        """
        # Generate sample games data
        np.random.seed(42)
        n_games = 100
        
        sample_games = pd.DataFrame({
            'game_id': range(1, n_games + 1),
            'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
            'home_team': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], n_games),
            'away_team': np.random.choice(['Team E', 'Team F', 'Team G', 'Team H'], n_games),
            'home_score': np.random.randint(60, 100, n_games),
            'away_score': np.random.randint(60, 100, n_games),
            'home_points_allowed': np.random.randint(60, 100, n_games),
            'away_points_allowed': np.random.randint(60, 100, n_games),
            'home_possessions': np.random.randint(60, 80, n_games),
            'away_possessions': np.random.randint(60, 80, n_games),
            'home_minutes': np.random.randint(200, 240, n_games),
            'away_minutes': np.random.randint(200, 240, n_games),
            'is_home': np.random.choice([0, 1], n_games),
            'team': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H'], n_games),
            'result': np.random.choice(['win', 'loss'], n_games),
            'points': np.random.randint(60, 100, n_games),
            'opponent_points': np.random.randint(60, 100, n_games)
        })
        
        # Generate sample odds data
        sample_odds = pd.DataFrame({
            'game_id': range(1, n_games + 1),
            'open_spread': np.random.normal(0, 10, n_games),
            'close_spread': np.random.normal(0, 10, n_games),
            'open_total': np.random.normal(140, 20, n_games),
            'close_total': np.random.normal(140, 20, n_games),
            'open_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games),
            'close_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games)
        })
        
        # Generate sample player data
        sample_players = pd.DataFrame({
            'game_id': np.repeat(range(1, n_games + 1), 10),  # 10 players per game
            'player_id': range(1, n_games * 10 + 1),
            'team': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D', 'Team E', 'Team F', 'Team G', 'Team H'], n_games * 10),
            'name': [f'Player_{i}' for i in range(1, n_games * 10 + 1)],
            'minutes': np.random.randint(0, 40, n_games * 10),
            'points': np.random.randint(0, 30, n_games * 10),
            'fouls': np.random.randint(0, 5, n_games * 10),
            'injury_status': np.random.choice(['healthy', 'questionable', 'out'], n_games * 10, p=[0.8, 0.15, 0.05])
        })
        
        return sample_games, sample_odds, sample_players
    
    def build_features(self, games_df=None, odds_df=None, players_df=None):
        """
        Build the complete feature set by applying all feature modules.
        
        Args:
            games_df: Games DataFrame (if None, loads sample data)
            odds_df: Odds DataFrame (if None, loads sample data)
            players_df: Players DataFrame (if None, loads sample data)
            
        Returns:
            DataFrame with all engineered features
        """
        # Load data if not provided
        if games_df is None or odds_df is None or players_df is None:
            games_df, odds_df, players_df = self.load_sample_data()
        
        print("Building features...")
        print(f"Input data shapes: Games={games_df.shape}, Odds={odds_df.shape}, Players={players_df.shape}")
        
        # Apply team features
        print("Computing team features...")
        team_features_df = self.team_features.transform(games_df)
        
        # Apply player features
        print("Computing player features...")
        player_features_df = self.player_features.transform(players_df)
        
        # Apply market features
        print("Computing market features...")
        market_features_df = self.market_features.transform(odds_df)
        
        # Apply dynamic features
        print("Computing dynamic features...")
        dynamic_features_df = self.dynamic_features.transform(games_df)
        
        # Merge all features by game_id
        print("Merging features...")
        merged_features = self._merge_features(
            team_features_df, player_features_df, market_features_df, dynamic_features_df
        )
        
        # Handle missing values
        print("Handling missing values...")
        merged_features = handle_missing(merged_features, strategy="zero")
        
        # Compute Game Strength Index
        print("Computing Game Strength Index...")
        merged_features = self.compute_game_index(merged_features)
        
        print(f"Final feature set shape: {merged_features.shape}")
        return merged_features
    
    def _merge_features(self, team_df, player_df, market_df, dynamic_df):
        """
        Merge all feature DataFrames by game_id.
        
        Args:
            team_df: Team features DataFrame
            player_df: Player features DataFrame
            market_df: Market features DataFrame
            dynamic_df: Dynamic features DataFrame
            
        Returns:
            Merged DataFrame with all features
        """
        # Ensure all DataFrames have game_id column
        if 'game_id' not in team_df.columns:
            team_df['game_id'] = range(1, len(team_df) + 1)
        if 'game_id' not in player_df.columns:
            player_df['game_id'] = range(1, len(player_df) + 1)
        if 'game_id' not in market_df.columns:
            market_df['game_id'] = range(1, len(market_df) + 1)
        if 'game_id' not in dynamic_df.columns:
            dynamic_df['game_id'] = range(1, len(dynamic_df) + 1)
        
        # Start with team features as base
        merged = team_df.copy()
        
        # Merge player features (aggregate by game_id)
        if not player_df.empty:
            player_agg = player_df.groupby('game_id').agg({
                'injury_flag': 'sum',
                'foul_rate': 'mean',
                'projected_minutes_lost': 'sum',
                'bench_contribution_pct': 'mean',
                'bench_depth': 'mean'
            }).reset_index()
            merged = merged.merge(player_agg, on='game_id', how='left')
        
        # Merge market features
        if not market_df.empty:
            market_cols = [col for col in market_df.columns if col != 'game_id']
            market_subset = market_df[['game_id'] + market_cols]
            merged = merged.merge(market_subset, on='game_id', how='left')
        
        # Merge dynamic features
        if not dynamic_df.empty:
            dynamic_cols = [col for col in dynamic_df.columns if col != 'game_id']
            dynamic_subset = dynamic_df[['game_id'] + dynamic_cols]
            merged = merged.merge(dynamic_subset, on='game_id', how='left')
        
        return merged
    
    def compute_game_index(self, features_df):
        """
        Compute the Game Strength Index (GSI) using the formula:
        GSI = 0.35*normalize(team_eff) + 0.25*normalize(player_avail) + 
              0.20*normalize(dyn_factors) + 0.20*normalize(market_signals)
        
        Args:
            features_df: DataFrame with all features
            
        Returns:
            DataFrame with GSI column added
        """
        df = features_df.copy()
        
        # Team efficiency component (35%)
        team_cols = [col for col in df.columns if 'team_' in col and df[col].dtype in [np.number]]
        if team_cols:
            team_efficiency = df[team_cols].mean(axis=1)
            team_efficiency_norm = normalize(team_efficiency, method="minmax")
        else:
            team_efficiency_norm = pd.Series(0.5, index=df.index)
        
        # Player availability component (25%)
        player_cols = [col for col in df.columns if any(x in col for x in ['injury', 'foul', 'bench', 'minutes']) and df[col].dtype in [np.number]]
        if player_cols:
            player_availability = df[player_cols].mean(axis=1)
            player_availability_norm = normalize(player_availability, method="minmax")
        else:
            player_availability_norm = pd.Series(0.5, index=df.index)
        
        # Dynamic factors component (20%)
        dynamic_cols = [col for col in df.columns if any(x in col for x in ['streak', 'rest', 'travel', 'altitude']) and df[col].dtype in [np.number]]
        if dynamic_cols:
            dynamic_factors = df[dynamic_cols].mean(axis=1)
            dynamic_factors_norm = normalize(dynamic_factors, method="minmax")
        else:
            dynamic_factors_norm = pd.Series(0.5, index=df.index)
        
        # Market signals component (20%)
        market_cols = [col for col in df.columns if any(x in col for x in ['movement', 'market', 'clv', 'edge']) and df[col].dtype in [np.number]]
        if market_cols:
            market_signals = df[market_cols].mean(axis=1)
            market_signals_norm = normalize(market_signals, method="minmax")
        else:
            market_signals_norm = pd.Series(0.5, index=df.index)
        
        # Compute weighted GSI
        df['game_strength_index'] = (
            self.feature_weights['team_efficiency'] * team_efficiency_norm +
            self.feature_weights['player_availability'] * player_availability_norm +
            self.feature_weights['dynamic_factors'] * dynamic_factors_norm +
            self.feature_weights['market_signals'] * market_signals_norm
        )
        
        # GSI categories
        df['gsi_category'] = pd.cut(
            df['game_strength_index'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['very_low', 'low', 'moderate', 'high', 'very_high']
        )
        
        return df
    
    def save_features(self, features_df, output_dir="data"):
        """
        Save the engineered features to CSV file.
        
        Args:
            features_df: DataFrame with features
            output_dir: Output directory for saving features
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        filename = f"features_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        features_df.to_csv(filepath, index=False)
        print(f"Features saved to: {filepath}")
        
        return filepath
    
    def print_summary(self, features_df):
        """
        Print a summary of the engineered features.
        
        Args:
            features_df: DataFrame with features
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING SUMMARY")
        print("="*60)
        
        print(f"Total features: {len(features_df.columns)}")
        print(f"Total games: {len(features_df)}")
        
        # Feature categories
        team_features = [col for col in features_df.columns if 'team_' in col]
        player_features = [col for col in features_df.columns if any(x in col for x in ['injury', 'foul', 'bench', 'minutes'])]
        market_features = [col for col in features_df.columns if any(x in col for x in ['movement', 'market', 'clv', 'edge'])]
        dynamic_features = [col for col in features_df.columns if any(x in col for x in ['streak', 'rest', 'travel', 'altitude'])]
        
        print(f"\nFeature breakdown:")
        print(f"  Team features: {len(team_features)}")
        print(f"  Player features: {len(player_features)}")
        print(f"  Market features: {len(market_features)}")
        print(f"  Dynamic features: {len(dynamic_features)}")
        
        # GSI statistics
        if 'game_strength_index' in features_df.columns:
            gsi_stats = features_df['game_strength_index'].describe()
            print(f"\nGame Strength Index statistics:")
            print(f"  Mean: {gsi_stats['mean']:.3f}")
            print(f"  Std: {gsi_stats['std']:.3f}")
            print(f"  Min: {gsi_stats['min']:.3f}")
            print(f"  Max: {gsi_stats['max']:.3f}")
        
        # Sample features
        print(f"\nSample features (first 5 columns):")
        sample_cols = features_df.columns[:5].tolist()
        for col in sample_cols:
            print(f"  {col}")
        
        print("="*60)


if __name__ == "__main__":
    # Demo the feature pipeline
    pipeline = FeaturePipeline()
    
    # Build features
    features = pipeline.build_features()
    
    # Print summary
    pipeline.print_summary(features)
    
    # Save features
    output_file = pipeline.save_features(features)
    
    print(f"\nFeature engineering complete! Features saved to: {output_file}")
    print(f"Feature set shape: {features.shape}")
    print(f"Sample features:\n{features.head()}")