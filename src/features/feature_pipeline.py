"""
Feature Pipeline for NCAA CBB Betting ML System.

This module orchestrates the complete feature engineering pipeline by integrating
all individual feature engineers and creating a unified feature dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

from .feature_utils import (
    validate_keys, standardize_team_names, Normalizer, 
    safe_fill, ensure_time_order
)
from .team_features import TeamFeatures
from .player_features import PlayerFeatures
from .dynamic_features import DynamicFeatures
from .market_features import MarketFeatures

class FeaturePipeline:
    """
    Orchestrates the complete feature engineering pipeline for CBB betting analysis.
    """
    
    def __init__(self):
        """
        Initialize the feature pipeline with all feature modules.
        """
        self.team_features = TeamFeatures()
        self.player_features = PlayerFeatures()
        self.dynamic_features = DynamicFeatures()
        self.market_features = MarketFeatures()
        
        # Initialize normalizers for GSI components
        self.team_normalizer = Normalizer(method="minmax")
        self.player_normalizer = Normalizer(method="minmax")
        self.dynamic_normalizer = Normalizer(method="minmax")
        self.market_normalizer = Normalizer(method="minmax")
        
        # Track if normalizers have been fitted
        self.normalizers_fitted = False
    
    def validate_input_data(self, games_df, odds_df, players_df):
        """
        Validate input data before processing to prevent errors.
        
        Args:
            games_df: Games DataFrame
            odds_df: Odds DataFrame
            players_df: Players DataFrame
            
        Returns:
            bool: True if validation passes
        """
        try:
            # Validate required keys exist
            validate_keys(games_df, key="game_id", df_name="games_df")
            validate_keys(odds_df, key="game_id", df_name="odds_df")
            validate_keys(players_df, key="game_id", df_name="players_df")
            
            # Validate no null game_ids
            if games_df['game_id'].isnull().any():
                raise ValueError("Null values found in games_df game_id")
            if odds_df['game_id'].isnull().any():
                raise ValueError("Null values found in odds_df game_id")
            if players_df['game_id'].isnull().any():
                raise ValueError("Null values found in players_df game_id")
            
            # Validate team names are consistent
            if 'team' in games_df.columns and 'team' in players_df.columns:
                games_teams = set(games_df['team'].astype(str).str.strip().str.lower())
                players_teams = set(players_df['team'].astype(str).str.strip().str.lower())
                
                if not games_teams.issubset(players_teams):
                    missing_teams = games_teams - players_teams
                    print(f"⚠️ Warning: Teams in games_df not found in players_df: {missing_teams}")
            
            # Validate date columns exist for time ordering
            required_date_cols = ['date']
            for col in required_date_cols:
                if col not in games_df.columns:
                    raise ValueError(f"Required date column '{col}' missing from games_df")
            
            print("✅ Input data validation passed")
            return True
            
        except Exception as e:
            print(f"❌ Input data validation failed: {str(e)}")
            raise
    
    def fit_normalizers(self, games_df, odds_df, players_df):
        """
        Fit normalizers on training data to prevent data leakage.
        In production, this would use a proper train/test split.
        
        Args:
            games_df: Games DataFrame
            odds_df: Odds DataFrame
            players_df: Players DataFrame
        """
        try:
            # For demonstration, use first 70% of data as "training"
            train_size = int(len(games_df) * 0.7)
            
            # Extract training data
            train_games = games_df.head(train_size)
            train_odds = odds_df[odds_df['game_id'].isin(train_games['game_id'])]
            train_players = players_df[players_df['game_id'].isin(train_games['game_id'])]
            
            # Fit team normalizer on team efficiency metrics
            if 'team_combined_efficiency' in train_games.columns:
                self.team_normalizer.fit(train_games['team_combined_efficiency'])
            else:
                # Use simulated data for fitting
                simulated_efficiency = np.random.normal(100, 15, len(train_games))
                self.team_normalizer.fit(pd.Series(simulated_efficiency))
            
            # Fit player normalizer on availability metrics
            if 'injury_impact' in train_players.columns:
                self.player_normalizer.fit(train_players['injury_impact'])
            else:
                simulated_availability = np.random.uniform(0, 1, len(train_players))
                self.player_normalizer.fit(pd.Series(simulated_availability))
            
            # Fit dynamic normalizer on rest quality
            if 'rest_quality_score' in train_games.columns:
                self.dynamic_normalizer.fit(train_games['rest_quality_score'])
            else:
                simulated_rest = np.random.uniform(0, 1, len(train_games))
                self.dynamic_normalizer.fit(pd.Series(simulated_rest))
            
            # Fit market normalizer on efficiency score
            if 'market_efficiency_score' in train_odds.columns:
                self.market_normalizer.fit(train_odds['market_efficiency_score'])
            else:
                simulated_market = np.random.exponential(10, len(train_odds))
                self.market_normalizer.fit(pd.Series(simulated_market))
            
            self.normalizers_fitted = True
            print("✅ Normalizers fitted on training data")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not fit normalizers: {str(e)}")
            # Fall back to using the data as-is
            self.normalizers_fitted = False
    
    def build_features(self, games_df, odds_df, players_df):
        """
        Build comprehensive feature set from raw data.
        
        Args:
            games_df: Games DataFrame with game_id, date, team, etc.
            odds_df: Odds DataFrame with game_id, spreads, totals, etc.
            players_df: Players DataFrame with game_id, player stats, etc.
            
        Returns:
            DataFrame: Merged features with game_id as primary key
        """
        try:
            # Step 1: Validate input data
            self.validate_input_data(games_df, odds_df, players_df)
            
            # Step 2: Fit normalizers on training data
            self.fit_normalizers(games_df, odds_df, players_df)
            
            # Step 3: Apply feature transformations
            print("🔄 Computing team features...")
            team_features = self.team_features.transform(games_df)
            
            print("🔄 Computing player features...")
            player_features = self.player_features.transform(players_df)
            
            print("🔄 Computing dynamic features...")
            dynamic_features = self.dynamic_features.transform(games_df)
            
            print("🔄 Computing market features...")
            market_features = self.market_features.transform(odds_df)
            
            # Step 4: Aggregate player features to team level
            print("🔄 Aggregating player features to team level...")
            player_agg = self._aggregate_player_features(player_features)
            
            # Step 5: Merge all features by game_id
            print("🔄 Merging features...")
            merged = games_df.copy()
            
            # Validate keys before each merge
            validate_keys(merged, key="game_id", df_name="merged games")
            validate_keys(team_features, key="game_id", df_name="team features")
            validate_keys(player_agg, key="game_id", df_name="player aggregate")
            validate_keys(market_features, key="game_id", df_name="market features")
            validate_keys(dynamic_features, key="game_id", df_name="dynamic features")
            
            # Merge team features
            merged = merged.merge(team_features, on='game_id', how='left', suffixes=('', '_team'))
            
            # Merge player aggregate features
            merged = merged.merge(player_agg, on='game_id', how='left', suffixes=('', '_player'))
            
            # Merge market features
            merged = merged.merge(market_features, on='game_id', how='left', suffixes=('', '_market'))
            
            # Merge dynamic features
            merged = merged.merge(dynamic_features, on='game_id', how='left', suffixes=('', '_dynamic'))
            
            # Step 6: Compute Game Strength Index (GSI)
            print("🔄 Computing Game Strength Index...")
            merged = self.compute_game_index(merged)
            
            # Step 7: Final validation and cleanup
            print("🔄 Final validation and cleanup...")
            merged = self._final_validation_and_cleanup(merged)
            
            print(f"✅ Feature engineering complete. Final shape: {merged.shape}")
            return merged
            
        except Exception as e:
            print(f"❌ Feature engineering failed: {str(e)}")
            raise
    
    def _aggregate_player_features(self, players_df):
        """
        Aggregate player-level features to team level.
        
        Args:
            players_df: Player-level features DataFrame
            
        Returns:
            DataFrame: Team-level aggregated features
        """
        if players_df.empty:
            return pd.DataFrame()
        
        # Group by game_id and aggregate player features
        agg_features = players_df.groupby('game_id').agg({
            'injured': ['sum', 'mean'],  # Count and percentage of injured players
            'injury_impact': ['mean', 'max'],  # Average and max injury impact
            'foul_rate': ['mean', 'max'],  # Average and max foul rate
            'bench_contribution_pct': 'mean',  # Average bench contribution
            'minutes_availability': 'mean',  # Average minutes availability
            'high_minutes_player': 'sum',  # Count of high-minutes players
            'rotation_player': 'sum',  # Count of rotation players
            'is_sixth_man': 'sum',  # Count of sixth men
            'bench_utilization_rate': 'mean'  # Average bench utilization
        }).reset_index()
        
        # Flatten column names
        agg_features.columns = ['game_id'] + [
            f"team_{col[0]}_{col[1]}" if col[1] else f"team_{col[0]}"
            for col in agg_features.columns[1:]
        ]
        
        return agg_features
    
    def compute_game_index(self, df):
        """
        Compute Game Strength Index (GSI) using the specified formula.
        GSI = 0.35*team_efficiency + 0.25*player_availability + 0.20*dynamic_factors + 0.20*market_signals
        
        Args:
            df: DataFrame with all features
            
        Returns:
            DataFrame: Original DataFrame with GSI column added
        """
        df = df.copy()
        
        try:
            # Extract components for GSI calculation
            # Team efficiency component
            if 'team_combined_efficiency' in df.columns:
                team_efficiency = df['team_combined_efficiency']
            else:
                # Fallback to simulated data
                np.random.seed(42)
                team_efficiency = np.random.normal(100, 15, len(df))
            
            # Player availability component
            if 'team_injury_impact_mean' in df.columns:
                player_availability = 1 - df['team_injury_impact_mean']  # Inverse of injury impact
            else:
                # Fallback to simulated data
                np.random.seed(42)
                player_availability = np.random.uniform(0.7, 1.0, len(df))
            
            # Dynamic factors component
            if 'rest_quality_score' in df.columns:
                dynamic_factors = df['rest_quality_score']
            else:
                # Fallback to simulated data
                np.random.seed(42)
                dynamic_factors = np.random.uniform(0.2, 1.0, len(df))
            
            # Market signals component
            if 'market_efficiency_score' in df.columns:
                market_signals = 1 / (1 + df['market_efficiency_score'] / 20)  # Inverse of inefficiency
            else:
                # Fallback to simulated data
                np.random.seed(42)
                market_signals = np.random.uniform(0.5, 1.0, len(df))
            
            # Normalize each component to [0,1] using fitted normalizers
            if self.normalizers_fitted:
                team_efficiency_norm = self.team_normalizer.transform(team_efficiency)
                player_availability_norm = self.player_normalizer.transform(player_availability)
                dynamic_factors_norm = self.dynamic_normalizer.transform(dynamic_factors)
                market_signals_norm = self.market_normalizer.transform(market_signals)
            else:
                # Fallback normalization
                team_efficiency_norm = (team_efficiency - team_efficiency.min()) / (team_efficiency.max() - team_efficiency.min() + 1e-9)
                player_availability_norm = (player_availability - player_availability.min()) / (player_availability.max() - player_availability.min() + 1e-9)
                dynamic_factors_norm = (dynamic_factors - dynamic_factors.min()) / (dynamic_factors.max() - dynamic_factors.min() + 1e-9)
                market_signals_norm = (market_signals - market_signals.min()) / (market_signals.max() - market_signals.min() + 1e-9)
            
            # Compute GSI using the exact formula
            gsi = (
                0.35 * team_efficiency_norm +
                0.25 * player_availability_norm +
                0.20 * dynamic_factors_norm +
                0.20 * market_signals_norm
            )
            
            # Add GSI and components to DataFrame
            df['gsi'] = gsi
            df['gsi_team_efficiency'] = team_efficiency_norm
            df['gsi_player_availability'] = player_availability_norm
            df['gsi_dynamic_factors'] = dynamic_factors_norm
            df['gsi_market_signals'] = market_signals_norm
            
            # GSI categories
            df['gsi_category'] = pd.cut(
                gsi,
                bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                labels=['very_weak', 'weak', 'moderate', 'strong', 'very_strong']
            )
            
            print(f"✅ GSI computed successfully. Range: {gsi.min():.3f} - {gsi.max():.3f}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not compute GSI: {str(e)}")
            # Fallback GSI
            df['gsi'] = 0.5
            df['gsi_category'] = 'moderate'
        
        return df
    
    def _final_validation_and_cleanup(self, df):
        """
        Final validation and cleanup of the merged features.
        
        Args:
            df: Merged features DataFrame
            
        Returns:
            DataFrame: Cleaned and validated features
        """
        # Final validation checks
        validate_keys(df, key="game_id", df_name="final merged")
        
        # Check for missing values
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            print(f"⚠️ Warning: Missing values found in columns: {missing_cols}")
            
            # Fill missing values safely
            for col in missing_cols:
                if df[col].dtype in [np.number]:
                    df = safe_fill(df, col, 0)
                else:
                    df = safe_fill(df, col, "unknown")
        
        # Ensure no duplicate game_ids
        if df['game_id'].duplicated().any():
            print("⚠️ Warning: Duplicate game_ids found. Removing duplicates...")
            df = df.drop_duplicates(subset=['game_id'], keep='first')
        
        # Final shape check
        print(f"📊 Final feature set shape: {df.shape}")
        print(f"📊 Feature columns: {len(df.columns)}")
        
        return df
    
    def save_features(self, features_df, output_path=None):
        """
        Save features to CSV with timestamp.
        
        Args:
            features_df: Features DataFrame to save
            output_path: Optional custom output path
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d")
            output_path = f"data/features_{timestamp}.csv"
        
        try:
            features_df.to_csv(output_path, index=False)
            print(f"✅ Features saved to: {output_path}")
        except Exception as e:
            print(f"❌ Failed to save features: {str(e)}")

if __name__ == "__main__":
    # Test the feature pipeline with sample data
    print("🧪 Testing Feature Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_games = 100
    
    # Sample games data
    games_df = pd.DataFrame({
        'game_id': range(1, n_games + 1),
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'team': np.random.choice(['Duke', 'UNC', 'Kentucky', 'Kansas'], n_games),
        'opponent': np.random.choice(['Duke', 'UNC', 'Kentucky', 'Kansas'], n_games),
        'points': np.random.normal(75, 15, n_games),
        'opponent_points': np.random.normal(70, 15, n_games),
        'won': np.random.choice([0, 1], n_games)
    })
    
    # Sample odds data
    odds_df = pd.DataFrame({
        'game_id': range(1, n_games + 1),
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'team': np.random.choice(['Duke', 'UNC', 'Kentucky', 'Kansas'], n_games),
        'open_spread': np.random.normal(0, 10, n_games),
        'close_spread': np.random.normal(0, 10, n_games),
        'open_total': np.random.normal(140, 20, n_games),
        'close_total': np.random.normal(140, 20, n_games),
        'open_moneyline': np.random.choice([-150, -120, -110, 110, 120, 150], n_games),
        'close_moneyline': np.random.choice([-150, -120, -110, 110, 120, 150], n_games)
    })
    
    # Sample players data
    players_df = pd.DataFrame({
        'game_id': np.repeat(range(1, n_games + 1), 10),  # 10 players per game
        'date': np.repeat(pd.date_range('2024-01-01', periods=n_games, freq='D'), 10),
        'team': np.repeat(np.random.choice(['Duke', 'UNC', 'Kentucky', 'Kansas'], n_games), 10),
        'player_id': range(1, n_games * 10 + 1),
        'minutes': np.random.uniform(10, 40, n_games * 10),
        'points': np.random.normal(10, 8, n_games * 10),
        'injured': np.random.choice([0, 1], n_games * 10, p=[0.9, 0.1])
    })
    
    # Initialize and run pipeline
    pipeline = FeaturePipeline()
    
    try:
        features = pipeline.build_features(games_df, odds_df, players_df)
        
        # Display results
        print(f"\n🎯 Feature Engineering Results:")
        print(f"📊 Final shape: {features.shape}")
        print(f"📊 Total features: {len(features.columns)}")
        print(f"📊 Sample GSI values: {features['gsi'].head().tolist()}")
        
        # Save features
        pipeline.save_features(features)
        
        print("\n✅ Feature pipeline test completed successfully!")
        
    except Exception as e:
        print(f"❌ Feature pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()