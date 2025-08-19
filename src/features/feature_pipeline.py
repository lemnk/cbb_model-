"""
Feature Pipeline for CBB Betting ML System.

This module orchestrates the complete feature engineering pipeline:
- Team context features
- Dynamic game flow features
- Player availability features
- Market efficiency features
- Feature validation and quality assurance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os

from ..utils import get_logger, ConfigManager
from .team_features import TeamFeatureEngineer
from .dynamic_features import DynamicFeatureEngineer
from .player_features import PlayerFeatureEngineer
from .market_features import MarketFeatureEngineer
from .feature_utils import FeatureUtils


class FeaturePipeline:
    """Orchestrates the complete feature engineering pipeline."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the feature pipeline.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('feature_pipeline')
        
        # Initialize feature engineers
        self.team_engineer = TeamFeatureEngineer(config)
        self.dynamic_engineer = DynamicFeatureEngineer(config)
        self.player_engineer = PlayerFeatureEngineer(config)
        self.market_engineer = MarketFeatureEngineer(config)
        self.feature_utils = FeatureUtils(config)
        
        # Get configuration
        self.raw_data_dir = self.config.get('data_collection.raw_data_dir', 'data/raw')
        self.processed_data_dir = self.config.get('data_collection.processed_data_dir', 'data/processed')
        self.backup_dir = self.config.get('data_collection.backup_dir', 'data/backup')
    
    def build_feature_set(
        self, 
        games_df: pd.DataFrame,
        odds_df: pd.DataFrame,
        pbp_df: Optional[pd.DataFrame] = None,
        injury_df: Optional[pd.DataFrame] = None,
        lineup_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Build complete feature set from all available data sources.
        
        Args:
            games_df: DataFrame with games data
            odds_df: DataFrame with betting odds data
            pbp_df: DataFrame with play-by-play data (optional)
            injury_df: DataFrame with player injury data (optional)
            lineup_df: DataFrame with lineup data (optional)
            
        Returns:
            DataFrame with complete feature set
        """
        self.logger.info("Starting feature engineering pipeline...")
        
        if games_df.empty:
            self.logger.error("Games DataFrame is empty, cannot proceed")
            return pd.DataFrame()
        
        # Start with games data as base
        features_df = games_df.copy()
        
        # Step 1: Team context features
        self.logger.info("Step 1: Computing team context features...")
        team_features = self.team_engineer.compute_team_context(games_df)
        features_df = self._merge_features(features_df, team_features, 'game_id')
        
        # Step 2: Dynamic game flow features (if PBP data available)
        if pbp_df is not None and not pbp_df.empty:
            self.logger.info("Step 2: Computing dynamic game flow features...")
            dynamic_features = self.dynamic_engineer.compute_game_flow(pbp_df)
            features_df = self._merge_features(features_df, dynamic_features, 'game_id')
        else:
            self.logger.info("Step 2: Skipping dynamic features (no PBP data)")
        
        # Step 3: Player availability features (if available)
        if injury_df is not None and not injury_df.empty:
            self.logger.info("Step 3: Computing player availability features...")
            player_features = self.player_engineer.compute_player_availability(injury_df, lineup_df)
            
            # Aggregate player features to team level for merging
            if 'team_id' in player_features.columns:
                team_player_features = self._aggregate_player_features_to_team(player_features)
                features_df = self._merge_features(features_df, team_player_features, 'home_team', 'away_team')
            else:
                self.logger.warning("Player features missing team_id, skipping merge")
        else:
            self.logger.info("Step 3: Skipping player features (no injury/lineup data)")
        
        # Step 4: Market features
        if not odds_df.empty:
            self.logger.info("Step 4: Computing market features...")
            market_features = self.market_engineer.compute_market_signals(odds_df)
            features_df = self._merge_features(features_df, market_features, 'game_id')
        else:
            self.logger.info("Step 4: Skipping market features (no odds data)")
        
        # Step 5: Feature engineering and enhancement
        self.logger.info("Step 5: Enhancing features...")
        features_df = self._enhance_features(features_df)
        
        # Step 6: Feature validation and quality checks
        self.logger.info("Step 6: Validating feature set...")
        validation_results = self._validate_feature_set(features_df)
        
        if not validation_results['valid']:
            self.logger.warning("Feature validation issues found:")
            for error in validation_results['errors']:
                self.logger.warning(f"  - {error}")
        
        # Step 7: Feature selection and finalization
        self.logger.info("Step 7: Finalizing feature set...")
        features_df = self._finalize_feature_set(features_df)
        
        self.logger.info(f"Feature engineering pipeline completed: {len(features_df)} rows, {len(features_df.columns)} columns")
        
        return features_df
    
    def _merge_features(
        self, 
        base_df: pd.DataFrame, 
        new_features: pd.DataFrame, 
        *merge_keys: str
    ) -> pd.DataFrame:
        """Merge new features into base DataFrame.
        
        Args:
            base_df: Base DataFrame
            new_features: New features to merge
            *merge_keys: Keys to merge on
            
        Returns:
            Merged DataFrame
        """
        if new_features.empty:
            return base_df
        
        # Handle different merge scenarios
        if len(merge_keys) == 1:
            # Simple merge on single key
            merge_key = merge_keys[0]
            if merge_key in base_df.columns and merge_key in new_features.columns:
                return base_df.merge(new_features, on=merge_key, how='left')
            else:
                self.logger.warning(f"Merge key {merge_key} not found in both DataFrames")
                return base_df
        
        elif len(merge_keys) == 2:
            # Merge on home and away team keys
            home_key, away_key = merge_keys
            
            # For team-level features, we need to handle home/away team mapping
            if 'home_team' in base_df.columns and 'away_team' in base_df.columns:
                # Create separate home and away features
                home_features = new_features.copy()
                away_features = new_features.copy()
                
                # Rename columns to distinguish home/away
                home_features = home_features.rename(columns={
                    col: f'home_{col}' for col in home_features.columns 
                    if col not in [home_key, away_key]
                })
                away_features = away_features.rename(columns={
                    col: f'away_{col}' for col in away_features.columns 
                    if col not in [home_key, away_key]
                })
                
                # Merge home features
                base_df = base_df.merge(home_features, left_on=home_key, right_on=home_key, how='left')
                # Merge away features
                base_df = base_df.merge(away_features, left_on=away_key, right_on=away_key, how='left')
                
                return base_df
        
        return base_df
    
    def _aggregate_player_features_to_team(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate player-level features to team level.
        
        Args:
            player_df: Player features DataFrame
            
        Returns:
            Team-level player features DataFrame
        """
        if 'team_id' not in player_df.columns:
            return pd.DataFrame()
        
        # Aggregate numeric columns by team
        numeric_columns = player_df.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != 'team_id']
        
        team_aggregations = {}
        for col in numeric_columns:
            team_aggregations[col] = ['mean', 'std', 'min', 'max']
        
        team_features = player_df.groupby('team_id').agg(team_aggregations).reset_index()
        
        # Flatten column names
        team_features.columns = ['team_id'] + [
            f'team_{col[0]}_{col[1]}' if col[1] else f'team_{col[0]}'
            for col in team_features.columns[1:]
        ]
        
        return team_features
    
    def _enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance features with additional engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Enhanced DataFrame
        """
        enhanced_df = df.copy()
        
        # Add interaction features
        enhanced_df = self._add_interaction_features(enhanced_df)
        
        # Add polynomial features for key numeric columns
        enhanced_df = self._add_polynomial_features(enhanced_df)
        
        # Add ratio features
        enhanced_df = self._add_ratio_features(enhanced_df)
        
        # Add categorical encodings
        enhanced_df = self._add_categorical_features(enhanced_df)
        
        # Add time-based features
        enhanced_df = self._add_time_features(enhanced_df)
        
        return enhanced_df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between key variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        # Key interaction features for CBB betting
        interactions = [
            ('home_rolling_score_5', 'away_rolling_score_5'),
            ('home_rolling_conceded_5', 'away_rolling_conceded_5'),
            ('home_efficiency_rating', 'away_efficiency_rating'),
            ('home_adj_o', 'away_adj_d'),
            ('home_adj_d', 'away_adj_o'),
            ('spread_drift', 'total_drift'),
            ('moneyline_home_drift', 'spread_drift')
        ]
        
        for col1, col2 in interactions:
            if col1 in df.columns and col2 in df.columns:
                interaction_name = f"{col1}_x_{col2}"
                df[interaction_name] = df[col1] * df[col2]
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for key numeric columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with polynomial features
        """
        # Key columns for polynomial features
        poly_columns = [
            'home_score_diff', 'away_score_diff', 'game_margin', 'total_score',
            'home_efficiency_rating', 'away_efficiency_rating', 'spread_drift'
        ]
        
        for col in poly_columns:
            if col in df.columns:
                # Square features
                df[f'{col}_squared'] = df[col] ** 2
                # Cube features (for some key metrics)
                if col in ['home_score_diff', 'game_margin', 'spread_drift']:
                    df[f'{col}_cubed'] = df[col] ** 3
        
        return df
    
    def _add_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ratio features between related variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with ratio features
        """
        # Efficiency ratios
        if 'home_adj_o' in df.columns and 'home_adj_d' in df.columns:
            df['home_offense_defense_ratio'] = df['home_adj_o'] / df['home_adj_d'].clip(0.1)
        
        if 'away_adj_o' in df.columns and 'away_adj_d' in df.columns:
            df['away_offense_defense_ratio'] = df['away_adj_o'] / df['away_adj_d'].clip(0.1)
        
        # Scoring ratios
        if 'home_rolling_score_5' in df.columns and 'away_rolling_score_5' in df.columns:
            df['scoring_ratio_5'] = df['home_rolling_score_5'] / df['away_rolling_score_5'].clip(0.1)
        
        # Market efficiency ratios
        if 'market_efficiency_score' in df.columns and 'market_volatility' in df.columns:
            df['efficiency_volatility_ratio'] = df['market_efficiency_score'] / df['market_volatility'].clip(0.01)
        
        return df
    
    def _add_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add categorical feature encodings.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical features
        """
        # Conference encodings
        if 'home_conference' in df.columns:
            df['home_conference_encoded'] = pd.Categorical(df['home_conference']).codes
        
        if 'away_conference' in df.columns:
            df['away_conference_encoded'] = pd.Categorical(df['away_conference']).codes
        
        # Division encodings
        if 'home_division' in df.columns:
            df['home_division_encoded'] = pd.Categorical(df['home_division']).codes
        
        if 'away_division' in df.columns:
            df['away_division_encoded'] = pd.Categorical(df['away_division']).codes
        
        # Volatility level encoding
        if 'volatility_level' in df.columns:
            df['volatility_level_encoded'] = pd.Categorical(df['volatility_level']).codes
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with time features
        """
        # Ensure date column exists and is datetime
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'])
            
            # Season features
            df['season'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_year'] = df['date'].dt.dayofyear
            
            # Season progression
            df['season_progression'] = df['day_of_year'] / 365.0
            
            # Weekend indicator
            df['weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            # Holiday season indicator (March Madness)
            df['march_madness'] = (df['month'] == 3).astype(int)
            
            # Regular season vs postseason
            df['postseason'] = (df['month'].isin([3, 4])).astype(int)
        
        return df
    
    def _validate_feature_set(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the complete feature set.
        
        Args:
            df: Feature DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        # Required columns for CBB betting model
        required_columns = [
            'game_id', 'home_team', 'away_team', 'home_score', 'away_score',
            'home_win', 'away_win'
        ]
        
        return self.feature_utils.validate_feature_set(df, required_columns, max_missing_pct=0.3)
    
    def _finalize_feature_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize the feature set for ML model training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Finalized feature DataFrame
        """
        finalized_df = df.copy()
        
        # Remove duplicate columns
        finalized_df = finalized_df.loc[:, ~finalized_df.columns.duplicated()]
        
        # Remove columns with all NaN values
        finalized_df = finalized_df.dropna(axis=1, how='all')
        
        # Fill remaining NaN values with appropriate defaults
        finalized_df = self._fill_missing_values(finalized_df)
        
        # Ensure all numeric columns are float
        numeric_columns = finalized_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            finalized_df[col] = pd.to_numeric(finalized_df[col], errors='coerce')
        
        # Sort by date if available
        if 'date' in finalized_df.columns:
            finalized_df = finalized_df.sort_values('date')
        
        return finalized_df
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with filled missing values
        """
        filled_df = df.copy()
        
        # Fill numeric columns with 0 or mean
        numeric_columns = filled_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if filled_df[col].isnull().any():
                # For indicator columns, fill with 0
                if 'indicator' in col.lower() or 'flag' in col.lower():
                    filled_df[col] = filled_df[col].fillna(0)
                # For other numeric columns, fill with mean
                else:
                    filled_df[col] = filled_df[col].fillna(filled_df[col].mean())
        
        # Fill categorical columns with mode
        categorical_columns = filled_df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if filled_df[col].isnull().any():
                mode_value = filled_df[col].mode().iloc[0] if not filled_df[col].mode().empty else 'unknown'
                filled_df[col] = filled_df[col].fillna(mode_value)
        
        return filled_df
    
    def save_features(self, features_df: pd.DataFrame, filename: str = None) -> str:
        """Save features to CSV file.
        
        Args:
            features_df: Features DataFrame to save
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cbb_features_{timestamp}.csv"
        
        # Ensure output directory exists
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        filepath = os.path.join(self.processed_data_dir, filename)
        
        # Save features
        features_df.to_csv(filepath, index=False)
        self.logger.info(f"Features saved to: {filepath}")
        
        # Create backup
        backup_path = os.path.join(self.backup_dir, filename)
        features_df.to_csv(backup_path, index=False)
        self.logger.info(f"Features backed up to: {backup_path}")
        
        return filepath
    
    def load_features(self, filepath: str) -> pd.DataFrame:
        """Load features from CSV file.
        
        Args:
            filepath: Path to features CSV file
            
        Returns:
            Loaded features DataFrame
        """
        if not os.path.exists(filepath):
            self.logger.error(f"Features file not found: {filepath}")
            return pd.DataFrame()
        
        features_df = pd.read_csv(filepath)
        self.logger.info(f"Features loaded from: {filepath} ({len(features_df)} rows, {len(features_df.columns)} columns)")
        
        return features_df
    
    def get_feature_summary(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Get comprehensive summary of all features.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Feature summary DataFrame
        """
        return self.feature_utils.create_feature_summary(features_df)
    
    def run_full_pipeline(
        self,
        games_file: str = None,
        odds_file: str = None,
        pbp_file: str = None,
        injury_file: str = None,
        lineup_file: str = None,
        output_file: str = None
    ) -> pd.DataFrame:
        """Run the complete feature engineering pipeline from files.
        
        Args:
            games_file: Path to games CSV file
            odds_file: Path to odds CSV file
            pbp_file: Path to play-by-play CSV file
            injury_file: Path to injury CSV file
            lineup_file: Path to lineup CSV file
            output_file: Output filename for features
            
        Returns:
            Complete feature DataFrame
        """
        self.logger.info("Running complete feature engineering pipeline...")
        
        # Load data files
        games_df = self._load_data_file(games_file) if games_file else pd.DataFrame()
        odds_df = self._load_data_file(odds_file) if odds_file else pd.DataFrame()
        pbp_df = self._load_data_file(pbp_file) if pbp_file else pd.DataFrame()
        injury_df = self._load_data_file(injury_file) if injury_file else pd.DataFrame()
        lineup_df = self._load_data_file(lineup_file) if lineup_file else pd.DataFrame()
        
        # Build feature set
        features_df = self.build_feature_set(games_df, odds_df, pbp_df, injury_df, lineup_df)
        
        # Save features
        if not features_df.empty:
            saved_path = self.save_features(features_df, output_file)
            self.logger.info(f"Pipeline completed successfully. Features saved to: {saved_path}")
        else:
            self.logger.error("Pipeline failed - no features generated")
        
        return features_df
    
    def _load_data_file(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded DataFrame
        """
        if not filepath or not os.path.exists(filepath):
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(filepath)
            self.logger.info(f"Loaded {len(df)} rows from {filepath}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {e}")
            return pd.DataFrame()


def create_feature_pipeline(config_path: str = "config.yaml") -> FeaturePipeline:
    """Create and return a feature pipeline instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FeaturePipeline instance
    """
    config = ConfigManager(config_path)
    return FeaturePipeline(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the feature pipeline
    try:
        pipeline = create_feature_pipeline()
        
        # Create sample data
        sample_games = pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(10)],
            'date': pd.date_range('2024-01-01', periods=10, freq='D'),
            'home_team': ['Duke', 'Kentucky', 'Duke', 'Kansas', 'Michigan State'] * 2,
            'away_team': ['North Carolina', 'Kansas', 'Michigan State', 'Duke', 'Kentucky'] * 2,
            'home_score': [85, 78, 92, 88, 76, 82, 79, 91, 84, 87],
            'away_score': [78, 82, 88, 85, 79, 85, 82, 88, 81, 84]
        })
        
        sample_odds = pd.DataFrame({
            'game_id': [f'game_{i}' for i in range(10)],
            'open_spread': [-2.5, -3.0, -1.5, +1.5, +2.0] * 2,
            'close_spread': [-3.0, -2.5, -2.0, +2.0, +1.5] * 2,
            'open_total': [145.5, 150.0, 148.5, 152.0, 147.5] * 2,
            'close_total': [146.0, 149.5, 149.0, 151.5, 148.0] * 2
        })
        
        # Run pipeline
        features = pipeline.build_feature_set(sample_games, sample_odds)
        
        print(f"Generated {len(features)} feature rows")
        print(f"Feature columns: {len(features.columns)}")
        print(f"Sample features:\n{features.head()}")
        
        # Save features
        if not features.empty:
            saved_path = pipeline.save_features(features)
            print(f"Features saved to: {saved_path}")
        
    except Exception as e:
        print(f"Error: {e}")