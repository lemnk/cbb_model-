"""
Feature Pipeline for NCAA CBB Betting ML System.

This module orchestrates the complete feature engineering pipeline by integrating
all individual feature engineers and creating a unified feature dataset.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta

from src.db import get_engine
from src.features.team_features import TeamFeatures
from src.features.player_features import PlayerFeatures
from src.features.market_features import MarketFeatures
from src.features.dynamic_features import DynamicFeatures


class FeaturePipeline:
    """Orchestrates the complete feature engineering pipeline."""
    
    def __init__(self):
        """Initialize the feature pipeline."""
        self.engine = get_engine()
        
        # Initialize feature engineers
        self.team_features = TeamFeatures(self.engine)
        self.player_features = PlayerFeatures(self.engine)
        self.market_features = MarketFeatures(self.engine)
        self.dynamic_features = DynamicFeatures(self.engine)
    
    def build_features(self):
        """Build the complete feature set by orchestrating all feature engineers.
        
        Returns:
            DataFrame with all features merged
        """
        print("🚀 Starting feature engineering pipeline...")
        
        # 1. Load raw data from DB
        print("📊 Loading raw data from database...")
        games_df = self._load_games_data()
        odds_df = self._load_odds_data()
        
        if games_df.empty:
            print("❌ No games data found. Exiting pipeline.")
            return pd.DataFrame()
        
        print(f"✅ Loaded {len(games_df)} games and {len(odds_df)} odds records")
        
        # 2. Apply feature engineers
        print("🔧 Computing team features...")
        team_feats = self.team_features.transform(games_df)
        
        print("👥 Computing player features...")
        player_feats = self.player_features.transform(games_df)
        
        print("💰 Computing market features...")
        market_feats = self.market_features.transform(odds_df)
        
        print("⚡ Computing dynamic features...")
        dynamic_feats = self.dynamic_features.transform(games_df)
        
        # 3. Merge all features
        print("🔗 Merging all feature sets...")
        features = self._merge_features(games_df, team_feats, player_feats, market_feats, dynamic_feats)
        
        # 4. Finalize feature set
        print("✨ Finalizing feature set...")
        features = self._finalize_feature_set(features)
        
        # 5. Save features
        filename = self._save_features(features)
        
        # 6. Print summary
        self._print_summary(features, filename)
        
        return features
    
    def _load_games_data(self) -> pd.DataFrame:
        """Load games data from database.
        
        Returns:
            DataFrame with games data
        """
        try:
            query = "SELECT * FROM games ORDER BY date"
            games_df = pd.read_sql(query, self.engine)
            
            # Ensure date column is datetime
            if 'date' in games_df.columns:
                games_df['date'] = pd.to_datetime(games_df['date'])
            
            return games_df
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load games data: {e}")
            print("📝 Creating sample games data for demonstration...")
            return self._create_sample_games_data()
    
    def _load_odds_data(self) -> pd.DataFrame:
        """Load odds data from database.
        
        Returns:
            DataFrame with odds data
        """
        try:
            query = "SELECT * FROM odds ORDER BY game_id"
            odds_df = pd.read_sql(query, self.engine)
            return odds_df
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load odds data: {e}")
            print("📝 Creating sample odds data for demonstration...")
            return self._create_sample_odds_data()
    
    def _create_sample_games_data(self) -> pd.DataFrame:
        """Create sample games data for demonstration.
        
        Returns:
            DataFrame with sample games data
        """
        np.random.seed(42)
        
        # Create sample teams
        teams = ['Duke', 'North Carolina', 'Kentucky', 'Kansas', 'Michigan State', 
                'Villanova', 'Gonzaga', 'Baylor', 'Arizona', 'UCLA']
        
        # Create sample games
        n_games = 100
        games_data = []
        
        for i in range(n_games):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate realistic scores
            home_score = np.random.randint(60, 100)
            away_score = np.random.randint(60, 100)
            
            # Generate date (last 6 months)
            date = datetime.now() - timedelta(days=np.random.randint(0, 180))
            
            games_data.append({
                'game_id': f'game_{i:04d}',
                'date': date,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'location': f'{home_team} Arena',
                'overtime': False
            })
        
        return pd.DataFrame(games_data)
    
    def _create_sample_odds_data(self) -> pd.DataFrame:
        """Create sample odds data for demonstration.
        
        Returns:
            DataFrame with sample odds data
        """
        np.random.seed(42)
        
        # Create sample odds for the games
        n_games = 100
        odds_data = []
        
        for i in range(n_games):
            # Generate realistic odds
            open_spread = np.random.uniform(-15, 15)
            close_spread = open_spread + np.random.uniform(-3, 3)
            
            open_total = np.random.uniform(130, 180)
            close_total = open_total + np.random.uniform(-5, 5)
            
            # Generate moneyline odds
            if open_spread > 0:
                open_ml_home = -110
                open_ml_away = +110
            else:
                open_ml_home = +110
                open_ml_away = -110
            
            if close_spread > 0:
                close_ml_home = -110
                close_ml_away = +110
            else:
                close_ml_home = +110
                close_ml_away = -110
            
            odds_data.append({
                'game_id': f'game_{i:04d}',
                'book': 'pinnacle',
                'open_ml_home': open_ml_home,
                'close_ml_home': close_ml_home,
                'open_ml_away': open_ml_away,
                'close_ml_away': close_ml_away,
                'open_spread': open_spread,
                'close_spread': close_spread,
                'open_total': open_total,
                'close_total': close_total
            })
        
        return pd.DataFrame(odds_data)
    
    def _merge_features(self, games_df: pd.DataFrame, team_feats: pd.DataFrame, 
                       player_feats: pd.DataFrame, market_feats: pd.DataFrame, 
                       dynamic_feats: pd.DataFrame) -> pd.DataFrame:
        """Merge all feature sets into a single DataFrame.
        
        Args:
            games_df: Base games DataFrame
            team_feats: Team features DataFrame
            player_feats: Player features DataFrame
            market_feats: Market features DataFrame
            dynamic_feats: Dynamic features DataFrame
            
        Returns:
            Merged DataFrame with all features
        """
        # Start with games data
        features = games_df.copy()
        
        # Merge team features
        if not team_feats.empty:
            features = features.merge(team_feats, left_index=True, right_index=True, how='left')
        
        # Merge player features
        if not player_feats.empty:
            features = features.merge(player_feats, left_index=True, right_index=True, how='left')
        
        # Merge dynamic features
        if not dynamic_feats.empty:
            features = features.merge(dynamic_feats, left_index=True, right_index=True, how='left')
        
        # Merge market features (need to match by game_id)
        if not market_feats.empty and 'game_id' in market_feats.columns:
            features = features.merge(market_feats, on='game_id', how='left')
        
        return features
    
    def _finalize_feature_set(self, features: pd.DataFrame) -> pd.DataFrame:
        """Finalize the feature set with additional enhancements.
        
        Args:
            features: Raw merged features DataFrame
            
        Returns:
            Finalized features DataFrame
        """
        # Remove duplicate columns (keep first occurrence)
        features = features.loc[:, ~features.columns.duplicated()]
        
        # Fill missing values
        features = self._fill_missing_values(features)
        
        # Add feature metadata
        features['feature_generation_date'] = datetime.now()
        features['feature_version'] = '2.0.0'
        
        # Sort by date and game_id
        if 'date' in features.columns:
            features = features.sort_values(['date', 'game_id'])
        
        return features
    
    def _fill_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the feature set.
        
        Args:
            features: Features DataFrame with missing values
            
        Returns:
            DataFrame with filled missing values
        """
        # Fill numeric columns with median
        numeric_cols = features.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if features[col].isnull().any():
                median_val = features[col].median()
                features[col] = features[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_cols = features.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if features[col].isnull().any():
                mode_val = features[col].mode().iloc[0] if not features[col].mode().empty else 'Unknown'
                features[col] = features[col].fillna(mode_val)
        
        return features
    
    def _save_features(self, features: pd.DataFrame) -> str:
        """Save features to CSV file.
        
        Args:
            features: Features DataFrame to save
            
        Returns:
            Filename where features were saved
        """
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d')
        filename = f"data/features_{timestamp}.csv"
        
        # Ensure data directory exists
        import os
        os.makedirs('data', exist_ok=True)
        
        # Save to CSV
        features.to_csv(filename, index=False)
        
        return filename
    
    def _print_summary(self, features: pd.DataFrame, filename: str):
        """Print summary of the generated features.
        
        Args:
            features: Features DataFrame
            filename: Filename where features were saved
        """
        print("\n" + "="*60)
        print("🎯 FEATURE ENGINEERING PIPELINE COMPLETE!")
        print("="*60)
        
        print(f"📊 Dataset Shape: {features.shape[0]} rows × {features.shape[1]} columns")
        print(f"💾 Saved to: {filename}")
        
        # Feature count by category
        feature_categories = {
            'Team Features': [col for col in features.columns if col.startswith('team_')],
            'Player Features': [col for col in features.columns if col.startswith('player_')],
            'Market Features': [col for col in features.columns if col.startswith('market_')],
            'Dynamic Features': [col for col in features.columns if col.startswith('dynamic_')]
        }
        
        print("\n📈 Feature Breakdown:")
        for category, cols in feature_categories.items():
            print(f"   {category}: {len(cols)} features")
        
        # Sample features
        print(f"\n🔍 Sample Features (first 10):")
        sample_cols = [col for col in features.columns if not col.startswith('feature_')][:10]
        for col in sample_cols:
            print(f"   • {col}")
        
        # Data quality info
        missing_pct = (features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100
        print(f"\n✅ Data Quality:")
        print(f"   Missing Values: {missing_pct:.2f}%")
        print(f"   Duplicate Rows: {features.duplicated().sum()}")
        
        print("\n🚀 Features are ready for ML model training!")
        print("="*60)


def get_engine():
    """Get database engine for feature pipeline.
    
    Returns:
        SQLAlchemy engine
    """
    try:
        # Try to get engine from db module
        from src.db import DatabaseManager
        db_manager = DatabaseManager()
        return db_manager.engine
    except ImportError:
        # Fallback to creating a dummy engine for demo
        print("⚠️ Warning: Could not import DatabaseManager, using dummy engine")
        return None


if __name__ == "__main__":
    """Run the feature pipeline when executed directly."""
    try:
        pipeline = FeaturePipeline()
        features = pipeline.build_features()
        
        if not features.empty:
            print(f"\n🎉 Successfully generated {len(features.columns)} features!")
            print(f"📁 Features saved to data/features_YYYYMMDD.csv")
        else:
            print("\n❌ Feature generation failed!")
            
    except Exception as e:
        print(f"\n💥 Error running feature pipeline: {e}")
        import traceback
        traceback.print_exc()