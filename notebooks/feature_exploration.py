"""
CBB Betting ML System - Feature Exploration

This script demonstrates the feature engineering capabilities of Phase 2, including:
- Team context and performance features
- Dynamic game flow and momentum features
- Player availability and injury features
- Market efficiency and odds-based features
- Feature correlation analysis and visualization

Run this script to explore the feature engineering pipeline interactively.
"""

import sys
import os
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Import feature engineering modules
from src.features import (
    create_feature_pipeline,
    create_team_feature_engineer,
    create_dynamic_feature_engineer,
    create_player_feature_engineer,
    create_market_feature_engineer
)

def setup_plotting():
    """Setup matplotlib and seaborn for plotting."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    print("Plotting setup complete!")

def create_sample_data():
    """Create realistic sample data for demonstration."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample games data
    n_games = 100
    teams = ['Duke', 'Kentucky', 'Kansas', 'Michigan State', 'North Carolina', 
              'Villanova', 'Gonzaga', 'Baylor', 'Arizona', 'Houston']
    
    sample_games = pd.DataFrame({
        'game_id': [f'game_{i:03d}' for i in range(n_games)],
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'home_team': np.random.choice(teams, n_games),
        'away_team': np.random.choice(teams, n_games),
        'home_score': np.random.randint(60, 100, n_games),
        'away_score': np.random.randint(60, 100, n_games),
        'overtime': np.random.choice([True, False], n_games, p=[0.1, 0.9])
    })
    
    # Ensure no team plays against itself
    sample_games = sample_games[sample_games['home_team'] != sample_games['away_team']]
    n_games = len(sample_games)
    
    # Create sample odds data
    sample_odds = pd.DataFrame({
        'game_id': sample_games['game_id'],
        'book': np.random.choice(['pinnacle', 'draftkings', 'fanduel'], n_games),
        'open_moneyline_home': np.random.choice([-110, -105, -100, +100, +105, +110], n_games),
        'close_moneyline_home': np.random.choice([-110, -105, -100, +100, +105, +110], n_games),
        'open_moneyline_away': np.random.choice([-110, -105, -100, +100, +105, +110], n_games),
        'close_moneyline_away': np.random.choice([-110, -105, -100, +100, +105, +110], n_games),
        'open_spread': np.random.uniform(-15, 15, n_games),
        'close_spread': np.random.uniform(-15, 15, n_games),
        'open_total': np.random.uniform(120, 180, n_games),
        'close_total': np.random.uniform(120, 180, n_games)
    })
    
    # Create sample play-by-play data
    sample_pbp = []
    for _, game in sample_games.iterrows():
        n_plays = np.random.randint(80, 120)
        for play in range(n_plays):
            sample_pbp.append({
                'game_id': game['game_id'],
                'quarter': np.random.randint(1, 5),
                'time_remaining': np.random.randint(0, 1200),
                'home_score': np.random.randint(0, game['home_score']),
                'away_score': np.random.randint(0, game['away_score'])
            })
    
    sample_pbp = pd.DataFrame(sample_pbp)
    
    print(f"Sample data created:")
    print(f"  - Games: {len(sample_games)} rows")
    print(f"  - Odds: {len(sample_odds)} rows")
    print(f"  - Play-by-play: {len(sample_pbp)} rows")
    
    return sample_games, sample_odds, sample_pbp

def explore_team_features(sample_games):
    """Explore team context features."""
    print("\n" + "="*50)
    print("TEAM FEATURE ENGINEERING")
    print("="*50)
    
    # Initialize team feature engineer
    team_engineer = create_team_feature_engineer()
    
    # Compute team features
    team_features = team_engineer.compute_team_context(sample_games)
    
    print(f"Team features generated: {len(team_features)} rows, {len(team_features.columns)} columns")
    print("\nFeature categories:")
    print(f"  - Basic performance: {len([col for col in team_features.columns if 'score' in col or 'win' in col])} features")
    print(f"  - Efficiency: {len([col for col in team_features.columns if 'adj' in col or 'tempo' in col])} features")
    print(f"  - Travel: {len([col for col in team_features.columns if 'travel' in col or 'altitude' in col])} features")
    print(f"  - Rolling: {len([col for col in team_features.columns if 'rolling' in col])} features")
    
    # Display sample features
    print("\nSample team features:")
    print(team_features[['game_id', 'home_team', 'away_team', 'home_adj_o', 'home_adj_d', 
                        'home_tempo', 'home_days_rest', 'home_rolling_score_5']].head())
    
    return team_features

def explore_dynamic_features(sample_pbp):
    """Explore dynamic game flow features."""
    print("\n" + "="*50)
    print("DYNAMIC FEATURE ENGINEERING")
    print("="*50)
    
    # Initialize dynamic feature engineer
    dynamic_engineer = create_dynamic_feature_engineer()
    
    # Compute dynamic features
    dynamic_features = dynamic_engineer.compute_game_flow(sample_pbp)
    
    print(f"Dynamic features generated: {len(dynamic_features)} rows, {len(dynamic_features.columns)} columns")
    print("\nFeature categories:")
    print(f"  - Momentum: {len([col for col in dynamic_features.columns if 'momentum' in col])} features")
    print(f"  - Streaks: {len([col for col in dynamic_features.columns if 'streak' in col])} features")
    print(f"  - Possession: {len([col for col in dynamic_features.columns if 'possession' in col])} features")
    print(f"  - Time: {len([col for col in dynamic_features.columns if 'time' in col])} features")
    
    # Display sample features
    print("\nSample dynamic features:")
    print(dynamic_features[['game_id', 'quarter', 'momentum_index', 'home_scoring_streak', 
                           'possession_efficiency', 'time_pressure']].head())
    
    return dynamic_features

def explore_market_features(sample_odds):
    """Explore market efficiency features."""
    print("\n" + "="*50)
    print("MARKET FEATURE ENGINEERING")
    print("="*50)
    
    # Initialize market feature engineer
    market_engineer = create_market_feature_engineer()
    
    # Compute market features
    market_features = market_engineer.compute_market_signals(sample_odds)
    
    print(f"Market features generated: {len(market_features)} rows, {len(market_features.columns)} columns")
    print("\nFeature categories:")
    print(f"  - Line movements: {len([col for col in market_features.columns if 'drift' in col or 'movement' in col])} features")
    print(f"  - Probabilities: {len([col for col in market_features.columns if 'prob' in col])} features")
    print(f"  - CLV: {len([col for col in market_features.columns if 'clv' in col])} features")
    print(f"  - Efficiency: {len([col for col in market_features.columns if 'efficiency' in col])} features")
    
    # Display sample features
    print("\nSample market features:")
    print(market_features[['game_id', 'spread_drift', 'total_drift', 'market_efficiency_score', 
                          'clv_home', 'clv_away']].head())
    
    return market_features

def run_complete_pipeline(sample_games, sample_odds, sample_pbp):
    """Run the complete feature engineering pipeline."""
    print("\n" + "="*50)
    print("COMPLETE FEATURE PIPELINE")
    print("="*50)
    
    # Initialize feature pipeline
    pipeline = create_feature_pipeline()
    
    # Run complete pipeline
    complete_features = pipeline.build_feature_set(sample_games, sample_odds, sample_pbp)
    
    print(f"Complete feature set generated: {len(complete_features)} rows, {len(complete_features.columns)} columns")
    print("\nFeature breakdown:")
    print(f"  - Original game columns: {len(sample_games.columns)}")
    print(f"  - Engineered features: {len(complete_features.columns) - len(sample_games.columns)}")
    
    # Show feature summary
    feature_summary = pipeline.get_feature_summary(complete_features)
    print("\nFeature summary (first 10):")
    print(feature_summary.head(10))
    
    return complete_features, pipeline

def analyze_correlations(complete_features):
    """Analyze feature correlations."""
    print("\n" + "="*50)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*50)
    
    # Select numeric features for correlation analysis
    numeric_features = complete_features.select_dtypes(include=[np.number])
    print(f"Analyzing correlations for {len(numeric_features.columns)} numeric features")
    
    # Calculate correlation matrix
    correlation_matrix = numeric_features.corr()
    
    # Find highly correlated features
    high_corr_threshold = 0.8
    high_corr_pairs = []
    
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > high_corr_threshold:
                high_corr_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    corr_value
                ))
    
    print(f"\nFound {len(high_corr_pairs)} feature pairs with correlation > {high_corr_threshold}:")
    for feature1, feature2, corr in high_corr_pairs[:10]:  # Show first 10
        print(f"  {feature1} <-> {feature2}: {corr:.3f}")
    
    return correlation_matrix, high_corr_pairs

def analyze_feature_importance(complete_features):
    """Analyze feature importance for predicting game outcomes."""
    print("\n" + "="*50)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*50)
    
    if 'home_win' in complete_features.columns:
        # Select features and target
        numeric_features = complete_features.select_dtypes(include=[np.number])
        feature_cols = [col for col in numeric_features.columns if col not in ['home_win', 'away_win']]
        X = numeric_features[feature_cols].fillna(0)
        y = complete_features['home_win']
        
        # Calculate feature correlations with target
        target_correlations = []
        for col in feature_cols:
            if col in X.columns:
                corr = X[col].corr(y)
                target_correlations.append((col, abs(corr)))
        
        # Sort by absolute correlation
        target_correlations.sort(key=lambda x: x[1], reverse=True)
        
        print("Top 10 features by correlation with home team win:")
        for i, (feature, corr) in enumerate(target_correlations[:10]):
            print(f"  {i+1:2d}. {feature}: {corr:.3f}")
        
        return target_correlations
    else:
        print("Target variable 'home_win' not found in features")
        return []

def assess_feature_quality(complete_features, pipeline):
    """Assess the quality of engineered features."""
    print("\n" + "="*50)
    print("FEATURE QUALITY ASSESSMENT")
    print("="*50)
    
    # Validate feature set
    validation_results = pipeline.feature_utils.validate_feature_set(complete_features)
    
    print("Feature Set Validation Results:")
    print(f"  Valid: {validation_results['valid']}")
    print(f"  Total Rows: {validation_results['total_rows']}")
    print(f"  Total Columns: {validation_results['total_columns']}")
    print(f"  Warnings: {len(validation_results['warnings'])}")
    print(f"  Errors: {len(validation_results['errors'])}")
    
    if validation_results['warnings']:
        print("\nWarnings:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    if validation_results['errors']:
        print("\nErrors:")
        for error in validation_results['errors']:
            print(f"  - {error}")
    
    # Missing data analysis
    missing_data = validation_results['missing_data']
    high_missing_features = [(col, data['percentage']) for col, data in missing_data.items() 
                             if data['percentage'] > 20]
    
    if high_missing_features:
        print(f"\nFeatures with >20% missing data ({len(high_missing_features)}):")
        for feature, missing_pct in sorted(high_missing_features, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {feature}: {missing_pct:.1f}% missing")
    else:
        print("\nAll features have good data quality (<20% missing)")
    
    return validation_results

def save_features(complete_features, pipeline):
    """Save engineered features."""
    print("\n" + "="*50)
    print("SAVING ENGINEERED FEATURES")
    print("="*50)
    
    if not complete_features.empty:
        saved_path = pipeline.save_features(complete_features, 'cbb_features_demo.csv')
        print(f"Features saved to: {saved_path}")
        
        # Verify file was saved
        if os.path.exists(saved_path):
            file_size = os.path.getsize(saved_path) / (1024 * 1024)  # MB
            print(f"File size: {file_size:.2f} MB")
            
            # Load back to verify
            loaded_features = pipeline.load_features(saved_path)
            print(f"Loaded back: {len(loaded_features)} rows, {len(loaded_features.columns)} columns")
            
            if len(loaded_features) == len(complete_features):
                print("✅ Feature saving and loading successful!")
            else:
                print("❌ Feature saving/loading mismatch")
        else:
            print("❌ Failed to save features")
    else:
        print("❌ No features to save")

def main():
    """Main function to run the feature exploration."""
    print("🏀 CBB Betting ML System - Feature Exploration")
    print("="*60)
    
    # Setup
    setup_plotting()
    
    # Create sample data
    sample_games, sample_odds, sample_pbp = create_sample_data()
    
    # Explore individual feature types
    team_features = explore_team_features(sample_games)
    dynamic_features = explore_dynamic_features(sample_pbp)
    market_features = explore_market_features(sample_odds)
    
    # Run complete pipeline
    complete_features, pipeline = run_complete_pipeline(sample_games, sample_odds, sample_pbp)
    
    # Analyze features
    correlation_matrix, high_corr_pairs = analyze_correlations(complete_features)
    target_correlations = analyze_feature_importance(complete_features)
    validation_results = assess_feature_quality(complete_features, pipeline)
    
    # Save features
    save_features(complete_features, pipeline)
    
    # Summary
    print("\n" + "="*60)
    print("FEATURE EXPLORATION COMPLETE!")
    print("="*60)
    print(f"✅ Generated {len(complete_features.columns)} features")
    print(f"✅ Found {len(high_corr_pairs)} highly correlated feature pairs")
    print(f"✅ Feature set validation: {validation_results['valid']}")
    print("\nReady for Phase 3: Model Training! 🚀")

if __name__ == "__main__":
    main()