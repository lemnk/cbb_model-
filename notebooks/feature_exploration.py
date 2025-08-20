#!/usr/bin/env python3
"""
Feature Exploration Script for CBB Betting ML System
Phase 2: Feature Engineering

This script demonstrates the feature engineering capabilities by:
1. Loading sample data (placeholder for real data from Phase 1)
2. Running the hardened feature pipeline
3. Analyzing feature correlations and distributions
4. Visualizing key features and GSI components

Usage:
    python3 notebooks/feature_exploration.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from features.feature_pipeline import FeaturePipeline
    from features.feature_utils import validate_keys, standardize_team_names
    print("✅ Successfully imported feature modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

def create_sample_data():
    """
    Create comprehensive sample data for feature exploration.
    In production, this would load from Phase 1 database.
    
    Returns:
        tuple: (games_df, odds_df, players_df)
    """
    print("🔄 Creating sample data...")
    
    np.random.seed(42)
    n_games = 200  # More games for better analysis
    
    # Sample games data with realistic features
    teams = ['Duke', 'UNC', 'Kentucky', 'Kansas', 'Michigan State', 'Villanova', 
             'Gonzaga', 'Baylor', 'Arizona', 'Houston']
    
    games_df = pd.DataFrame({
        'game_id': range(1, n_games + 1),
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'team': np.random.choice(teams, n_games),
        'opponent': np.random.choice(teams, n_games),
        'points': np.random.normal(75, 15, n_games),
        'opponent_points': np.random.normal(70, 15, n_games),
        'won': np.random.choice([0, 1], n_games),
        'home_team': np.random.choice(teams, n_games),
        'away_team': np.random.choice(teams, n_games),
        'is_home': np.random.choice([0, 1], n_games),
        'venue': np.random.choice(['Cameron Indoor', 'Dean Dome', 'Rupp Arena', 'Allen Fieldhouse'], n_games),
        'conference_game': np.random.choice([0, 1], n_games, p=[0.6, 0.4]),
        'rivalry_game': np.random.choice([0, 1], n_games, p=[0.8, 0.2])
    })
    
    # Ensure no team plays itself
    games_df = games_df[games_df['team'] != games_df['opponent']]
    
    # Sample odds data with realistic movements
    odds_df = pd.DataFrame({
        'game_id': range(1, n_games + 1),
        'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
        'team': np.random.choice(teams, n_games),
        'open_spread': np.random.normal(0, 10, n_games),
        'close_spread': np.random.normal(0, 10, n_games),
        'open_total': np.random.normal(140, 20, n_games),
        'close_total': np.random.normal(140, 20, n_games),
        'open_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games),
        'close_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games),
        'open_timestamp': pd.date_range('2024-01-01', periods=n_games, freq='D') - pd.Timedelta(days=1),
        'market_timestamp': pd.date_range('2024-01-01', periods=n_games, freq='D') - pd.Timedelta(hours=2),
        'game_date': pd.date_range('2024-01-01', periods=n_games, freq='D')
    })
    
    # Ensure closing odds are before game start (leakage prevention)
    odds_df['market_timestamp'] = odds_df['market_timestamp'].clip(upper=odds_df['game_date'])
    
    # Sample players data with realistic distributions
    players_df = pd.DataFrame({
        'game_id': np.repeat(range(1, n_games + 1), 12),  # 12 players per game
        'date': np.repeat(pd.date_range('2024-01-01', periods=n_games, freq='D'), 12),
        'team': np.repeat(np.random.choice(teams, n_games), 12),
        'player_id': range(1, n_games * 12 + 1),
        'name': [f'Player_{i}' for i in range(1, n_games * 12 + 1)],
        'minutes': np.random.uniform(5, 40, n_games * 12),
        'points': np.random.normal(8, 6, n_games * 12),
        'rebounds': np.random.normal(3, 2, n_games * 12),
        'assists': np.random.normal(2, 2, n_games * 12),
        'fouls': np.random.poisson(2, n_games * 12),
        'injured': np.random.choice([0, 1], n_games * 12, p=[0.92, 0.08]),
        'status': np.random.choice(['healthy', 'questionable', 'out'], n_games * 12, p=[0.92, 0.05, 0.03]),
        'is_starter': np.random.choice([0, 1], n_games * 12, p=[0.7, 0.3]),
        'role': np.random.choice(['starter', 'rotation', 'bench'], n_games * 12, p=[0.3, 0.4, 0.3])
    })
    
    # Ensure starters get more minutes
    players_df.loc[players_df['is_starter'] == 1, 'minutes'] = np.random.uniform(25, 40, 
        size=players_df[players_df['is_starter'] == 1].shape[0])
    
    # Ensure injured players get fewer minutes
    players_df.loc[players_df['injured'] == 1, 'minutes'] = np.random.uniform(0, 15, 
        size=players_df[players_df['injured'] == 1].shape[0])
    
    print(f"✅ Sample data created: {len(games_df)} games, {len(odds_df)} odds records, {len(players_df)} player records")
    return games_df, odds_df, players_df

def run_feature_pipeline(games_df, odds_df, players_df):
    """
    Run the hardened feature pipeline.
    
    Args:
        games_df: Games DataFrame
        odds_df: Odds DataFrame
        players_df: Players DataFrame
        
    Returns:
        DataFrame: Engineered features
    """
    print("🔄 Running hardened feature pipeline...")
    
    try:
        # Initialize pipeline
        pipeline = FeaturePipeline()
        
        # Build features with validation
        features = pipeline.build_features(games_df, odds_df, players_df)
        
        print(f"✅ Feature pipeline completed successfully!")
        print(f"📊 Final feature set: {features.shape[0]} rows × {features.shape[1]} columns")
        
        return features, pipeline
        
    except Exception as e:
        print(f"❌ Feature pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def analyze_features(features_df):
    """
    Analyze the engineered features for insights.
    
    Args:
        features_df: DataFrame with engineered features
    """
    print("\n🔍 Analyzing engineered features...")
    
    # Basic statistics
    print(f"📊 Feature Analysis Summary:")
    print(f"   Total features: {len(features_df.columns)}")
    print(f"   Total games: {len(features_df)}")
    print(f"   Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Feature categories
    feature_categories = {
        'Team Features': [col for col in features_df.columns if col.startswith('team_')],
        'Player Features': [col for col in features_df.columns if col.startswith('team_') and 'player' in col],
        'Dynamic Features': [col for col in features_df.columns if any(x in col for x in ['streak', 'rest', 'travel', 'altitude', 'situational'])],
        'Market Features': [col for col in features_df.columns if any(x in col for x in ['movement', 'market', 'clv', 'edge', 'efficiency'])],
        'GSI Components': [col for col in features_df.columns if col.startswith('gsi_')],
        'Core Features': [col for col in features_df.columns if col in ['game_id', 'date', 'team', 'opponent', 'points', 'won']]
    }
    
    print(f"\n📋 Feature Categories:")
    for category, cols in feature_categories.items():
        print(f"   {category}: {len(cols)} features")
    
    # GSI analysis
    if 'gsi' in features_df.columns:
        gsi_stats = features_df['gsi'].describe()
        print(f"\n🎯 Game Strength Index (GSI) Analysis:")
        print(f"   Mean: {gsi_stats['mean']:.3f}")
        print(f"   Std: {gsi_stats['std']:.3f}")
        print(f"   Min: {gsi_stats['min']:.3f}")
        print(f"   Max: {gsi_stats['max']:.3f}")
        print(f"   Range: {gsi_stats['max'] - gsi_stats['min']:.3f}")
        
        # GSI distribution
        gsi_categories = features_df['gsi_category'].value_counts()
        print(f"   GSI Categories: {dict(gsi_categories)}")
    
    # Missing value analysis
    missing_data = features_df.isnull().sum()
    missing_cols = missing_data[missing_data > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠️ Missing Value Analysis:")
        for col, count in missing_cols.items():
            percentage = (count / len(features_df)) * 100
            print(f"   {col}: {count} ({percentage:.1f}%)")
    else:
        print(f"\n✅ No missing values found in engineered features")
    
    # Data types
    dtype_counts = features_df.dtypes.value_counts()
    print(f"\n📊 Data Types:")
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")

def create_visualizations(features_df):
    """
    Create visualizations of key features and relationships.
    
    Args:
        features_df: DataFrame with engineered features
    """
    print("\n🎨 Creating visualizations...")
    
    try:
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CBB Betting ML System - Feature Engineering Analysis', fontsize=16, fontweight='bold')
        
        # 1. GSI Distribution
        if 'gsi' in features_df.columns:
            axes[0, 0].hist(features_df['gsi'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 0].set_title('Game Strength Index Distribution')
            axes[0, 0].set_xlabel('GSI Value')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Team Efficiency vs GSI
        if 'team_combined_efficiency' in features_df.columns and 'gsi' in features_df.columns:
            axes[0, 1].scatter(features_df['team_combined_efficiency'], features_df['gsi'], alpha=0.6)
            axes[0, 1].set_title('Team Efficiency vs GSI')
            axes[0, 1].set_xlabel('Team Combined Efficiency')
            axes[0, 1].set_ylabel('Game Strength Index')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Market Efficiency Distribution
        if 'market_efficiency_score' in features_df.columns:
            axes[0, 2].hist(features_df['market_efficiency_score'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 2].set_title('Market Efficiency Score Distribution')
            axes[0, 2].set_xlabel('Market Efficiency Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Rest Quality vs Win Rate
        if 'rest_quality_score' in features_df.columns and 'won' in features_df.columns:
            rest_win_data = features_df.groupby('rest_quality_score')['won'].mean()
            axes[1, 0].plot(rest_win_data.index, rest_win_data.values, marker='o')
            axes[1, 0].set_title('Rest Quality vs Win Rate')
            axes[1, 0].set_xlabel('Rest Quality Score')
            axes[1, 0].set_ylabel('Win Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Line Movement Analysis
        if 'spread_movement_magnitude' in features_df.columns:
            axes[1, 1].hist(features_df['spread_movement_magnitude'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Spread Movement Magnitude')
            axes[1, 1].set_xlabel('Movement Magnitude')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Correlation Heatmap (top features)
        if 'gsi' in features_df.columns:
            # Select top numeric features for correlation
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            top_features = ['gsi'] + [col for col in numeric_cols if col != 'gsi'][:9]  # Top 10 features
            
            if len(top_features) >= 2:
                corr_matrix = features_df[top_features].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           ax=axes[1, 2], square=True, fmt='.2f')
                axes[1, 2].set_title('Feature Correlation Heatmap (Top Features)')
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"data/feature_analysis_{timestamp}.png"
        os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to: {plot_filename}")
        
        # Show the plot
        plt.show()
        
    except Exception as e:
        print(f"⚠️ Warning: Could not create visualizations: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the feature exploration.
    """
    print("🚀 CBB Betting ML System - Feature Exploration")
    print("=" * 60)
    
    try:
        # Step 1: Create sample data
        games_df, odds_df, players_df = create_sample_data()
        
        # Step 2: Run hardened feature pipeline
        features_df, pipeline = run_feature_pipeline(games_df, odds_df, players_df)
        
        if features_df is None:
            print("❌ Feature pipeline failed. Exiting.")
            return
        
        # Step 3: Analyze features
        analyze_features(features_df)
        
        # Step 4: Create visualizations
        create_visualizations(features_df)
        
        # Step 5: Save features for further analysis
        if pipeline:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"data/features_exploration_{timestamp}.csv"
            pipeline.save_features(features_df, output_path)
        
        print("\n🎉 Feature exploration completed successfully!")
        print(f"📊 Final feature set: {features_df.shape[0]} rows × {features_df.shape[1]} columns")
        print(f"🎯 GSI range: {features_df['gsi'].min():.3f} - {features_df['gsi'].max():.3f}")
        
        # Verify feature richness requirement
        if len(features_df.columns) > 30:
            print(f"✅ Feature richness requirement met: {len(features_df.columns)} features")
        else:
            print(f"⚠️ Feature richness below requirement: {len(features_df.columns)} features (need >30)")
        
    except Exception as e:
        print(f"❌ Feature exploration failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()