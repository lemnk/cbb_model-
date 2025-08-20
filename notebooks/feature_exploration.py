#!/usr/bin/env python3
"""
Feature Exploration Notebook for CBB Betting ML System - Phase 2

This script demonstrates the feature engineering capabilities of the system,
including loading sample data, running the feature pipeline, and generating
visualizations for analysis.

Usage:
    python notebooks/feature_exploration.py
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from features.feature_pipeline import FeaturePipeline
from features.feature_utils import feature_correlation_analysis, normalize, handle_missing

def setup_plotting():
    """Setup matplotlib and seaborn for better visualizations."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    print("✅ Plotting setup complete")

def load_sample_data():
    """Load sample data for exploration."""
    print("📊 Loading sample data...")
    
    # Initialize pipeline to get sample data
    pipeline = FeaturePipeline()
    games_df, odds_df, players_df = pipeline.load_sample_data()
    
    print(f"   Games: {games_df.shape}")
    print(f"   Odds: {odds_df.shape}")
    print(f"   Players: {players_df.shape}")
    
    return games_df, odds_df, players_df

def run_feature_pipeline(games_df, odds_df, players_df):
    """Run the complete feature engineering pipeline."""
    print("\n🔧 Running feature engineering pipeline...")
    
    pipeline = FeaturePipeline()
    features = pipeline.build_features(games_df, odds_df, players_df)
    
    print(f"✅ Generated {len(features.columns)} features for {len(features)} games")
    return features, pipeline

def explore_feature_categories(features):
    """Explore and categorize the generated features."""
    print("\n📈 Exploring feature categories...")
    
    # Categorize features
    feature_categories = {
        'Team Features': [col for col in features.columns if 'team_' in col],
        'Player Features': [col for col in features.columns if any(x in col for x in ['injury', 'foul', 'bench', 'minutes'])],
        'Market Features': [col for col in features.columns if any(x in col for x in ['movement', 'market', 'clv', 'edge'])],
        'Dynamic Features': [col for col in features.columns if any(x in col for x in ['streak', 'rest', 'travel', 'altitude'])],
        'Other Features': [col for col in features.columns if not any(x in col for x in ['team_', 'injury', 'foul', 'bench', 'minutes', 'movement', 'market', 'clv', 'edge', 'streak', 'rest', 'travel', 'altitude'])]
    }
    
    print("Feature breakdown:")
    for category, cols in feature_categories.items():
        print(f"   {category}: {len(cols)} features")
        if cols:
            print(f"     Sample: {', '.join(cols[:3])}{'...' if len(cols) > 3 else ''}")
    
    return feature_categories

def analyze_feature_statistics(features):
    """Analyze basic statistics of the features."""
    print("\n📊 Analyzing feature statistics...")
    
    # Basic info
    print(f"Dataset shape: {features.shape}")
    print(f"Memory usage: {features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data types
    dtype_counts = features.dtypes.value_counts()
    print("\nData types:")
    for dtype, count in dtype_counts.items():
        print(f"   {dtype}: {count} columns")
    
    # Missing values
    missing_counts = features.isnull().sum()
    missing_pct = (missing_counts / len(features)) * 100
    missing_summary = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing %': missing_pct
    }).sort_values('Missing Count', ascending=False)
    
    print(f"\nMissing values summary:")
    print(f"   Total missing: {missing_summary['Missing Count'].sum()}")
    print(f"   Average missing %: {missing_summary['Missing %'].mean():.2f}%")
    
    # Show columns with missing values
    if missing_summary['Missing Count'].sum() > 0:
        print(f"   Columns with missing values:")
        for col in missing_summary[missing_summary['Missing Count'] > 0].index[:5]:
            count = missing_summary.loc[col, 'Missing Count']
            pct = missing_summary.loc[col, 'Missing %']
            print(f"     {col}: {count} ({pct:.1f}%)")
    
    return missing_summary

def explore_game_strength_index(features):
    """Explore the Game Strength Index (GSI) feature."""
    print("\n🎯 Exploring Game Strength Index...")
    
    if 'game_strength_index' not in features.columns:
        print("   ❌ Game Strength Index not found in features")
        return
    
    gsi = features['game_strength_index']
    
    # Basic statistics
    print(f"   Mean: {gsi.mean():.4f}")
    print(f"   Std: {gsi.std():.4f}")
    print(f"   Min: {gsi.min():.4f}")
    print(f"   Max: {gsi.max():.4f}")
    print(f"   Range: {gsi.max() - gsi.min():.4f}")
    
    # Distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(gsi, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Game Strength Index Distribution')
    plt.xlabel('GSI Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    gsi.boxplot()
    plt.title('Game Strength Index Box Plot')
    plt.ylabel('GSI Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # GSI categories if available
    if 'gsi_category' in features.columns:
        gsi_cats = features['gsi_category'].value_counts()
        print(f"\n   GSI Categories:")
        for cat, count in gsi_cats.items():
            pct = (count / len(features)) * 100
            print(f"     {cat}: {count} games ({pct:.1f}%)")

def create_correlation_heatmap(features, max_features=20):
    """Create a correlation heatmap for the most important features."""
    print("\n🔥 Creating correlation heatmap...")
    
    # Select numeric features
    numeric_features = features.select_dtypes(include=[np.number])
    
    if len(numeric_features.columns) > max_features:
        # Select features with highest variance
        variances = numeric_features.var().sort_values(ascending=False)
        selected_features = variances.head(max_features).index.tolist()
        numeric_features = numeric_features[selected_features]
        print(f"   Selected top {max_features} features by variance")
    
    # Calculate correlation matrix
    corr_matrix = numeric_features.corr()
    
    # Create heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        square=True,
        fmt='.2f',
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f'Feature Correlation Heatmap ({len(numeric_features.columns)} features)')
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = corr_matrix.iloc[i, j]
            if abs(corr_value) > 0.8:
                high_corr_pairs.append({
                    'feature1': corr_matrix.columns[i],
                    'feature2': corr_matrix.columns[j],
                    'correlation': corr_value
                })
    
    if high_corr_pairs:
        print(f"   High correlation pairs (|r| > 0.8):")
        for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]:
            print(f"     {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
    else:
        print(f"   No highly correlated features found (|r| > 0.8)")

def explore_feature_relationships(features):
    """Explore relationships between key features."""
    print("\n🔍 Exploring feature relationships...")
    
    # Select some key features for analysis
    key_features = []
    for prefix in ['team_', 'injury_', 'spread_', 'streak_']:
        cols = [col for col in features.columns if col.startswith(prefix)]
        if cols:
            key_features.extend(cols[:2])  # Take first 2 from each category
    
    if len(key_features) < 4:
        # Fallback to first few numeric columns
        numeric_cols = features.select_dtypes(include=[np.number]).columns[:8]
        key_features = numeric_cols.tolist()
    
    print(f"   Analyzing relationships between: {', '.join(key_features[:6])}")
    
    # Create pairplot for key features
    try:
        plot_data = features[key_features[:6]].copy()
        plot_data = handle_missing(plot_data, strategy="zero")
        
        plt.figure(figsize=(15, 10))
        sns.pairplot(plot_data, diag_kind='kde')
        plt.suptitle('Feature Relationships Pairplot', y=1.02)
        plt.show()
        
    except Exception as e:
        print(f"   ⚠️ Could not create pairplot: {e}")
        # Fallback to correlation matrix
        corr_subset = features[key_features[:6]].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_subset, annot=True, cmap='RdBu_r', center=0)
        plt.title('Key Features Correlation Matrix')
        plt.tight_layout()
        plt.show()

def analyze_feature_importance(features):
    """Analyze feature importance using correlation with GSI."""
    print("\n⭐ Analyzing feature importance...")
    
    if 'game_strength_index' not in features.columns:
        print("   ❌ Game Strength Index not found for importance analysis")
        return
    
    # Get numeric features excluding GSI
    numeric_features = features.select_dtypes(include=[np.number])
    if 'game_strength_index' in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=['game_strength_index'])
    
    # Calculate correlations with GSI
    gsi_correlations = numeric_features.corrwith(features['game_strength_index']).abs().sort_values(ascending=False)
    
    # Top features
    print(f"   Top 10 features correlated with GSI:")
    for i, (feature, corr) in enumerate(gsi_correlations.head(10).items()):
        print(f"     {i+1:2d}. {feature}: {corr:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    top_features = gsi_correlations.head(15)
    
    plt.barh(range(len(top_features)), top_features.values)
    plt.yticks(range(len(top_features)), top_features.index)
    plt.xlabel('Absolute Correlation with GSI')
    plt.title('Top 15 Features by Correlation with Game Strength Index')
    plt.grid(True, alpha=0.3)
    
    # Add correlation values as text
    for i, (feature, corr) in enumerate(top_features.items()):
        plt.text(corr + 0.01, i, f'{corr:.3f}', va='center')
    
    plt.tight_layout()
    plt.show()

def save_exploration_results(features, output_dir="data"):
    """Save exploration results and sample data."""
    print("\n💾 Saving exploration results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sample features
    sample_file = os.path.join(output_dir, "feature_exploration_sample.csv")
    features.head(50).to_csv(sample_file, index=False)
    print(f"   Sample features saved to: {sample_file}")
    
    # Save feature summary
    summary_file = os.path.join(output_dir, "feature_exploration_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Feature Exploration Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total features: {len(features.columns)}\n")
        f.write(f"Total games: {len(features)}\n")
        f.write(f"Dataset shape: {features.shape}\n\n")
        
        # Feature categories
        feature_categories = {
            'Team Features': [col for col in features.columns if 'team_' in col],
            'Player Features': [col for col in features.columns if any(x in col for x in ['injury', 'foul', 'bench', 'minutes'])],
            'Market Features': [col for col in features.columns if any(x in col for x in ['movement', 'market', 'clv', 'edge'])],
            'Dynamic Features': [col for col in features.columns if any(x in col for x in ['streak', 'rest', 'travel', 'altitude'])]
        }
        
        for category, cols in feature_categories.items():
            f.write(f"{category}: {len(cols)} features\n")
        
        f.write(f"\nColumns:\n")
        for col in features.columns:
            f.write(f"  {col}\n")
    
    print(f"   Summary saved to: {summary_file}")

def main():
    """Main exploration function."""
    print("🚀 CBB Betting ML System - Feature Exploration")
    print("=" * 60)
    
    try:
        # Setup
        setup_plotting()
        
        # Load data
        games_df, odds_df, players_df = load_sample_data()
        
        # Run feature pipeline
        features, pipeline = run_feature_pipeline(games_df, odds_df, players_df)
        
        # Explore features
        feature_categories = explore_feature_categories(features)
        
        # Analyze statistics
        missing_summary = analyze_feature_statistics(features)
        
        # Explore GSI
        explore_game_strength_index(features)
        
        # Create visualizations
        create_correlation_heatmap(features)
        explore_feature_relationships(features)
        analyze_feature_importance(features)
        
        # Save results
        save_exploration_results(features)
        
        print("\n✅ Feature exploration complete!")
        print(f"📊 Generated {len(features.columns)} features for {len(features)} games")
        print("📁 Results saved to data/ directory")
        
    except Exception as e:
        print(f"\n❌ Error during feature exploration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()