#!/usr/bin/env python3
"""
Test script for Phase 2: Feature Engineering
Verifies that all required files are created and can be imported.
"""

import os
import sys
import importlib

def test_file_structure():
    """Test that all required Phase 2 files exist."""
    print("🔍 Testing Phase 2 file structure...")
    
    required_files = [
        "src/features/__init__.py",
        "src/features/team_features.py",
        "src/features/player_features.py",
        "src/features/dynamic_features.py",
        "src/features/market_features.py",
        "src/features/feature_utils.py",
        "src/features/feature_pipeline.py",
        "PHASE2_SUMMARY.md",
        "notebooks/feature_exploration.py"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"   ✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"   ❌ {file_path}")
    
    print(f"\n📊 File Structure Summary:")
    print(f"   Existing: {len(existing_files)}/{len(required_files)}")
    print(f"   Missing: {len(missing_files)}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False
    else:
        print(f"\n✅ All required files exist!")
        return True

def test_imports():
    """Test that all Phase 2 modules can be imported."""
    print("\n🔧 Testing Phase 2 imports...")
    
    # Add src to path
    sys.path.append('src')
    
    modules_to_test = [
        "features.team_features",
        "features.player_features", 
        "features.dynamic_features",
        "features.market_features",
        "features.feature_utils",
        "features.feature_pipeline"
    ]
    
    failed_imports = []
    successful_imports = []
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            successful_imports.append(module_name)
            print(f"   ✅ {module_name}")
        except Exception as e:
            failed_imports.append((module_name, str(e)))
            print(f"   ❌ {module_name}: {e}")
    
    print(f"\n📊 Import Summary:")
    print(f"   Successful: {len(successful_imports)}/{len(modules_to_test)}")
    print(f"   Failed: {len(failed_imports)}")
    
    if failed_imports:
        print(f"\n❌ Failed imports:")
        for module_name, error in failed_imports:
            print(f"   {module_name}: {error}")
        return False
    else:
        print(f"\n✅ All modules imported successfully!")
        return True

def test_class_instantiation():
    """Test that all feature classes can be instantiated."""
    print("\n🏗️ Testing class instantiation...")
    
    try:
        from features.team_features import TeamFeatures
        from features.player_features import PlayerFeatures
        from features.dynamic_features import DynamicFeatures
        from features.market_features import MarketFeatures
        from features.feature_pipeline import FeaturePipeline
        
        # Test instantiation
        team_features = TeamFeatures()
        player_features = PlayerFeatures()
        dynamic_features = DynamicFeatures()
        market_features = MarketFeatures()
        feature_pipeline = FeaturePipeline()
        
        print("   ✅ All feature classes instantiated successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ Class instantiation failed: {e}")
        return False

def test_feature_pipeline():
    """Test basic feature pipeline functionality."""
    print("\n🚀 Testing feature pipeline...")
    
    try:
        from features.feature_pipeline import FeaturePipeline
        
        # Initialize pipeline
        pipeline = FeaturePipeline()
        print("   ✅ Pipeline initialized")
        
        # Test sample data loading
        games_df, odds_df, players_df = pipeline.load_sample_data()
        print(f"   ✅ Sample data loaded: Games={games_df.shape}, Odds={odds_df.shape}, Players={players_df.shape}")
        
        # Test feature building (with small sample)
        sample_games = games_df.head(10)
        sample_odds = odds_df.head(10)
        sample_players = players_df.head(20)
        
        features = pipeline.build_features(sample_games, sample_odds, sample_players)
        print(f"   ✅ Features built: {features.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_utils():
    """Test feature utility functions."""
    print("\n🛠️ Testing feature utilities...")
    
    try:
        import pandas as pd
        import numpy as np
        from features.feature_utils import normalize, scale, handle_missing
        
        # Create test data
        test_df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'with_nulls': [1, np.nan, 3, np.nan, 5],
            'categorical': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Test normalize
        normalized = normalize(test_df['numeric'], method="minmax")
        print("   ✅ normalize function works")
        
        # Test scale
        scaled = scale(test_df, columns=['numeric'], method="zscore")
        print("   ✅ scale function works")
        
        # Test handle_missing
        cleaned = handle_missing(test_df, strategy="zero")
        print("   ✅ handle_missing function works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature utilities test failed: {e}")
        return False

def main():
    """Run all Phase 2 tests."""
    print("🧪 Phase 2: Feature Engineering - System Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Module Imports", test_imports),
        ("Class Instantiation", test_class_instantiation),
        ("Feature Pipeline", test_feature_pipeline),
        ("Feature Utilities", test_feature_utils)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PHASE 2 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Phase 2 is ready for use!")
        print("🚀 All feature engineering components are working correctly.")
    else:
        print(f"\n⚠️ {total - passed} tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)