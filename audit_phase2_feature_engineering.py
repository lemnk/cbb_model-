#!/usr/bin/env python3
"""
Phase 2 Enterprise Audit: Feature Engineering
Comprehensive audit of feature pipeline, leakage detection, and statistical validation.
"""

import sys
import os
import time
import json
import hashlib
import psutil
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Add cbb_model/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cbb_model', 'src'))

def audit_feature_validity():
    """Audit feature validity and formula implementation."""
    print("="*80)
    print("🔍 FEATURE VALIDITY AUDIT")
    print("="*80)
    
    try:
        from features.team_features import TeamFeatures
        from features.player_features import PlayerFeatures
        from features.dynamic_features import DynamicFeatures
        from features.market_features import MarketFeatures
        from features.feature_pipeline import FeaturePipeline
        
        print("✅ Feature classes imported successfully")
        
        # 1. FEATURE CLASS IMPLEMENTATION CHECK
        print("\n1. FEATURE CLASS IMPLEMENTATION CHECK")
        
        feature_classes = {
            'TeamFeatures': TeamFeatures,
            'PlayerFeatures': PlayerFeatures,
            'DynamicFeatures': DynamicFeatures,
            'MarketFeatures': MarketFeatures,
            'FeaturePipeline': FeaturePipeline
        }
        
        implementation_status = {}
        
        for class_name, feature_class in feature_classes.items():
            # Check if class has required methods
            has_transform = hasattr(feature_class, 'transform')
            has_init = hasattr(feature_class, '__init__')
            
            # Check if transform method returns DataFrame
            if has_transform:
                try:
                    instance = feature_class()
                    test_df = pd.DataFrame({'test': [1, 2, 3]})
                    result = instance.transform(test_df)
                    returns_dataframe = isinstance(result, pd.DataFrame)
                except Exception as e:
                    returns_dataframe = False
                    transform_error = str(e)
            else:
                returns_dataframe = False
            
            implementation_status[class_name] = {
                'has_init': has_init,
                'has_transform': has_transform,
                'returns_dataframe': returns_dataframe,
                'fully_implemented': has_init and has_transform and returns_dataframe
            }
            
            status = "✅" if implementation_status[class_name]['fully_implemented'] else "❌"
            print(f"  {status} {class_name}: transform={has_transform}, returns_df={returns_dataframe}")
        
        # 2. FEATURE FORMULA VERIFICATION
        print("\n2. FEATURE FORMULA VERIFICATION")
        
        # Test TeamFeatures implementation
        team_features = TeamFeatures()
        
        # Create test data
        test_data = {
            'game_id': [f'game_{i}' for i in range(100)],
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'home_team': ['Duke', 'Kansas', 'Michigan'] * 33 + ['Duke'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'] * 33 + ['UNC'],
            'home_score': np.random.randint(60, 100, 100),
            'away_score': np.random.randint(60, 100, 100)
        }
        
        test_df = pd.DataFrame(test_data)
        
        # Test feature computation
        try:
            features_df = team_features.transform(test_df)
            
            # Check if expected features were created
            expected_features = [
                'team_offensive_efficiency',
                'team_defensive_efficiency', 
                'team_pace',
                'team_home_win_pct',
                'team_away_win_pct',
                'team_scoring_consistency_3g',
                'team_scoring_consistency_10g'
            ]
            
            created_features = []
            missing_features = []
            
            for feature in expected_features:
                if feature in features_df.columns:
                    created_features.append(feature)
                    # Check feature statistics
                    feature_values = features_df[feature].dropna()
                    if len(feature_values) > 0:
                        print(f"    ✅ {feature}: mean={feature_values.mean():.3f}, std={feature_values.std():.3f}")
                    else:
                        print(f"    ⚠️ {feature}: No valid values")
                else:
                    missing_features.append(feature)
                    print(f"    ❌ {feature}: Not created")
            
            print(f"\n  Features created: {len(created_features)}/{len(expected_features)}")
            
            if missing_features:
                print(f"  Missing features: {missing_features}")
                formula_verification = False
            else:
                print("  ✅ All expected features created")
                formula_verification = True
                
        except Exception as e:
            print(f"  ❌ Feature computation failed: {e}")
            formula_verification = False
        
        # 3. FEATURE COMPUTATION ACCURACY
        print("\n3. FEATURE COMPUTATION ACCURACY")
        
        # Test with known data to verify calculations
        accuracy_test = False
        
        try:
            # Create simple test case
            simple_data = pd.DataFrame({
                'game_id': ['test_1', 'test_2'],
                'home_score': [80, 90],
                'away_score': [70, 85]
            })
            
            simple_features = team_features.transform(simple_data)
            
            # Check if features are reasonable
            if 'team_offensive_efficiency' in simple_features.columns:
                off_eff = simple_features['team_offensive_efficiency'].iloc[0]
                if 50 <= off_eff <= 150:  # Reasonable range
                    print("  ✅ Offensive efficiency in reasonable range")
                    accuracy_test = True
                else:
                    print(f"  ⚠️ Offensive efficiency out of range: {off_eff}")
            else:
                print("  ❌ Offensive efficiency feature not found")
                
        except Exception as e:
            print(f"  ❌ Accuracy test failed: {e}")
        
        # Summary
        print("\n" + "="*50)
        print("FEATURE VALIDITY AUDIT SUMMARY")
        print("="*50)
        
        total_checks = len(feature_classes) + 2  # classes + formula + accuracy
        passed_checks = sum(1 for status in implementation_status.values() if status['fully_implemented'])
        passed_checks += (1 if formula_verification else 0) + (1 if accuracy_test else 0)
        
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {total_checks - passed_checks}")
        
        if passed_checks >= total_checks * 0.8:  # 80% threshold
            print("🎉 Feature validity audit PASSED")
            return True
        else:
            print("⚠️ Feature validity audit has issues")
            return False
            
    except Exception as e:
        print(f"❌ Feature validity audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_leakage_detection():
    """Audit for data leakage in feature engineering."""
    print("\n" + "="*80)
    print("🚫 LEAKAGE DETECTION AUDIT")
    print("="*80)
    
    try:
        from features.team_features import TeamFeatures
        
        print("✅ Feature classes imported for leakage audit")
        
        # 1. TRAIN/TEST SPLIT SIMULATION
        print("\n1. TRAIN/TEST SPLIT SIMULATION")
        
        # Create time-series data
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=200, freq='D')
        
        time_series_data = {
            'game_id': [f'game_{i}' for i in range(200)],
            'date': dates,
            'home_team': ['Duke', 'Kansas', 'Michigan'] * 66 + ['Duke', 'Kansas'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'] * 66 + ['UNC', 'Kentucky'],
            'home_score': np.random.randint(60, 100, 200),
            'away_score': np.random.randint(60, 100, 200),
            'target': np.random.randint(0, 2, 200)  # Binary outcome
        }
        
        df = pd.DataFrame(time_series_data)
        
        # Split by time (no future data leakage)
        split_date = '2024-04-01'
        train_mask = df['date'] < split_date
        test_mask = df['date'] >= split_date
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"  Training set: {len(train_df)} samples (before {split_date})")
        print(f"  Test set: {len(test_df)} samples (after {split_date})")
        
        # 2. FEATURE COMPUTATION WITHOUT LEAKAGE
        print("\n2. FEATURE COMPUTATION WITHOUT LEAKAGE")
        
        team_features = TeamFeatures()
        
        # Compute features on training data only
        train_features = team_features.transform(train_df)
        test_features = team_features.transform(test_df)
        
        print(f"  Training features shape: {train_features.shape}")
        print(f"  Test features shape: {test_features.shape}")
        
        # 3. TARGET LEAKAGE CHECK
        print("\n3. TARGET LEAKAGE CHECK")
        
        target_leakage_detected = False
        
        # Check if target column appears in features
        if 'target' in train_features.columns:
            print("  ❌ Target column found in training features - LEAKAGE DETECTED!")
            target_leakage_detected = True
        else:
            print("  ✅ Target column not in training features")
        
        # Check for other potential leakage indicators
        leakage_indicators = []
        
        # Check for future date information
        if 'date' in train_features.columns:
            max_train_date = train_features['date'].max()
            min_test_date = test_features['date'].min()
            
            if max_train_date >= min_test_date:
                leakage_indicators.append("Date overlap between train/test")
                print("  ⚠️ Date overlap detected - potential leakage")
            else:
                print("  ✅ No date overlap - proper temporal split")
        
        # 4. ADVERSARIAL VALIDATION
        print("\n4. ADVERSARIAL VALIDATION")
        
        try:
            # Prepare data for adversarial validation
            train_features_clean = train_features.drop(['date', 'game_id'], axis=1, errors='ignore')
            test_features_clean = test_features.drop(['date', 'game_id'], axis=1, errors='ignore')
            
            # Remove any non-numeric columns
            numeric_cols = train_features_clean.select_dtypes(include=[np.number]).columns
            train_features_numeric = train_features_clean[numeric_cols].fillna(0)
            test_features_numeric = test_features_clean[numeric_cols].fillna(0)
            
            if len(numeric_cols) > 0 and len(train_features_numeric) > 10:
                # Create labels: 0 for train, 1 for test
                train_labels = np.zeros(len(train_features_numeric))
                test_labels = np.ones(len(test_features_numeric))
                
                # Combine data
                X_combined = pd.concat([train_features_numeric, test_features_numeric], ignore_index=True)
                y_combined = np.concatenate([train_labels, test_labels])
                
                # Train model to distinguish train vs test
                rf = RandomForestClassifier(n_estimators=10, random_state=42)
                rf.fit(X_combined, y_combined)
                
                # Predict probabilities
                y_pred_proba = rf.predict_proba(X_combined)[:, 1]
                auc_score = roc_auc_score(y_combined, y_pred_proba)
                
                print(f"  Adversarial validation AUC: {auc_score:.3f}")
                
                if auc_score > 0.7:
                    print("  ⚠️ High AUC - potential data leakage detected")
                    leakage_indicators.append(f"Adversarial validation AUC: {auc_score:.3f}")
                elif auc_score > 0.6:
                    print("  ⚠️ Moderate AUC - possible data leakage")
                    leakage_indicators.append(f"Adversarial validation AUC: {auc_score:.3f}")
                else:
                    print("  ✅ Low AUC - no obvious data leakage")
                    
            else:
                print("  ⚠️ Insufficient numeric features for adversarial validation")
                
        except Exception as e:
            print(f"  ❌ Adversarial validation failed: {e}")
            leakage_indicators.append(f"Adversarial validation error: {e}")
        
        # Summary
        print("\n" + "="*50)
        print("LEAKAGE DETECTION AUDIT SUMMARY")
        print("="*50)
        
        if target_leakage_detected or leakage_indicators:
            print("❌ DATA LEAKAGE DETECTED!")
            print("Issues found:")
            if target_leakage_detected:
                print("  - Target column in training features")
            for indicator in leakage_indicators:
                print(f"  - {indicator}")
            return False
        else:
            print("🎉 No data leakage detected")
            return True
            
    except Exception as e:
        print(f"❌ Leakage detection audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_statistical_validation():
    """Audit feature statistical properties and distributions."""
    print("\n" + "="*80)
    print("📊 STATISTICAL VALIDATION AUDIT")
    print("="*80)
    
    try:
        from features.team_features import TeamFeatures
        
        print("✅ Feature classes imported for statistical audit")
        
        # 1. FEATURE DISTRIBUTION ANALYSIS
        print("\n1. FEATURE DISTRIBUTION ANALYSIS")
        
        # Create test data
        np.random.seed(42)
        test_data = {
            'game_id': [f'game_{i}' for i in range(500)],
            'date': pd.date_range('2024-01-01', periods=500, freq='D'),
            'home_team': ['Duke', 'Kansas', 'Michigan', 'UNC', 'Kentucky'] * 100,
            'away_team': ['Ohio State', 'Wisconsin', 'Purdue', 'Indiana', 'Michigan State'] * 100,
            'home_score': np.random.randint(60, 100, 500),
            'away_score': np.random.randint(60, 100, 500)
        }
        
        df = pd.DataFrame(test_data)
        team_features = TeamFeatures()
        
        # Generate features
        try:
            features_df = team_features.transform(df)
            
            # Analyze numeric features
            numeric_features = features_df.select_dtypes(include=[np.number])
            
            if len(numeric_features.columns) == 0:
                print("  ⚠️ No numeric features found for analysis")
                return False
            
            print(f"  Analyzing {len(numeric_features.columns)} numeric features")
            
            distribution_issues = []
            outlier_issues = []
            constant_columns = []
            
            for col in numeric_features.columns:
                values = numeric_features[col].dropna()
                
                if len(values) == 0:
                    print(f"    ❌ {col}: No valid values")
                    continue
                
                # Basic statistics
                mean_val = values.mean()
                std_val = values.std()
                min_val = values.min()
                max_val = values.max()
                
                print(f"    📊 {col}: mean={mean_val:.3f}, std={std_val:.3f}, range=[{min_val:.3f}, {max_val:.3f}]")
                
                # Check for constant columns
                if std_val == 0:
                    constant_columns.append(col)
                    print(f"      ⚠️ {col}: Constant column (std=0)")
                
                # Check for outliers (beyond 3 standard deviations)
                outlier_threshold = 3
                outliers = values[(values < mean_val - outlier_threshold * std_val) | 
                                (values > mean_val + outlier_threshold * std_val)]
                
                if len(outliers) > 0:
                    outlier_pct = len(outliers) / len(values) * 100
                    outlier_issues.append(f"{col}: {outlier_pct:.1f}% outliers")
                    print(f"      ⚠️ {col}: {len(outliers)} outliers ({outlier_pct:.1f}%)")
                
                # Check for reasonable ranges based on feature type
                if 'efficiency' in col.lower():
                    if mean_val < 50 or mean_val > 150:
                        distribution_issues.append(f"{col}: Unreasonable efficiency range")
                        print(f"      ⚠️ {col}: Efficiency outside reasonable range [50, 150]")
                
                elif 'win_pct' in col.lower():
                    if mean_val < 0 or mean_val > 1:
                        distribution_issues.append(f"{col}: Win percentage outside [0, 1]")
                        print(f"      ❌ {col}: Win percentage outside [0, 1]")
                
                elif 'consistency' in col.lower():
                    if abs(mean_val) > 5:  # Consistency should be around 0
                        distribution_issues.append(f"{col}: Consistency mean too far from 0")
                        print(f"      ⚠️ {col}: Consistency mean {mean_val:.3f} (expected near 0)")
            
            # 2. CATEGORICAL FEATURE ENCODING
            print("\n2. CATEGORICAL FEATURE ENCODING")
            
            categorical_features = features_df.select_dtypes(include=['object', 'category'])
            
            if len(categorical_features.columns) > 0:
                print(f"  Analyzing {len(categorical_features.columns)} categorical features")
                
                encoding_issues = []
                
                for col in categorical_features.columns:
                    unique_values = categorical_features[col].nunique()
                    null_count = categorical_features[col].isnull().sum()
                    
                    print(f"    📊 {col}: {unique_values} unique values, {null_count} nulls")
                    
                    # Check for high cardinality
                    if unique_values > 100:
                        encoding_issues.append(f"{col}: High cardinality ({unique_values} values)")
                        print(f"      ⚠️ {col}: High cardinality - consider encoding")
                    
                    # Check for missing values
                    if null_count > 0:
                        null_pct = null_count / len(features_df) * 100
                        if null_pct > 10:
                            encoding_issues.append(f"{col}: High missing values ({null_pct:.1f}%)")
                            print(f"      ⚠️ {col}: {null_pct:.1f}% missing values")
            else:
                print("  No categorical features found")
            
            # 3. FEATURE CORRELATION ANALYSIS
            print("\n3. FEATURE CORRELATION ANALYSIS")
            
            if len(numeric_features.columns) > 1:
                correlation_matrix = numeric_features.corr()
                
                # Find high correlations
                high_correlations = []
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_val = correlation_matrix.iloc[i, j]
                        if abs(corr_val) > 0.95:
                            col1 = correlation_matrix.columns[i]
                            col2 = correlation_matrix.columns[j]
                            high_correlations.append(f"{col1} ↔ {col2}: {corr_val:.3f}")
                            print(f"    ⚠️ High correlation: {col1} ↔ {col2}: {corr_val:.3f}")
                
                if not high_correlations:
                    print("  ✅ No extremely high correlations detected")
            else:
                print("  ⚠️ Insufficient features for correlation analysis")
            
            # Summary
            print("\n" + "="*50)
            print("STATISTICAL VALIDATION AUDIT SUMMARY")
            print("="*50)
            
            total_issues = len(distribution_issues) + len(outlier_issues) + len(constant_columns) + len(encoding_issues)
            
            if total_issues == 0:
                print("🎉 Statistical validation audit PASSED")
                return True
            else:
                print(f"⚠️ Statistical validation audit has {total_issues} issues")
                return False
                
        except Exception as e:
            print(f"  ❌ Feature generation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Statistical validation audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_reproducibility():
    """Audit feature pipeline reproducibility."""
    print("\n" + "="*80)
    print("🔄 REPRODUCIBILITY AUDIT")
    print("="*80)
    
    try:
        from features.team_features import TeamFeatures
        
        print("✅ Feature classes imported for reproducibility audit")
        
        # 1. IDENTICAL INPUT REPRODUCIBILITY
        print("\n1. IDENTICAL INPUT REPRODUCIBILITY")
        
        # Create test data
        test_data = {
            'game_id': [f'game_{i}' for i in range(100)],
            'date': pd.date_range('2024-01-01', periods=100, freq='D'),
            'home_team': ['Duke', 'Kansas', 'Michigan'] * 33 + ['Duke'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'] * 33 + ['UNC'],
            'home_score': np.random.randint(60, 100, 100),
            'away_score': np.random.randint(60, 100, 100)
        }
        
        df = pd.DataFrame(test_data)
        team_features = TeamFeatures()
        
        # Run feature generation twice
        try:
            print("  Running feature generation twice with identical input...")
            
            # First run
            np.random.seed(42)
            features_1 = team_features.transform(df)
            hash_1 = hashlib.md5(features_1.to_string().encode()).hexdigest()
            
            # Second run
            np.random.seed(42)
            features_2 = team_features.transform(df)
            hash_2 = hashlib.md5(features_2.to_string().encode()).hexdigest()
            
            if hash_1 == hash_2:
                print("  ✅ Identical input reproducibility: PASSED")
                identical_reproducibility = True
            else:
                print("  ❌ Identical input reproducibility: FAILED")
                identical_reproducibility = False
                
        except Exception as e:
            print(f"  ❌ Identical input test failed: {e}")
            identical_reproducibility = False
        
        # 2. SHUFFLED INPUT INVARIANCE
        print("\n2. SHUFFLED INPUT INVARIANCE")
        
        try:
            print("  Testing invariance to input order...")
            
            # Shuffle the input data
            df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
            
            # Generate features on shuffled data
            np.random.seed(42)
            features_shuffled = team_features.transform(df_shuffled)
            
            # Sort both results by game_id for comparison
            features_1_sorted = features_1.sort_values('game_id').reset_index(drop=True)
            features_shuffled_sorted = features_shuffled.sort_values('game_id').reset_index(drop=True)
            
            # Compare sorted results
            if features_1_sorted.equals(features_shuffled_sorted):
                print("  ✅ Shuffled input invariance: PASSED")
                shuffled_invariance = True
            else:
                print("  ❌ Shuffled input invariance: FAILED")
                print("  Differences detected:")
                
                # Find differences
                for col in features_1_sorted.columns:
                    if not features_1_sorted[col].equals(features_shuffled_sorted[col]):
                        print(f"    Column {col} has differences")
                        
                shuffled_invariance = False
                
        except Exception as e:
            print(f"  ❌ Shuffled input test failed: {e}")
            shuffled_invariance = False
        
        # 3. SEED INDEPENDENCE
        print("\n3. SEED INDEPENDENCE")
        
        try:
            print("  Testing seed independence...")
            
            # Test with different seeds
            seeds = [42, 123, 456, 789, 999]
            results = []
            
            for seed in seeds:
                np.random.seed(seed)
                features_seed = team_features.transform(df)
                hash_seed = hashlib.md5(features_seed.to_string().encode()).hexdigest()
                results.append(hash_seed)
            
            # Check if all results are identical (deterministic)
            if len(set(results)) == 1:
                print("  ✅ Seed independence: PASSED (deterministic)")
                seed_independence = True
            else:
                print("  ⚠️ Seed independence: Results vary with seed")
                print(f"  Unique results: {len(set(results))}/{len(seeds)}")
                seed_independence = False
                
        except Exception as e:
            print(f"  ❌ Seed independence test failed: {e}")
            seed_independence = False
        
        # Summary
        print("\n" + "="*50)
        print("REPRODUCIBILITY AUDIT SUMMARY")
        print("="*50)
        
        total_tests = 3
        passed_tests = sum([identical_reproducibility, shuffled_invariance, seed_independence])
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        
        if passed_tests == total_tests:
            print("🎉 Reproducibility audit PASSED")
            return True
        else:
            print("⚠️ Reproducibility audit has issues")
            return False
            
    except Exception as e:
        print(f"❌ Reproducibility audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_performance_scaling():
    """Audit feature pipeline performance and scaling."""
    print("\n" + "="*80)
    print("🚀 PERFORMANCE & SCALING AUDIT")
    print("="*80)
    
    try:
        from features.team_features import TeamFeatures
        
        print("✅ Feature classes imported for performance audit")
        
        # 1. BASELINE PERFORMANCE TEST
        print("\n1. BASELINE PERFORMANCE TEST")
        
        # Test data sizes
        test_sizes = [100, 1000, 10000]
        performance_results = {}
        
        for size in test_sizes:
            print(f"\nTesting with {size} rows...")
            
            # Create test data
            test_data = {
                'game_id': [f'game_{i}' for i in range(size)],
                'date': pd.date_range('2024-01-01', periods=size, freq='D'),
                'home_team': ['Duke', 'Kansas', 'Michigan'] * (size // 3 + 1),
                'away_team': ['UNC', 'Kentucky', 'Ohio State'] * (size // 3 + 1),
                'home_score': np.random.randint(60, 100, size),
                'away_score': np.random.randint(60, 100, size)
            }
            
            test_df = pd.DataFrame(test_data)
            team_features = TeamFeatures()
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Generate features
                features_df = team_features.transform(test_df)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                performance_results[size] = {
                    'processing_time': processing_time,
                    'memory_used': memory_used,
                    'rows_processed': len(features_df),
                    'features_created': len(features_df.columns)
                }
                
                print(f"  Processing time: {processing_time:.3f}s")
                print(f"  Memory used: {memory_used:.2f} MB")
                print(f"  Rows processed: {len(features_df)}")
                print(f"  Features created: {len(features_df.columns)}")
                
            except Exception as e:
                print(f"  ❌ Performance test failed: {e}")
                performance_results[size] = {'error': str(e)}
        
        # 2. SCALING ANALYSIS
        print("\n2. SCALING ANALYSIS")
        
        if len(performance_results) >= 2:
            sizes = list(performance_results.keys())
            times = [r.get('processing_time', 0) for r in performance_results.values() if 'error' not in r]
            
            if len(times) >= 2:
                # Calculate scaling factor
                scaling_factor = times[-1] / times[0]
                size_factor = sizes[-1] / sizes[0]
                
                print(f"Size increase: {size_factor}x")
                print(f"Time increase: {scaling_factor:.2f}x")
                
                if scaling_factor <= size_factor * 1.5:  # Linear scaling with 50% tolerance
                    print("✅ Feature pipeline scales linearly (good)")
                    scaling_ok = True
                else:
                    print("⚠️ Feature pipeline scales poorly (may have bottlenecks)")
                    scaling_ok = False
            else:
                scaling_ok = False
        else:
            scaling_ok = False
        
        # 3. MEMORY EFFICIENCY
        print("\n3. MEMORY EFFICIENCY")
        
        memory_issues = []
        for size, result in performance_results.items():
            if 'error' not in result:
                memory_per_row = result['memory_used'] / result['rows_processed']
                if memory_per_row > 0.1:  # More than 100KB per row
                    memory_issues.append(f"{size} rows: {memory_per_row:.3f} MB/row")
                    print(f"  ⚠️ High memory usage: {memory_per_row:.3f} MB/row")
                else:
                    print(f"  ✅ {size} rows: {memory_per_row:.3f} MB/row")
        
        # 4. FEATURE EXPLOSION DETECTION
        print("\n4. FEATURE EXPLOSION DETECTION")
        
        feature_explosion_issues = []
        for size, result in performance_results.items():
            if 'error' not in result:
                features_per_row = result['features_created'] / result['rows_processed']
                if features_per_row > 0.5:  # More than 0.5 features per row
                    feature_explosion_issues.append(f"{size} rows: {features_per_row:.3f} features/row")
                    print(f"  ⚠️ High feature density: {features_per_row:.3f} features/row")
                else:
                    print(f"  ✅ {size} rows: {features_per_row:.3f} features/row")
        
        # Summary
        print("\n" + "="*50)
        print("PERFORMANCE AUDIT SUMMARY")
        print("="*50)
        
        if scaling_ok and not memory_issues and not feature_explosion_issues:
            print("🎉 Performance audit PASSED")
            return True
        else:
            print("⚠️ Performance audit has issues")
            if not scaling_ok:
                print("  - Scaling performance needs improvement")
            if memory_issues:
                print("  - Memory efficiency needs improvement")
            if feature_explosion_issues:
                print("  - Feature explosion detected")
            return False
            
    except Exception as e:
        print(f"❌ Performance audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_phase2_audit_report(audit_results: Dict[str, bool]):
    """Generate comprehensive Phase 2 audit report."""
    print("\n" + "="*80)
    print("📊 PHASE 2 ENTERPRISE AUDIT REPORT")
    print("="*80)
    
    # Calculate overall score
    total_audits = len(audit_results)
    passed_audits = sum(audit_results.values())
    overall_score = (passed_audits / total_audits) * 100
    
    # Determine overall status
    if overall_score >= 90:
        overall_status = "EXCELLENT"
        status_emoji = "🎉"
    elif overall_score >= 80:
        overall_status = "GOOD"
        status_emoji = "✅"
    elif overall_score >= 70:
        overall_status = "ACCEPTABLE"
        status_emoji = "⚠️"
    elif overall_score >= 60:
        overall_status = "NEEDS IMPROVEMENT"
        status_emoji = "🔧"
    else:
        overall_status = "CRITICAL ISSUES"
        status_emoji = "🚨"
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "audit_type": "Phase 2 Enterprise Audit: Feature Engineering",
        "system": "CBB Betting ML System",
        "overall_score": overall_score,
        "overall_status": overall_status,
        "audit_results": audit_results,
        "total_audits": total_audits,
        "passed_audits": passed_audits,
        "failed_audits": total_audits - passed_audits,
        "audit_components": {
            "Feature Validity": "Formula verification, implementation correctness",
            "Leakage Detection": "Train/test split, target leakage, adversarial validation",
            "Statistical Validation": "Distributions, outliers, categorical encoding",
            "Reproducibility": "Identical input, shuffled input, seed independence",
            "Performance & Scaling": "Runtime, memory, feature explosion detection"
        }
    }
    
    # Print summary
    print(f"\n{status_emoji} PHASE 2 ENTERPRISE AUDIT COMPLETED {status_emoji}")
    print(f"Overall Score: {overall_score:.1f}/100")
    print(f"Overall Status: {overall_status}")
    print(f"Audits Passed: {passed_audits}/{total_audits}")
    
    print("\nDetailed Results:")
    for audit_name, result in audit_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {audit_name}: {status}")
    
    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    
    if overall_score >= 90:
        print("🎉 Excellent feature engineering quality!")
        print("  - Continue current practices")
        print("  - Plan for future scaling")
        print("  - Maintain quality standards")
    elif overall_score >= 80:
        print("✅ Good feature engineering with minor issues")
        print("  - Address identified issues")
        print("  - Improve weak areas")
        print("  - Plan for optimization")
    elif overall_score >= 70:
        print("⚠️ Acceptable quality with notable issues")
        print("  - Prioritize critical fixes")
        print("  - Address data leakage concerns")
        print("  - Improve performance bottlenecks")
    elif overall_score >= 60:
        print("🔧 Feature engineering needs significant improvement")
        print("  - Immediate action required")
        print("  - Focus on critical issues")
        print("  - Consider architectural changes")
    else:
        print("🚨 Critical issues detected!")
        print("  - IMMEDIATE ACTION REQUIRED")
        print("  - Address all failed audits")
        print("  - Consider feature pipeline redesign")
    
    # Save report
    try:
        report_filename = f"phase2_enterprise_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Comprehensive report saved to: {report_filename}")
    except Exception as e:
        print(f"❌ Failed to save comprehensive report: {e}")
    
    return report

def run_phase2_enterprise_audit():
    """Run complete Phase 2 enterprise audit."""
    print("PHASE 2 ENTERPRISE AUDIT: FEATURE ENGINEERING")
    print("="*80)
    print("CBB Betting ML System - Phase 2")
    print("Feature Pipeline, Leakage Detection, Statistical Validation")
    print("="*80)
    print("Starting comprehensive Phase 2 audit...")
    
    start_time = time.time()
    
    # Run all audit components
    audit_results = {}
    
    # 1. Feature Validity Audit
    print("\n🔍 Starting Feature Validity Audit...")
    audit_results["Feature Validity"] = audit_feature_validity()
    
    # 2. Leakage Detection Audit
    print("\n🚫 Starting Leakage Detection Audit...")
    audit_results["Leakage Detection"] = audit_leakage_detection()
    
    # 3. Statistical Validation Audit
    print("\n📊 Starting Statistical Validation Audit...")
    audit_results["Statistical Validation"] = audit_statistical_validation()
    
    # 4. Reproducibility Audit
    print("\n🔄 Starting Reproducibility Audit...")
    audit_results["Reproducibility"] = audit_reproducibility()
    
    # 5. Performance & Scaling Audit
    print("\n🚀 Starting Performance & Scaling Audit...")
    audit_results["Performance & Scaling"] = audit_performance_scaling()
    
    # Calculate audit time
    audit_time = time.time() - start_time
    
    # Generate comprehensive report
    report = generate_phase2_audit_report(audit_results)
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏁 PHASE 2 ENTERPRISE AUDIT COMPLETED")
    print("="*80)
    print(f"Total Audit Time: {audit_time:.1f} seconds")
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Status: {report['overall_status']}")
    
    if report['overall_score'] >= 80:
        print("\n🎉 CONGRATULATIONS! Your Phase 2 feature engineering meets enterprise standards!")
        return True
    elif report['overall_score'] >= 70:
        print("\n⚠️ Your Phase 2 feature engineering is acceptable but needs improvements.")
        return True
    else:
        print("\n🚨 Your Phase 2 feature engineering has critical issues that must be addressed.")
        return False

if __name__ == "__main__":
    try:
        success = run_phase2_enterprise_audit()
        
        if success:
            print("\n✅ Phase 2 enterprise audit completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Phase 2 enterprise audit completed with critical issues!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Phase 2 audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Phase 2 enterprise audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)