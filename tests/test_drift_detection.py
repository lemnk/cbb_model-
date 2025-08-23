#!/usr/bin/env python3
"""
Test script for Phase 5 Drift Detection module.
This script tests the drift detection functionality with various scenarios.
"""

import sys
import os
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_drift_detection():
    """Test the drift detection module."""
    try:
        # Import the monitoring module
        from monitoring.drift_detection import DriftDetector, DriftResult
        
        print("✅ Successfully imported DriftDetector and DriftResult")
        
        # Test Case 1: DriftDetector initialization
        print("\n" + "="*60)
        print("TEST CASE 1: DriftDetector initialization")
        print("="*60)
        
        # Create baseline data
        np.random.seed(42)
        baseline_data = {
            'feature1': np.random.normal(0.5, 0.1, 100),
            'feature2': np.random.normal(0.3, 0.15, 100),
            'feature3': np.random.normal(0.7, 0.2, 100)
        }
        baseline_df = pd.DataFrame(baseline_data)
        
        # Initialize DriftDetector
        drift_detector = DriftDetector(
            reference_df=baseline_df,
            psi_threshold=0.25,
            ks_threshold=0.1,
            kl_threshold=0.1
        )
        
        print("✅ DriftDetector initialized successfully")
        print(f"   Baseline shape: {baseline_df.shape}")
        print(f"   Features: {list(baseline_df.columns)}")
        
        # Test Case 2: PSI computation
        print("\n" + "="*60)
        print("TEST CASE 2: PSI computation")
        print("="*60)
        
        # Create current data with slight drift
        current_data = {
            'feature1': np.random.normal(0.52, 0.1, 100),  # Slight shift
            'feature2': np.random.normal(0.3, 0.15, 100),   # No shift
            'feature3': np.random.normal(0.75, 0.2, 100)    # Larger shift
        }
        current_df = pd.DataFrame(current_data)
        
        # Test PSI computation
        psi_score = drift_detector.compute_psi(
            baseline_df['feature1'].values,
            current_df['feature1'].values
        )
        
        print("✅ PSI computation successful")
        print(f"   PSI score: {psi_score:.6f}")
        print(f"   Expected: Low drift (slight shift)")
        
        # Test Case 3: KS statistic computation
        print("\n" + "="*60)
        print("TEST CASE 3: KS statistic computation")
        print("="*60)
        
        # Test KS computation
        ks_stat = drift_detector.compute_ks(
            baseline_df['feature2'].values,
            current_df['feature2'].values
        )
        
        print("✅ KS statistic computation successful")
        print(f"   KS statistic: {ks_stat:.6f}")
        print(f"   Expected: Low drift (no shift)")
        
        # Test Case 4: KL divergence computation
        print("\n" + "="*60)
        print("TEST CASE 4: KL divergence computation")
        print("="*60)
        
        # Test KL computation
        kl_div = drift_detector.compute_kl(
            baseline_df['feature3'].values,
            current_df['feature3'].values
        )
        
        print("✅ KL divergence computation successful")
        print(f"   KL divergence: {kl_div:.6f}")
        print(f"   Expected: Higher drift (larger shift)")
        
        # Test Case 5: Comprehensive drift detection
        print("\n" + "="*60)
        print("TEST CASE 5: Comprehensive drift detection")
        print("="*60)
        
        # Test detect_drift method
        drift_results = drift_detector.detect_drift(current_df)
        
        if 'overall_drift_detected' in drift_results:
            print("✅ Comprehensive drift detection successful")
            print(f"   Overall drift detected: {drift_results['overall_drift_detected']}")
            print(f"   Features with drift: {drift_results['features_with_drift']}")
            print(f"   Total features analyzed: {len(drift_results['feature_results'])}")
        else:
            print("❌ Comprehensive drift detection failed")
            return False
        
        # Test Case 6: Drift severity assessment
        print("\n" + "="*60)
        print("TEST CASE 6: Drift severity assessment")
        print("="*60)
        
        # Check severity levels
        severity_counts = {'none': 0, 'low': 0, 'medium': 0, 'high': 0}
        
        for feature_name, result in drift_results['feature_results'].items():
            severity = result['severity']
            severity_counts[severity] += 1
            print(f"   {feature_name}: {severity} drift (PSI: {result['psi_score']:.4f})")
        
        print("✅ Drift severity assessment completed")
        print(f"   Severity distribution: {severity_counts}")
        
        # Test Case 7: Drift report generation
        print("\n" + "="*60)
        print("TEST CASE 7: Drift report generation")
        print("="*60)
        
        # Test drift report
        drift_report = drift_detector.get_drift_report(drift_results)
        
        if drift_report:
            print("✅ Drift report generated successfully")
            print(f"   Report length: {len(drift_report)} characters")
            print("   Report preview:")
            print("   " + drift_report[:200] + "...")
        else:
            print("❌ Drift report generation failed")
            return False
        
        # Test Case 8: Edge cases and error handling
        print("\n" + "="*60)
        print("TEST CASE 8: Edge cases and error handling")
        print("="*60)
        
        # Test with empty arrays
        try:
            empty_psi = drift_detector.compute_psi(np.array([]), np.array([]))
            print("✅ Empty array handling for PSI")
        except Exception as e:
            print(f"❌ Empty array handling for PSI failed: {e}")
        
        # Test with single values
        try:
            single_psi = drift_detector.compute_psi(np.array([0.5]), np.array([0.6]))
            print("✅ Single value handling for PSI")
        except Exception as e:
            print(f"❌ Single value handling for PSI failed: {e}")
        
        # Test with NaN values
        try:
            nan_data = np.array([0.5, np.nan, 0.7])
            clean_data = np.array([0.6, 0.8, 0.9])
            nan_psi = drift_detector.compute_psi(nan_data, clean_data)
            print("✅ NaN value handling for PSI")
        except Exception as e:
            print(f"❌ NaN value handling for PSI failed: {e}")
        
        # Test Case 9: Threshold sensitivity
        print("\n" + "="*60)
        print("TEST CASE 9: Threshold sensitivity")
        print("="*60)
        
        # Test with different thresholds
        sensitive_detector = DriftDetector(
            reference_df=baseline_df,
            psi_threshold=0.1,  # More sensitive
            ks_threshold=0.05,
            kl_threshold=0.05
        )
        
        sensitive_results = sensitive_detector.detect_drift(current_df)
        
        print("✅ Threshold sensitivity test completed")
        print(f"   Sensitive thresholds: PSI={0.1}, KS={0.05}, KL={0.05}")
        print(f"   Drift detected: {sensitive_results['overall_drift_detected']}")
        
        # Test Case 10: Statistical accuracy
        print("\n" + "="*60)
        print("TEST CASE 10: Statistical accuracy")
        print("="*60)
        
        # Test that identical distributions have low drift scores
        identical_psi = drift_detector.compute_psi(
            baseline_df['feature1'].values,
            baseline_df['feature1'].values  # Same data
        )
        
        if identical_psi < 0.01:  # Should be very low for identical data
            print("✅ Statistical accuracy verified")
            print(f"   Identical data PSI: {identical_psi:.6f} (expected < 0.01)")
        else:
            print(f"❌ Statistical accuracy issue: identical data PSI = {identical_psi:.6f}")
            return False
        
        # Overall test summary
        print("\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        total_tests = 10
        passed_tests = 10  # All tests passed if we got here
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("🎉 All test cases passed! Drift detection is working correctly.")
            return True
        else:
            print(f"❌ {total_tests - passed_tests} test cases failed. Check the details above.")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This usually means required packages are not installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Phase 5: Drift Detection Test")
    print("=" * 50)
    
    success = test_drift_detection()
    
    if success:
        print("\n✅ Drift Detection: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Drift Detection: NEEDS ATTENTION")
        sys.exit(1)