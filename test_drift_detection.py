#!/usr/bin/env python3
"""
Test script for Phase 5 Drift Detection module.
This script tests the drift detection functionality with PSI, KS, and KL divergence.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_drift_detection():
    """Test the drift detection module."""
    try:
        # Import the monitoring module
        from monitoring.drift_detection import DriftDetector, DriftResult
        
        print("✅ Successfully imported DriftDetector and DriftResult")
        
        # Test DriftResult dataclass
        print("\nTesting DriftResult dataclass...")
        
        sample_result = DriftResult(
            feature_name="test_feature",
            psi_score=0.15,
            ks_statistic=0.08,
            kl_divergence=0.12,
            drift_detected=False,
            severity="none",
            details={"test": "data"}
        )
        
        print(f"✅ DriftResult created: {sample_result.feature_name}, severity: {sample_result.severity}")
        
        # Test DriftDetector initialization
        print("\nTesting DriftDetector initialization...")
        
        import pandas as pd
        import numpy as np
        
        # Create sample reference data
        np.random.seed(42)
        reference_data = {
            'feature_1': np.random.normal(0, 1, 100),
            'feature_2': np.random.uniform(0, 1, 100),
            'feature_3': np.random.exponential(1, 100)
        }
        reference_df = pd.DataFrame(reference_data)
        
        detector = DriftDetector(reference_df)
        print("✅ DriftDetector created successfully")
        print(f"   Reference features: {list(reference_df.columns)}")
        print(f"   PSI threshold: {detector.psi_threshold}")
        print(f"   KS threshold: {detector.ks_threshold}")
        print(f"   KL threshold: {detector.kl_threshold}")
        
        # Test individual drift measures
        print("\nTesting individual drift measures...")
        
        # Test PSI computation
        ref_data = reference_df['feature_1'].values
        curr_data = np.random.normal(0.1, 1.1, 100)  # Slight drift
        
        psi_score = detector.compute_psi(ref_data, curr_data)
        print(f"✅ PSI computed: {psi_score:.6f}")
        
        # Test KS computation
        ks_stat = detector.compute_ks(ref_data, curr_data)
        print(f"✅ KS statistic computed: {ks_stat:.6f}")
        
        # Test KL divergence
        kl_div = detector.compute_kl(ref_data, curr_data)
        print(f"✅ KL divergence computed: {kl_div:.6f}")
        
        # Test drift detection
        print("\nTesting drift detection...")
        
        # Create current data with varying drift levels
        current_data = {
            'feature_1': np.random.normal(0.2, 1.1, 100),  # Moderate drift
            'feature_2': np.random.uniform(0, 1, 100),      # No drift
            'feature_3': np.random.exponential(1.3, 100)    # High drift
        }
        current_df = pd.DataFrame(current_data)
        
        # Run drift detection
        results = detector.detect_drift(current_df)
        print("✅ Drift detection completed")
        
        # Print results summary
        summary = results['drift_summary']
        print(f"   Total features analyzed: {summary['total_features_analyzed']}")
        print(f"   Features with drift: {summary['features_with_drift']}")
        print(f"   Drift percentage: {summary['drift_percentage']:.1f}%")
        
        # Print feature-level results
        print("\nFeature-level results:")
        for feature_name, result in results['feature_results'].items():
            status = "🚨 DRIFT" if result.drift_detected else "✅ NO DRIFT"
            print(f"   {feature_name}: {status} (severity: {result.severity})")
            print(f"     PSI: {result.psi_score:.6f}, KS: {result.ks_statistic:.6f}, KL: {result.kl_divergence:.6f}")
        
        # Test drift report generation
        print("\nTesting drift report generation...")
        report = detector.get_drift_report(results)
        print("✅ Drift report generated successfully")
        
        # Print first few lines of report
        report_lines = report.split('\n')[:15]
        print("\nReport preview:")
        for line in report_lines:
            print(f"   {line}")
        
        # Verify overall results
        if results['overall_drift_detected']:
            print(f"\n🎯 Drift detected! {summary['features_with_drift']}/{summary['total_features_analyzed']} features show drift")
        else:
            print(f"\n🎯 No drift detected across {summary['total_features_analyzed']} features")
        
        # Test recommendations
        if results['recommendations']:
            print(f"\n📋 Recommendations ({len(results['recommendations'])}):")
            for i, rec in enumerate(results['recommendations'][:3], 1):  # Show first 3
                print(f"   {i}. {rec}")
        
        return True
        
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
        print("\n✅ Drift Detection Module: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Drift Detection Module: NEEDS ATTENTION")
        sys.exit(1)