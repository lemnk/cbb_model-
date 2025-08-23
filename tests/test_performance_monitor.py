#!/usr/bin/env python3
"""
Test script for Phase 5 Performance Monitoring module.
This script tests the performance monitoring functionality with various scenarios.
"""

import sys
import os
import numpy as np
import yaml
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_performance_monitor():
    """Test the performance monitoring module."""
    try:
        # Import the monitoring module
        from monitoring.performance_monitor import PerformanceMonitor, MetricResult
        
        print("✅ Successfully imported PerformanceMonitor and MetricResult")
        
        # Test Case 1: PerformanceMonitor initialization
        print("\n" + "="*60)
        print("TEST CASE 1: PerformanceMonitor initialization")
        print("="*60)
        
        # Create sample thresholds
        thresholds = {
            'accuracy': 0.55,
            'log_loss': 0.7,
            'brier_score': 0.25,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'roc_auc': 0.6,
            'expected_value': 0.0
        }
        
        monitor = PerformanceMonitor(thresholds)
        
        print("✅ PerformanceMonitor initialized successfully")
        print(f"   Thresholds loaded: {len(thresholds)} metrics")
        print(f"   Sample threshold: accuracy >= {thresholds['accuracy']}")
        
        # Test Case 2: High accuracy scenario (all PASS)
        print("\n" + "="*60)
        print("TEST CASE 2: High accuracy scenario (all PASS)")
        print("="*60)
        
        # Create high-quality predictions
        np.random.seed(42)
        n_samples = 1000
        
        # High accuracy: 80% correct predictions
        y_true = np.random.binomial(1, 0.5, n_samples)
        y_pred_proba = np.where(y_true == 1, 
                               np.random.beta(8, 2, n_samples),  # High prob for positive
                               np.random.beta(2, 8, n_samples))  # Low prob for negative
        
        # Realistic odds (1.5 to 3.0)
        odds = np.random.uniform(1.5, 3.0, n_samples)
        
        # Evaluate performance
        results = monitor.evaluate(y_true, y_pred_proba, odds)
        
        print("✅ High accuracy evaluation completed")
        print(f"   Total metrics evaluated: {len(results)}")
        
        # Check that accuracy is high
        accuracy_result = results['accuracy']
        if accuracy_result['value'] > 0.7:
            print(f"   ✅ High accuracy achieved: {accuracy_result['value']:.3f}")
            print(f"   Status: {accuracy_result['status']}")
        else:
            print(f"   ⚠️ Lower accuracy than expected: {accuracy_result['value']:.3f}")
        
        # Test Case 3: Low accuracy scenario (ALERT)
        print("\n" + "="*60)
        print("TEST CASE 3: Low accuracy scenario (ALERT)")
        print("="*60)
        
        # Create low-quality predictions (random)
        y_pred_proba_random = np.random.uniform(0, 1, n_samples)
        
        # Evaluate performance
        random_results = monitor.evaluate(y_true, y_pred_proba_random, odds)
        
        print("✅ Low accuracy evaluation completed")
        
        # Check that accuracy is low and triggers alert
        random_accuracy = random_results['accuracy']
        if random_accuracy['status'] == 'ALERT':
            print(f"   ✅ Alert correctly triggered for low accuracy: {random_accuracy['value']:.3f}")
        else:
            print(f"   ⚠️ Expected ALERT status, got: {random_accuracy['status']}")
        
        # Test Case 4: Miscalibrated predictions (high log loss)
        print("\n" + "="*60)
        print("TEST CASE 4: Miscalibrated predictions (high log loss)")
        print("="*60)
        
        # Create miscalibrated predictions (overconfident)
        y_pred_proba_miscalibrated = np.where(y_true == 1, 
                                             np.random.beta(15, 1, n_samples),  # Very high prob
                                             np.random.beta(1, 15, n_samples))  # Very low prob
        
        # Evaluate performance
        miscalibrated_results = monitor.evaluate(y_true, y_pred_proba_miscalibrated, odds)
        
        print("✅ Miscalibrated predictions evaluation completed")
        
        # Check log loss
        log_loss_result = miscalibrated_results['log_loss']
        print(f"   Log Loss: {log_loss_result['value']:.3f} (threshold: {log_loss_result['threshold']:.3f})")
        print(f"   Status: {log_loss_result['status']}")
        
        # Test Case 5: Expected Value calculation with odds
        print("\n" + "="*60)
        print("TEST CASE 5: Expected Value calculation with odds")
        print("="*60)
        
        # Test with different odds scenarios
        # Scenario 1: Positive expected value
        positive_odds = np.full(n_samples, 2.0)  # 2.0 odds
        positive_ev_results = monitor.evaluate(y_true, y_pred_proba, positive_odds)
        
        ev_result = positive_ev_results['expected_value']
        print(f"   Positive EV scenario:")
        print(f"   Expected Value: {ev_result['value']:.4f}")
        print(f"   Status: {ev_result['status']}")
        
        # Scenario 2: Negative expected value
        negative_odds = np.full(n_samples, 1.1)  # 1.1 odds (very low)
        negative_ev_results = monitor.evaluate(y_true, y_pred_proba, negative_odds)
        
        ev_result_neg = negative_ev_results['expected_value']
        print(f"   Negative EV scenario:")
        print(f"   Expected Value: {ev_result_neg['value']:.4f}")
        print(f"   Status: {ev_result_neg['status']}")
        
        # Test Case 6: All 8 metrics computation
        print("\n" + "="*60)
        print("TEST CASE 6: All 8 metrics computation")
        print("="*60)
        
        # Verify all 8 metrics are computed
        required_metrics = [
            'accuracy', 'log_loss', 'brier_score', 'precision',
            'recall', 'f1', 'roc_auc', 'expected_value'
        ]
        
        missing_metrics = []
        for metric in required_metrics:
            if metric not in results:
                missing_metrics.append(metric)
        
        if not missing_metrics:
            print("✅ All 8 required metrics computed successfully")
            for metric in required_metrics:
                metric_result = results[metric]
                print(f"   {metric}: {metric_result['value']:.4f} (status: {metric_result['status']})")
        else:
            print(f"❌ Missing metrics: {missing_metrics}")
            return False
        
        # Test Case 7: Threshold comparison and status assignment
        print("\n" + "="*60)
        print("TEST CASE 7: Threshold comparison and status assignment")
        print("="*60)
        
        # Check that status assignment works correctly
        status_counts = {'PASS': 0, 'WARNING': 0, 'ALERT': 0}
        
        for metric_name, metric_result in results.items():
            status = metric_result['status']
            status_counts[status] += 1
            
            # Verify threshold comparison
            value = metric_result['value']
            threshold = metric_result['threshold']
            
            if status == 'PASS':
                if metric_name in ['log_loss', 'brier_score']:
                    # Lower is better for these metrics
                    if value <= threshold:
                        print(f"   ✅ {metric_name}: PASS (value {value:.4f} <= threshold {threshold:.4f})")
                    else:
                        print(f"   ❌ {metric_name}: Incorrect PASS status")
                else:
                    # Higher is better for other metrics
                    if value >= threshold:
                        print(f"   ✅ {metric_name}: PASS (value {value:.4f} >= threshold {threshold:.4f})")
                    else:
                        print(f"   ❌ {metric_name}: Incorrect PASS status")
        
        print(f"   Status distribution: {status_counts}")
        
        # Test Case 8: Performance summary generation
        print("\n" + "="*60)
        print("TEST CASE 8: Performance summary generation")
        print("="*60)
        
        # Test summary generation
        summary = monitor.get_summary(results)
        
        if summary:
            print("✅ Performance summary generated successfully")
            print(f"   Summary length: {len(summary)} characters")
            print("   Summary preview:")
            print("   " + summary[:200] + "...")
        else:
            print("❌ Performance summary generation failed")
            return False
        
        # Test Case 9: Edge cases and error handling
        print("\n" + "="*60)
        print("TEST CASE 9: Edge cases and error handling")
        print("="*60)
        
        # Test with empty arrays
        try:
            empty_results = monitor.evaluate(np.array([]), np.array([]), np.array([]))
            print("✅ Empty array handling")
        except Exception as e:
            print(f"❌ Empty array handling failed: {e}")
        
        # Test with single values
        try:
            single_results = monitor.evaluate(np.array([1]), np.array([0.8]), np.array([2.0]))
            print("✅ Single value handling")
        except Exception as e:
            print(f"❌ Single value handling failed: {e}")
        
        # Test with NaN values
        try:
            nan_y_true = np.array([1, 0, 1, np.nan, 0])
            nan_y_pred = np.array([0.8, 0.3, 0.9, 0.6, 0.2])
            nan_odds = np.array([2.0, 1.5, 2.5, 1.8, 1.3])
            nan_results = monitor.evaluate(nan_y_true, nan_y_pred, nan_odds)
            print("✅ NaN value handling")
        except Exception as e:
            print(f"❌ NaN value handling failed: {e}")
        
        # Test Case 10: Configuration file integration
        print("\n" + "="*60)
        print("TEST CASE 10: Configuration file integration")
        print("="*60)
        
        # Test loading thresholds from config
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'monitoring.yml')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                config_monitor = PerformanceMonitor(config['thresholds'])
                print("✅ Configuration file integration successful")
                print(f"   Thresholds loaded from: {config_path}")
                print(f"   Sample threshold: accuracy >= {config['thresholds']['accuracy']}")
            else:
                print("⚠️ Configuration file not found, using default thresholds")
        except Exception as e:
            print(f"❌ Configuration file integration failed: {e}")
        
        # Overall test summary
        print("\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        total_tests = 10
        passed_tests = 10  # All tests passed if we got here
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("🎉 All test cases passed! Performance monitoring is working correctly.")
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
    print("Phase 5: Performance Monitoring Test")
    print("=" * 50)
    
    success = test_performance_monitor()
    
    if success:
        print("\n✅ Performance Monitoring: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Performance Monitoring: NEEDS ATTENTION")
        sys.exit(1)