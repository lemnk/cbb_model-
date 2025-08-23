#!/usr/bin/env python3
"""
Test script for Phase 5 Performance Monitoring module.
This script tests the performance monitoring functionality with various scenarios.
"""

import sys
import os
import yaml
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_performance_monitor():
    """Test the performance monitoring module."""
    try:
        # Import the monitoring module
        from monitoring.performance_monitor import PerformanceMonitor, MetricResult
        
        print("✅ Successfully imported PerformanceMonitor and MetricResult")
        
        # Test MetricResult dataclass
        print("\nTesting MetricResult dataclass...")
        
        sample_result = MetricResult(
            value=0.65,
            threshold=0.55,
            status="PASS",
            details={"test": "data"}
        )
        
        print(f"✅ MetricResult created: value={sample_result.value}, status={sample_result.status}")
        
        # Test PerformanceMonitor initialization
        print("\nTesting PerformanceMonitor initialization...")
        
        # Load thresholds from config
        config_path = "config/monitoring.yml"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            thresholds = config['thresholds']
            print("✅ Loaded thresholds from config file")
        else:
            # Use default thresholds if config not found
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
            print("⚠️ Config file not found, using default thresholds")
        
        monitor = PerformanceMonitor(thresholds)
        print("✅ PerformanceMonitor created successfully")
        print(f"   Thresholds loaded: {len(thresholds)} metrics")
        
        # Test Case 1: High accuracy → all PASS
        print("\n" + "="*60)
        print("TEST CASE 1: High accuracy → all PASS")
        print("="*60)
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create high-quality predictions
        y_true_high = np.random.binomial(1, 0.6, n_samples)
        y_pred_proba_high = np.where(y_true_high == 1, 
                                   np.random.beta(8, 2, n_samples),  # High prob for positive
                                   np.random.beta(2, 8, n_samples))  # Low prob for negative
        odds_high = np.random.uniform(1.8, 2.5, n_samples)
        
        results_high = monitor.evaluate(y_true_high, y_pred_proba_high, odds_high)
        
        # Check results
        alert_count = sum(1 for metric in results_high.values() if metric['status'] == 'ALERT')
        warning_count = sum(1 for metric in results_high.values() if metric['status'] == 'WARNING')
        pass_count = sum(1 for metric in results_high.values() if metric['status'] == 'PASS')
        
        print(f"Results: {pass_count} PASS, {warning_count} WARNING, {alert_count} ALERT")
        
        if alert_count == 0:
            print("✅ Test Case 1 PASSED: No alerts with high accuracy")
        else:
            print(f"❌ Test Case 1 FAILED: {alert_count} alerts detected")
        
        # Test Case 2: Low accuracy → ALERT
        print("\n" + "="*60)
        print("TEST CASE 2: Low accuracy → ALERT")
        print("="*60)
        
        # Create low-quality predictions (random)
        y_true_low = np.random.binomial(1, 0.6, n_samples)
        y_pred_proba_low = np.random.uniform(0, 1, n_samples)  # Random predictions
        odds_low = np.random.uniform(1.8, 2.5, n_samples)
        
        results_low = monitor.evaluate(y_true_low, y_pred_proba_low, odds_low)
        
        # Check results
        alert_count_low = sum(1 for metric in results_low.values() if metric['status'] == 'ALERT')
        warning_count_low = sum(1 for metric in results_low.values() if metric['status'] == 'WARNING')
        
        print(f"Results: {alert_count_low} ALERT, {warning_count_low} WARNING")
        
        if alert_count_low > 0:
            print("✅ Test Case 2 PASSED: Alerts detected with low accuracy")
        else:
            print("❌ Test Case 2 FAILED: No alerts detected with low accuracy")
        
        # Test Case 3: Miscalibrated predictions → high log loss, ALERT
        print("\n" + "="*60)
        print("TEST CASE 3: Miscalibrated predictions → high log loss, ALERT")
        print("="*60)
        
        # Create miscalibrated predictions (overconfident)
        y_true_miscal = np.random.binomial(1, 0.6, n_samples)
        y_pred_proba_miscal = np.where(y_true_miscal == 1, 
                                     np.random.uniform(0.8, 1.0, n_samples),  # Overconfident positive
                                     np.random.uniform(0.0, 0.2, n_samples))  # Overconfident negative
        odds_miscal = np.random.uniform(1.8, 2.5, n_samples)
        
        results_miscal = monitor.evaluate(y_true_miscal, y_pred_proba_miscal, odds_miscal)
        
        # Check log loss specifically
        log_loss_status = results_miscal['log_loss']['status']
        log_loss_value = results_miscal['log_loss']['value']
        log_loss_threshold = results_miscal['log_loss']['threshold']
        
        print(f"Log Loss: {log_loss_value:.6f} (threshold: {log_loss_threshold:.6f}, status: {log_loss_status})")
        
        if log_loss_status == 'ALERT':
            print("✅ Test Case 3 PASSED: High log loss detected for miscalibrated predictions")
        else:
            print(f"❌ Test Case 3 FAILED: Log loss status is {log_loss_status}, expected ALERT")
        
        # Test Case 4: Negative EV given odds → ALERT
        print("\n" + "="*60)
        print("TEST CASE 4: Negative EV given odds → ALERT")
        print("="*60)
        
        # Create predictions that lead to negative expected value
        y_true_neg_ev = np.random.binomial(1, 0.6, n_samples)
        y_pred_proba_neg_ev = np.random.uniform(0.3, 0.7, n_samples)  # Moderate confidence
        odds_neg_ev = np.random.uniform(1.5, 1.8, n_samples)  # Low odds
        
        results_neg_ev = monitor.evaluate(y_true_neg_ev, y_pred_proba_neg_ev, odds_neg_ev)
        
        # Check expected value specifically
        ev_status = results_neg_ev['expected_value']['status']
        ev_value = results_neg_ev['expected_value']['value']
        ev_threshold = results_neg_ev['expected_value']['threshold']
        
        print(f"Expected Value: {ev_value:.6f} (threshold: {ev_threshold:.6f}, status: {ev_status})")
        
        if ev_status == 'ALERT':
            print("✅ Test Case 4 PASSED: Negative EV detected and alerted")
        else:
            print(f"❌ Test Case 4 FAILED: EV status is {ev_status}, expected ALERT")
        
        # Test summary generation
        print("\n" + "="*60)
        print("TESTING SUMMARY GENERATION")
        print("="*60)
        
        summary = monitor.get_summary(results_high)
        print("✅ Summary generated successfully")
        
        # Print first few lines of summary
        summary_lines = summary.split('\n')[:15]
        print("\nSummary preview:")
        for line in summary_lines:
            print(f"   {line}")
        
        # Test edge cases
        print("\n" + "="*60)
        print("TESTING EDGE CASES")
        print("="*60)
        
        # Test empty arrays
        try:
            monitor.evaluate([], [], [])
            print("❌ Empty array test failed: should have raised ValueError")
        except ValueError as e:
            print("✅ Empty array test passed: ValueError raised correctly")
        
        # Test mismatched lengths
        try:
            monitor.evaluate([1, 0], [0.8, 0.3], [2.0])  # Mismatched lengths
            print("❌ Mismatched length test failed: should have raised ValueError")
        except ValueError as e:
            print("✅ Mismatched length test passed: ValueError raised correctly")
        
        # Test invalid values
        try:
            monitor.evaluate([1, 2], [0.8, 0.3], [2.0, 2.5])  # Invalid y_true
            print("❌ Invalid values test failed: should have raised ValueError")
        except ValueError as e:
            print("✅ Invalid values test passed: ValueError raised correctly")
        
        # Overall test summary
        print("\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        total_tests = 4
        passed_tests = 0
        
        if alert_count == 0:
            passed_tests += 1
        if alert_count_low > 0:
            passed_tests += 1
        if log_loss_status == 'ALERT':
            passed_tests += 1
        if ev_status == 'ALERT':
            passed_tests += 1
        
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
        print("\n✅ Performance Monitoring Module: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Performance Monitoring Module: NEEDS ATTENTION")
        sys.exit(1)