#!/usr/bin/env python3
"""
Unit and Integration Testing Script for Phase 4 & 5 Audit
Tests all components with adversarial test cases and edge conditions.
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from typing import Dict, List, Tuple
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_phase4_components():
    """Test Phase 4 components with adversarial cases."""
    print("="*60)
    print("PHASE 4 COMPONENT TESTING")
    print("="*60)
    
    results = []
    
    # Test 1: Hyperparameter Optimizer
    print("\n1. HYPERPARAMETER OPTIMIZER")
    try:
        from optimization.hyperparameter_optimizer import GridSearchOptimizer
        
        # Test with empty dataset
        try:
            optimizer = GridSearchOptimizer(None, {}, cv=5)
            optimizer.optimize(np.array([]), np.array([]))
            print("❌ Should have failed with empty dataset")
            results.append(("HPO Empty Dataset", False))
        except Exception as e:
            print(f"✅ Correctly handled empty dataset: {e}")
            results.append(("HPO Empty Dataset", True))
        
        # Test with single sample
        try:
            optimizer = GridSearchOptimizer(None, {}, cv=5)
            optimizer.optimize(np.array([[0.1, 0.2]]), np.array([1]))
            print("❌ Should have failed with single sample")
            results.append(("HPO Single Sample", False))
        except Exception as e:
            print(f"✅ Correctly handled single sample: {e}")
            results.append(("HPO Single Sample", True))
            
    except Exception as e:
        print(f"❌ Hyperparameter optimizer test failed: {e}")
        results.append(("HPO Tests", False))
    
    # Test 2: Ensemble Methods
    print("\n2. ENSEMBLE METHODS")
    try:
        from ensemble.ensemble_methods import averaging_ensemble, weighted_ensemble
        
        # Test with empty predictions list
        try:
            result = averaging_ensemble([])
            print("❌ Should have failed with empty predictions")
            results.append(("Ensemble Empty List", False))
        except Exception as e:
            print(f"✅ Correctly handled empty predictions: {e}")
            results.append(("Ensemble Empty List", True))
        
        # Test with mismatched array lengths
        try:
            preds = [np.array([0.1, 0.2]), np.array([0.3, 0.4, 0.5])]
            result = averaging_ensemble(preds)
            print("❌ Should have failed with mismatched lengths")
            results.append(("Ensemble Mismatched Lengths", False))
        except Exception as e:
            print(f"✅ Correctly handled mismatched lengths: {e}")
            results.append(("Ensemble Mismatched Lengths", True))
        
        # Test with invalid weights (sum != 1)
        try:
            preds = [np.array([0.1, 0.2]), np.array([0.3, 0.4])]
            weights = [0.6, 0.5]  # Sum > 1
            result = weighted_ensemble(preds, weights)
            print("✅ Weighted ensemble handled invalid weights")
            results.append(("Ensemble Invalid Weights", True))
        except Exception as e:
            print(f"❌ Failed to handle invalid weights: {e}")
            results.append(("Ensemble Invalid Weights", False))
            
    except Exception as e:
        print(f"❌ Ensemble methods test failed: {e}")
        results.append(("Ensemble Tests", False))
    
    # Test 3: Calibration Methods
    print("\n3. CALIBRATION METHODS")
    try:
        from calibration.calibration_methods import Calibrator
        
        # Test with extreme probabilities
        try:
            calibrator = Calibrator(method='platt')
            extreme_probs = np.array([0.0, 1.0, 0.0001, 0.9999])
            y_true = np.array([0, 1, 0, 1])
            result = calibrator.fit_transform(extreme_probs, y_true)
            print("✅ Handled extreme probabilities")
            results.append(("Calibration Extreme Probs", True))
        except Exception as e:
            print(f"❌ Failed to handle extreme probabilities: {e}")
            results.append(("Calibration Extreme Probs", False))
        
        # Test with NaN values
        try:
            calibrator = Calibrator(method='platt')
            nan_probs = np.array([0.5, np.nan, 0.7, 0.3])
            y_true = np.array([1, 0, 1, 0])
            result = calibrator.fit_transform(nan_probs, y_true)
            print("❌ Should have failed with NaN values")
            results.append(("Calibration NaN Values", False))
        except Exception as e:
            print(f"✅ Correctly handled NaN values: {e}")
            results.append(("Calibration NaN Values", True))
            
    except Exception as e:
        print(f"❌ Calibration methods test failed: {e}")
        results.append(("Calibration Tests", False))
    
    return results

def test_phase5_components():
    """Test Phase 5 components with adversarial cases."""
    print("\n" + "="*60)
    print("PHASE 5 COMPONENT TESTING")
    print("="*60)
    
    results = []
    
    # Test 1: Schema Validation
    print("\n1. SCHEMA VALIDATION")
    try:
        from monitoring.schema_validation import SchemaValidator
        
        validator = SchemaValidator()
        
        # Test with missing columns
        try:
            invalid_data = {
                'game_id': ['1', '2'],
                'date': ['2024-01-01', '2024-01-02'],
                # Missing required columns
            }
            invalid_df = pd.DataFrame(invalid_data)
            result = validator.comprehensive_validation(invalid_df)
            
            if not result['is_valid']:
                print("✅ Correctly detected missing columns")
                results.append(("Schema Missing Columns", True))
            else:
                print("❌ Failed to detect missing columns")
                results.append(("Schema Missing Columns", False))
        except Exception as e:
            print(f"✅ Correctly handled missing columns: {e}")
            results.append(("Schema Missing Columns", True))
        
        # Test with wrong data types
        try:
            wrong_type_data = {
                'game_id': ['1', '2'],
                'date': ['2024-01-01', '2024-01-02'],
                'season': [2024, 2024],
                'home_team': ['Duke', 'UNC'],
                'away_team': ['UNC', 'Duke'],
                'team_efficiency': ['invalid', 'invalid'],  # Should be float
                'player_availability': [0.8, 0.9],
                'dynamic_factors': [0.7, 0.8],
                'market_signals': [0.6, 0.7],
                'target': [1, 0]
            }
            wrong_type_df = pd.DataFrame(wrong_type_data)
            result = validator.comprehensive_validation(wrong_type_df)
            
            if not result['is_valid']:
                print("✅ Correctly detected wrong data types")
                results.append(("Schema Wrong Types", True))
            else:
                print("❌ Failed to detect wrong data types")
                results.append(("Schema Wrong Types", False))
        except Exception as e:
            print(f"✅ Correctly handled wrong data types: {e}")
            results.append(("Schema Wrong Types", True))
            
    except Exception as e:
        print(f"❌ Schema validation test failed: {e}")
        results.append(("Schema Tests", False))
    
    # Test 2: Drift Detection
    print("\n2. DRIFT DETECTION")
    try:
        from monitoring.drift_detection import DriftDetector
        
        # Create test data
        np.random.seed(42)
        reference = np.random.normal(0, 1, 100)
        current = np.random.normal(0, 1, 100)  # No drift
        
        detector = DriftDetector(
            reference_df=pd.DataFrame({'feature': reference}),
            psi_threshold=0.25,
            ks_threshold=0.1,
            kl_threshold=0.1
        )
        
        # Test with injected drift
        try:
            # Inject significant drift
            current_drifted = np.random.normal(2.0, 1, 100)  # Large shift
            
            psi_score = detector.compute_psi(reference, current_drifted)
            print(f"PSI score with drift: {psi_score:.6f}")
            
            if psi_score > 0.25:
                print("✅ Correctly detected injected drift")
                results.append(("Drift Detection Injected", True))
            else:
                print("❌ Failed to detect injected drift")
                results.append(("Drift Detection Injected", False))
        except Exception as e:
            print(f"❌ Drift detection test failed: {e}")
            results.append(("Drift Detection Injected", False))
            
    except Exception as e:
        print(f"❌ Drift detection test failed: {e}")
        results.append(("Drift Tests", False))
    
    # Test 3: Performance Monitoring
    print("\n3. PERFORMANCE MONITORING")
    try:
        from monitoring.performance_monitor import PerformanceMonitor
        
        thresholds = {
            'accuracy': 0.8,
            'log_loss': 0.5,
            'brier_score': 0.3,
            'precision': 0.7,
            'recall': 0.7,
            'f1': 0.7,
            'roc_auc': 0.6,
            'expected_value': 0.0
        }
        
        monitor = PerformanceMonitor(thresholds)
        
        # Test with below-threshold performance
        try:
            # Create poor performance data
            y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
            y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8, 0.1, 0.9])  # Poor predictions
            odds = np.array([2.0, 1.5, 2.5, 1.8, 1.3, 1.2, 2.2, 1.6, 2.1, 1.4])
            
            results_dict = monitor.evaluate(y_true, y_pred_proba, odds)
            
            # Check if ALERT status is triggered
            alert_count = sum(1 for result in results_dict.values() if result['status'] == 'ALERT')
            print(f"Alerts triggered: {alert_count}")
            
            if alert_count > 0:
                print("✅ Correctly triggered alerts for poor performance")
                results.append(("Performance Monitoring Alerts", True))
            else:
                print("❌ Failed to trigger alerts for poor performance")
                results.append(("Performance Monitoring Alerts", False))
        except Exception as e:
            print(f"❌ Performance monitoring test failed: {e}")
            results.append(("Performance Monitoring Alerts", False))
            
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        results.append(("Performance Tests", False))
    
    # Test 4: Alerts System
    print("\n4. ALERTS SYSTEM")
    try:
        from monitoring.alerts import AlertManager
        
        # Test configuration
        alert_config = {
            'mode': 'console',
            'slack_webhook': 'https://hooks.slack.com/services/XXXX/XXXX/XXXX',
            'file_path': 'logs/alerts.log'
        }
        
        manager = AlertManager(alert_config)
        
        # Test with ALERT status results
        try:
            alert_results = {
                "accuracy": {"value": 0.52, "threshold": 0.55, "status": "ALERT"},
                "expected_value": {"value": -0.03, "threshold": 0.0, "status": "ALERT"}
            }
            
            alert_messages = manager.check_alerts(alert_results)
            print(f"Alert messages generated: {len(alert_messages)}")
            
            if len(alert_messages) == 2:
                print("✅ Correctly generated alerts for ALERT status")
                results.append(("Alerts ALERT Status", True))
            else:
                print("❌ Failed to generate correct number of alerts")
                results.append(("Alerts ALERT Status", False))
        except Exception as e:
            print(f"❌ Alerts test failed: {e}")
            results.append(("Alerts Tests", False))
            
    except Exception as e:
        print(f"❌ Alerts system test failed: {e}")
        results.append(("Alerts Tests", False))
    
    return results

def test_deployment_hardening():
    """Test deployment components with malformed requests."""
    print("\n" + "="*60)
    print("DEPLOYMENT HARDENING TESTING")
    print("="*60)
    
    results = []
    
    # Test 1: FastAPI Endpoints
    print("\n1. FASTAPI ENDPOINTS")
    try:
        from deployment.api import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test malformed prediction request
        try:
            malformed_request = {
                "features": "invalid_features"  # Should be dict
            }
            response = client.post("/predict", json=malformed_request)
            
            if response.status_code == 422:  # Validation error
                print("✅ Correctly rejected malformed prediction request")
                results.append(("FastAPI Malformed Request", True))
            else:
                print(f"❌ Failed to reject malformed request: {response.status_code}")
                results.append(("FastAPI Malformed Request", False))
        except Exception as e:
            print(f"✅ Correctly handled malformed request: {e}")
            results.append(("FastAPI Malformed Request", True))
        
        # Test health endpoint
        try:
            response = client.get("/health")
            if response.status_code == 200:
                print("✅ Health endpoint working correctly")
                results.append(("FastAPI Health Endpoint", True))
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                results.append(("FastAPI Health Endpoint", False))
        except Exception as e:
            print(f"❌ Health endpoint test failed: {e}")
            results.append(("FastAPI Health Endpoint", False))
            
    except Exception as e:
        print(f"❌ FastAPI testing failed: {e}")
        results.append(("FastAPI Tests", False))
    
    # Test 2: CLI Input Validation
    print("\n2. CLI INPUT VALIDATION")
    try:
        from deployment.cli import validate_input
        
        # Test with invalid input
        try:
            invalid_input = "invalid_data"
            result = validate_input(invalid_input)
            print("❌ Should have failed with invalid input")
            results.append(("CLI Invalid Input", False))
        except Exception as e:
            print(f"✅ Correctly handled invalid input: {e}")
            results.append(("CLI Invalid Input", True))
            
    except Exception as e:
        print(f"❌ CLI testing failed: {e}")
        results.append(("CLI Tests", False))
    
    return results

def test_monitoring_validation():
    """Test monitoring system validation."""
    print("\n" + "="*60)
    print("MONITORING SYSTEM VALIDATION")
    print("="*60)
    
    results = []
    
    # Test 1: End-to-End Monitoring Pipeline
    print("\n1. END-TO-END MONITORING PIPELINE")
    try:
        from monitoring.schema_validation import SchemaValidator
        from monitoring.drift_detection import DriftDetector
        from monitoring.performance_monitor import PerformanceMonitor
        from monitoring.alerts import AlertManager
        
        # Create test data
        test_data = {
            'game_id': ['1', '2', '3'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'season': [2024, 2024, 2024],
            'home_team': ['Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'],
            'team_efficiency': [0.75, 0.68, 0.82],
            'player_availability': [0.85, 0.92, 0.78],
            'dynamic_factors': [0.68, 0.71, 0.74],
            'market_signals': [0.72, 0.65, 0.81],
            'target': [1, 0, 1]
        }
        
        test_df = pd.DataFrame(test_data)
        
        # Step 1: Schema validation
        validator = SchemaValidator()
        schema_results = validator.comprehensive_validation(test_df)
        
        if schema_results['is_valid']:
            print("✅ Schema validation passed")
            results.append(("Schema Validation", True))
        else:
            print("❌ Schema validation failed")
            results.append(("Schema Validation", False))
        
        # Step 2: Drift detection
        detector = DriftDetector(
            reference_df=test_df,
            psi_threshold=0.25,
            ks_threshold=0.1,
            kl_threshold=0.1
        )
        
        drift_results = detector.detect_drift(test_df)
        
        if 'overall_drift_detected' in drift_results:
            print("✅ Drift detection completed")
            results.append(("Drift Detection", True))
        else:
            print("❌ Drift detection failed")
            results.append(("Drift Detection", False))
        
        # Step 3: Performance monitoring
        thresholds = {
            'accuracy': 0.5,
            'log_loss': 1.0,
            'brier_score': 0.5,
            'precision': 0.5,
            'recall': 0.5,
            'f1': 0.5,
            'roc_auc': 0.5,
            'expected_value': 0.0
        }
        
        monitor = PerformanceMonitor(thresholds)
        
        # Create test predictions
        y_true = np.array([1, 0, 1])
        y_pred_proba = np.array([0.8, 0.2, 0.9])
        odds = np.array([2.0, 1.5, 2.5])
        
        perf_results = monitor.evaluate(y_true, y_pred_proba, odds)
        
        if len(perf_results) == 8:  # All 8 metrics
            print("✅ Performance monitoring completed")
            results.append(("Performance Monitoring", True))
        else:
            print("❌ Performance monitoring failed")
            results.append(("Performance Monitoring", False))
        
        # Step 4: Alerts
        alert_config = {'mode': 'console'}
        alert_manager = AlertManager(alert_config)
        
        alert_messages = alert_manager.check_alerts(perf_results)
        
        if isinstance(alert_messages, list):
            print("✅ Alerts system completed")
            results.append(("Alerts System", True))
        else:
            print("❌ Alerts system failed")
            results.append(("Alerts System", False))
            
    except Exception as e:
        print(f"❌ Monitoring pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Monitoring Pipeline", False))
    
    return results

def run_comprehensive_testing():
    """Run all comprehensive tests."""
    print("COMPREHENSIVE UNIT & INTEGRATION TESTING AUDIT")
    print("="*60)
    print("Testing all components with adversarial cases and edge conditions...")
    
    all_results = []
    
    # Phase 4 Testing
    phase4_results = test_phase4_components()
    all_results.extend(phase4_results)
    
    # Phase 5 Testing
    phase5_results = test_phase5_components()
    all_results.extend(phase5_results)
    
    # Deployment Hardening Testing
    deployment_results = test_deployment_hardening()
    all_results.extend(deployment_results)
    
    # Monitoring Validation Testing
    monitoring_results = test_monitoring_validation()
    all_results.extend(monitoring_results)
    
    # Summary
    print("\n" + "="*60)
    print("COMPREHENSIVE TESTING SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(all_results)
    
    for test_name, result in all_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All comprehensive tests passed successfully!")
        return True
    else:
        print("⚠️ Some tests failed and need attention.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_testing()
    sys.exit(0 if success else 1)