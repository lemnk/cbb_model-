"""
Unit tests for Phase 4: Model Optimization & Deployment.
Tests all components including metrics, ensembles, calibration, validation, and optimization.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import warnings
warnings.filterwarnings('ignore')

# Import Phase 4 components
from src.metrics import (
    roc_auc_score, log_loss, expected_calibration_error, 
    roi, brier_score
)
from src.ensemble import (
    averaging_ensemble, weighted_ensemble, stacked_ensemble,
    EnsembleModel
)
from src.calibration import (
    platt_scaling, isotonic_calibration, Calibrator,
    evaluate_calibration
)
from src.validation import WalkForwardValidator, walk_forward_split
from src.optimization import (
    GridSearchOptimizer, RandomSearchOptimizer, 
    BayesianOptimizer, create_optimizer
)


class TestPhase4Metrics(unittest.TestCase):
    """Test cases for Phase 4 metrics."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.binomial(1, 0.6, self.n_samples)
        self.y_pred_proba = np.random.beta(2, 3, self.n_samples)
        
    def test_roc_auc_score(self):
        """Test ROC-AUC score calculation."""
        auc = roc_auc_score(self.y_true, self.y_pred_proba)
        self.assertIsInstance(auc, float)
        self.assertTrue(0 <= auc <= 1)
        
    def test_log_loss(self):
        """Test log loss calculation."""
        loss = log_loss(self.y_true, self.y_pred_proba)
        self.assertIsInstance(loss, float)
        self.assertTrue(loss >= 0)
        
    def test_expected_calibration_error(self):
        """Test expected calibration error."""
        ece = expected_calibration_error(self.y_pred_proba, self.y_true)
        self.assertIsInstance(ece, float)
        self.assertTrue(0 <= ece <= 1)
        
    def test_roi(self):
        """Test ROI calculation."""
        roi_value = roi(1200, 1000)
        self.assertEqual(roi_value, 0.2)
        
        roi_value = roi(800, 1000)
        self.assertEqual(roi_value, -0.2)
        
    def test_brier_score(self):
        """Test Brier score calculation."""
        bs = brier_score(self.y_true, self.y_pred_proba)
        self.assertIsInstance(bs, float)
        self.assertTrue(0 <= bs <= 1)


class TestPhase4Ensembles(unittest.TestCase):
    """Test cases for Phase 4 ensemble methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.n_models = 3
        
        # Create sample predictions
        self.predictions = [
            np.random.rand(self.n_samples) for _ in range(self.n_models)
        ]
        self.weights = [0.3, 0.4, 0.3]
        
    def test_averaging_ensemble(self):
        """Test averaging ensemble."""
        result = averaging_ensemble(self.predictions)
        self.assertEqual(len(result), self.n_samples)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
    def test_weighted_ensemble(self):
        """Test weighted ensemble."""
        result = weighted_ensemble(self.predictions, self.weights)
        self.assertEqual(len(result), self.n_samples)
        self.assertTrue(np.all(result >= 0))
        self.assertTrue(np.all(result <= 1))
        
    def test_weighted_ensemble_normalization(self):
        """Test that weights are normalized."""
        unnormalized_weights = [1, 2, 3]  # Sum = 6
        result = weighted_ensemble(self.predictions, unnormalized_weights)
        self.assertEqual(len(result), self.n_samples)
        
    def test_stacked_ensemble(self):
        """Test stacked ensemble."""
        y_true = np.random.binomial(1, 0.6, self.n_samples)
        base_preds = np.column_stack(self.predictions)
        
        meta_learner = stacked_ensemble(base_preds, y_true)
        self.assertIsNotNone(meta_learner)
        self.assertTrue(hasattr(meta_learner, 'predict_proba'))


class TestPhase4Calibration(unittest.TestCase):
    """Test cases for Phase 4 calibration methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.y_true = np.random.binomial(1, 0.6, self.n_samples)
        self.y_pred_proba = np.random.beta(2, 3, self.n_samples)
        
    def test_platt_scaling(self):
        """Test Platt scaling calibration."""
        calibrated = platt_scaling(self.y_pred_proba, self.y_true)
        self.assertEqual(len(calibrated), self.n_samples)
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
        
    def test_isotonic_calibration(self):
        """Test isotonic calibration."""
        calibrated = isotonic_calibration(self.y_pred_proba, self.y_true)
        self.assertEqual(len(calibrated), self.n_samples)
        self.assertTrue(np.all(calibrated >= 0))
        self.assertTrue(np.all(calibrated <= 1))
        
    def test_calibrator_class(self):
        """Test Calibrator class."""
        calibrator = Calibrator(method='platt')
        self.assertEqual(calibrator.method, 'platt')
        self.assertFalse(calibrator.is_fitted)
        
        # Test fit_transform
        calibrated = calibrator.fit_transform(self.y_pred_proba, self.y_true)
        self.assertEqual(len(calibrated), self.n_samples)
        self.assertTrue(calibrator.is_fitted)
        
    def test_evaluate_calibration(self):
        """Test calibration evaluation."""
        calibrated = platt_scaling(self.y_pred_proba, self.y_true)
        metrics = evaluate_calibration(self.y_true, self.y_pred_proba, calibrated)
        
        self.assertIn('ece_raw', metrics)
        self.assertIn('ece_calibrated', metrics)
        self.assertIn('brier_raw', metrics)
        self.assertIn('brier_calibrated', metrics)


class TestPhase4Validation(unittest.TestCase):
    """Test cases for Phase 4 validation methods."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.train_size = 30
        self.step_size = 10
        
        # Create sample data
        self.data = list(range(self.n_samples))
        
    def test_walk_forward_split(self):
        """Test walk-forward split generator."""
        splits = list(walk_forward_split(self.data, self.train_size, self.step_size))
        self.assertGreater(len(splits), 0)
        
        for train_data, val_data in splits:
            self.assertEqual(len(train_data), self.train_size)
            self.assertLessEqual(len(val_data), self.step_size)
            
    def test_walk_forward_validator(self):
        """Test WalkForwardValidator class."""
        validator = WalkForwardValidator(
            train_size=self.train_size,
            step_size=self.step_size
        )
        
        self.assertEqual(validator.train_size, self.train_size)
        self.assertEqual(validator.step_size, self.step_size)
        
    def test_walk_forward_validator_validation(self):
        """Test walk-forward validation process."""
        # Create sample DataFrame
        dates = pd.date_range('2020-01-01', periods=self.n_samples, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'feature_1': np.random.randn(self.n_samples),
            'feature_2': np.random.randn(self.n_samples),
            'target': np.random.binomial(1, 0.6, self.n_samples)
        })
        
        validator = WalkForwardValidator(
            train_size=self.train_size,
            step_size=self.step_size
        )
        
        def model_factory():
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(random_state=42)
        
        # Run validation
        results = validator.validate(
            df, model_factory,
            feature_cols=['feature_1', 'feature_2'],
            target_col='target',
            date_col='date'
        )
        
        self.assertIsInstance(results, dict)
        self.assertIn('n_splits', results)


class TestPhase4Optimization(unittest.TestCase):
    """Test cases for Phase 4 hyperparameter optimization."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 50
        self.X = np.random.randn(self.n_samples, 5)
        self.y = np.random.binomial(1, 0.6, self.n_samples)
        
        # Define parameter spaces
        self.param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5]
        }
        
        self.param_distributions = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5, 7]
        }
        
    def test_grid_search_optimizer(self):
        """Test GridSearchOptimizer."""
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42)
        
        optimizer = GridSearchOptimizer(
            base_model, self.param_grid, cv=2, random_state=42
        )
        
        self.assertIsInstance(optimizer, GridSearchOptimizer)
        self.assertEqual(optimizer.param_grid, self.param_grid)
        
    def test_random_search_optimizer(self):
        """Test RandomSearchOptimizer."""
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42)
        
        optimizer = RandomSearchOptimizer(
            base_model, self.param_distributions, n_iter=5, cv=2, random_state=42
        )
        
        self.assertIsInstance(optimizer, RandomSearchOptimizer)
        self.assertEqual(optimizer.param_distributions, self.param_distributions)
        
    def test_create_optimizer_factory(self):
        """Test optimizer factory function."""
        from sklearn.ensemble import RandomForestClassifier
        base_model = RandomForestClassifier(random_state=42)
        
        # Test grid search
        grid_opt = create_optimizer('grid', base_model, self.param_grid)
        self.assertIsInstance(grid_opt, GridSearchOptimizer)
        
        # Test random search
        random_opt = create_optimizer('random', base_model, self.param_distributions)
        self.assertIsInstance(random_opt, RandomSearchOptimizer)
        
        # Test invalid type
        with self.assertRaises(ValueError):
            create_optimizer('invalid', base_model, self.param_grid)


class TestPhase4Integration(unittest.TestCase):
    """Integration tests for Phase 4 components."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 100
        self.X = np.random.randn(self.n_samples, 10)
        self.y = np.random.binomial(1, 0.6, self.n_samples)
        
    def test_end_to_end_workflow(self):
        """Test end-to-end Phase 4 workflow."""
        # 1. Calculate metrics
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42)
        model.fit(self.X, self.y)
        y_pred_proba = model.predict_proba(self.X)[:, 1]
        
        auc = roc_auc_score(self.y, y_pred_proba)
        self.assertTrue(0 <= auc <= 1)
        
        # 2. Create ensemble
        predictions = [y_pred_proba, y_pred_proba * 0.9, y_pred_proba * 1.1]
        ensemble_pred = averaging_ensemble(predictions)
        self.assertEqual(len(ensemble_pred), self.n_samples)
        
        # 3. Calibrate probabilities
        calibrated = platt_scaling(y_pred_proba, self.y)
        self.assertEqual(len(calibrated), self.n_samples)
        
        # 4. Evaluate calibration
        metrics = evaluate_calibration(self.y, y_pred_proba, calibrated)
        self.assertIn('ece_improvement', metrics)
        self.assertIn('brier_improvement', metrics)
        
        print("✅ End-to-end Phase 4 workflow test passed!")


def run_phase4_tests():
    """Run all Phase 4 tests."""
    print("🧪 Running Phase 4 Unit Tests...")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPhase4Metrics,
        TestPhase4Ensembles,
        TestPhase4Calibration,
        TestPhase4Validation,
        TestPhase4Optimization,
        TestPhase4Integration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Phase 4 Test Results Summary")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    if result.wasSuccessful():
        print("\n🎉 All Phase 4 tests passed successfully!")
        return True
    else:
        print("\n💥 Some Phase 4 tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = run_phase4_tests()
    exit(0 if success else 1)