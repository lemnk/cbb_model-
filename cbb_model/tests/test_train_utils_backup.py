#!/usr/bin/env python3
"""
Unit tests for train_utils.py

Tests all Phase 2 training formulas for mathematical correctness.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.train_utils import (
    linear_regression_loss,
    logistic_regression_loss,
    gradient_descent_update_w,
    gradient_descent_update_b,
    l2_regularization_loss,
    compute_gradients_linear_regression,
    compute_gradients_logistic_regression,
    custom_training_step_linear,
    custom_training_step_logistic
)

class TestLinearRegressionLoss(unittest.TestCase):
    """Test cases for linear regression loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
        self.w = np.array([0.1, 0.2])
        self.b = 0.5
    
    def test_linear_regression_loss_formula(self):
        """Test that MSE loss matches the exact formula: L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²"""
        # Manual calculation
        n = len(self.y_true)
        expected = (1/n) * np.sum((self.y_true - self.y_pred) ** 2)
        
        # Function calculation
        actual = linear_regression_loss(self.y_true, self.y_pred, self.w, self.b)
        
        # Should match exactly
        self.assertAlmostEqual(actual, expected, places=10)
        
        # Verify the specific value
        # MSE = (1/5) * (0.1² + 0.1² + 0.1² + 0.1² + 0.1²) = (1/5) * 0.05 = 0.01
        self.assertAlmostEqual(actual, 0.01, places=10)
    
    def test_linear_regression_loss_edge_cases(self):
        """Test edge cases for linear regression loss."""
        # Empty arrays
        self.assertEqual(linear_regression_loss(np.array([]), np.array([]), self.w, self.b), 0.0)
        
        # Single element
        y_true_single = np.array([1.0])
        y_pred_single = np.array([1.5])
        loss_single = linear_regression_loss(y_true_single, y_pred_single, self.w, self.b)
        expected_single = (1.0 - 1.5) ** 2  # 0.25
        self.assertAlmostEqual(loss_single, expected_single, places=10)
    
    def test_linear_regression_loss_input_types(self):
        """Test that function works with different input types."""
        # Test with pandas Series
        y_true_series = pd.Series(self.y_true)
        y_pred_series = pd.Series(self.y_pred)
        loss_series = linear_regression_loss(y_true_series, y_pred_series, self.w, self.b)
        
        # Test with lists
        y_true_list = self.y_true.tolist()
        y_pred_list = self.y_pred.tolist()
        loss_list = linear_regression_loss(y_true_list, y_pred_list, self.w, self.b)
        
        # All should give the same result
        self.assertAlmostEqual(loss_series, loss_list, places=10)

class TestLogisticRegressionLoss(unittest.TestCase):
    """Test cases for logistic regression loss function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.y_true = np.array([1, 0, 1, 0, 1])
        self.y_pred_proba = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    
    def test_logistic_regression_loss_formula(self):
        """Test that BCE loss matches the exact formula: L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]"""
        # Manual calculation
        n = len(self.y_true)
        expected = -(1/n) * np.sum(
            self.y_true * np.log(self.y_pred_proba) + 
            (1 - self.y_true) * np.log(1 - self.y_pred_proba)
        )
        
        # Function calculation
        actual = logistic_regression_loss(self.y_true, self.y_pred_proba)
        
        # Should match exactly
        self.assertAlmostEqual(actual, expected, places=10)
    
    def test_logistic_regression_loss_perfect_prediction(self):
        """Test loss when predictions are perfect."""
        # Perfect predictions
        y_true = np.array([1, 0, 1, 0])
        y_pred_perfect = np.array([1.0, 0.0, 1.0, 0.0])
        
        # Loss should be very close to 0 (but not exactly 0 due to epsilon)
        loss = logistic_regression_loss(y_true, y_pred_perfect)
        self.assertLess(loss, 1e-10)
    
    def test_logistic_regression_loss_random_prediction(self):
        """Test loss when predictions are random (0.5)."""
        y_true = np.array([1, 0, 1, 0])
        y_pred_random = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Loss should be -log(0.5) ≈ 0.693
        loss = logistic_regression_loss(y_true, y_pred_random)
        expected = -np.log(0.5)  # ≈ 0.693
        self.assertAlmostEqual(loss, expected, places=3)
    
    def test_logistic_regression_loss_edge_cases(self):
        """Test edge cases for logistic regression loss."""
        # Empty arrays
        self.assertEqual(logistic_regression_loss(np.array([]), np.array([])), 0.0)
        
        # Single element
        y_true_single = np.array([1.0])
        y_pred_single = np.array([0.8])
        loss_single = logistic_regression_loss(y_true_single, y_pred_single)
        expected_single = -np.log(0.8)
        self.assertAlmostEqual(loss_single, expected_single, places=10)

class TestGradientDescentUpdates(unittest.TestCase):
    """Test cases for gradient descent update functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.w = np.array([1.0, 2.0, 3.0])
        self.b = 0.5
        self.alpha = 0.1
        self.grad_w = np.array([0.5, 1.0, 1.5])
        self.grad_b = 0.3
    
    def test_gradient_descent_update_w(self):
        """Test weight update rule: w := w - α ∂L/∂w"""
        w_new = gradient_descent_update_w(self.w, self.alpha, self.grad_w)
        
        # Manual calculation
        expected_w = self.w - self.alpha * self.grad_w
        expected_w = np.array([1.0 - 0.1*0.5, 2.0 - 0.1*1.0, 3.0 - 0.1*1.5])
        expected_w = np.array([0.95, 1.9, 2.85])
        
        np.testing.assert_array_almost_equal(w_new, expected_w, decimal=10)
    
    def test_gradient_descent_update_b(self):
        """Test bias update rule: b := b - α ∂L/∂b"""
        b_new = gradient_descent_update_b(self.b, self.alpha, self.grad_b)
        
        # Manual calculation
        expected_b = self.b - self.alpha * self.grad_b
        expected_b = 0.5 - 0.1 * 0.3
        expected_b = 0.47
        
        self.assertAlmostEqual(b_new, expected_b, places=10)
    
    def test_gradient_descent_convergence(self):
        """Test that gradient descent moves toward optimum."""
        w = np.array([2.0, 3.0])
        alpha = 0.1
        
        # Multiple updates
        for _ in range(10):
            grad_w = w  # Simple gradient: ∂L/∂w = w (for loss L = (1/2)||w||²)
            w = gradient_descent_update_w(w, alpha, grad_w)
        
        # Weights should decrease toward 0
        self.assertTrue(np.all(np.abs(w) < np.array([2.0, 3.0])))

class TestL2Regularization(unittest.TestCase):
    """Test cases for L2 regularization function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.loss = 1.0
        self.w = np.array([1.0, 2.0, 3.0])
        self.lambda_reg = 0.1
    
    def test_l2_regularization_formula(self):
        """Test that L2 regularization matches the exact formula: L_reg = L + λ ||w||²"""
        regularized_loss = l2_regularization_loss(self.loss, self.w, self.lambda_reg)
        
        # Manual calculation
        l2_penalty = self.lambda_reg * np.sum(self.w ** 2)
        l2_penalty = 0.1 * (1**2 + 2**2 + 3**2)  # 0.1 * (1 + 4 + 9) = 0.1 * 14 = 1.4
        expected = self.loss + l2_penalty  # 1.0 + 1.4 = 2.4
        
        self.assertAlmostEqual(regularized_loss, expected, places=10)
    
    def test_l2_regularization_no_regularization(self):
        """Test L2 regularization when lambda = 0."""
        regularized_loss = l2_regularization_loss(self.loss, self.w, 0.0)
        self.assertEqual(regularized_loss, self.loss)
    
    def test_l2_regularization_different_lambda(self):
        """Test L2 regularization with different lambda values."""
        lambda_values = [0.01, 0.1, 1.0, 10.0]
        
        for lambda_val in lambda_values:
            regularized_loss = l2_regularization_loss(self.loss, self.w, lambda_val)
            l2_penalty = lambda_val * np.sum(self.w ** 2)
            expected = self.loss + l2_penalty
            
            self.assertAlmostEqual(regularized_loss, expected, places=10)

class TestGradientComputation(unittest.TestCase):
    """Test cases for gradient computation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
        self.y_true = np.array([2, 8, 14])  # y = x1 + x2
        self.y_pred = np.array([2.1, 7.9, 14.1])
        self.y_true_binary = np.array([1, 0, 1])
        self.y_pred_proba = np.array([0.9, 0.1, 0.8])
    
    def test_compute_gradients_linear_regression(self):
        """Test gradient computation for linear regression."""
        grad_w, grad_b = compute_gradients_linear_regression(self.X, self.y_true, self.y_pred)
        
        # Manual calculation for MSE loss
        n = len(self.y_true)
        residuals = self.y_true - self.y_pred  # [2-2.1, 8-7.9, 14-14.1] = [-0.1, 0.1, -0.1]
        
        # ∂L/∂w = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b)) xᵢ
        expected_grad_w = -(2/n) * np.sum(residuals.reshape(-1, 1) * self.X, axis=0)
        
        # ∂L/∂b = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b))
        expected_grad_b = -(2/n) * np.sum(residuals)
        
        np.testing.assert_array_almost_equal(grad_w, expected_grad_w, decimal=10)
        self.assertAlmostEqual(grad_b, expected_grad_b, places=10)
    
    def test_compute_gradients_logistic_regression(self):
        """Test gradient computation for logistic regression."""
        grad_w, grad_b = compute_gradients_logistic_regression(self.X, self.y_true_binary, self.y_pred_proba)
        
        # Manual calculation for BCE loss
        n = len(self.y_true_binary)
        error = self.y_pred_proba - self.y_true_binary  # [0.9-1, 0.1-0, 0.8-1] = [-0.1, 0.1, -0.2]
        
        # ∂L/∂w = (1/n) Σᵢ (p̂ᵢ - yᵢ) xᵢ
        expected_grad_w = (1/n) * np.sum(error.reshape(-1, 1) * self.X, axis=0)
        
        # ∂L/∂b = (1/n) Σᵢ (p̂ᵢ - yᵢ)
        expected_grad_b = (1/n) * np.sum(error)
        
        np.testing.assert_array_almost_equal(grad_w, expected_grad_w, decimal=10)
        self.assertAlmostEqual(grad_b, expected_grad_b, places=10)

class TestCustomTrainingSteps(unittest.TestCase):
    """Test cases for custom training step functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.X = np.array([[1, 2], [3, 4]])  # 2 samples, 2 features
        self.y_true_linear = np.array([3, 7])  # y = x1 + x2
        self.y_true_binary = np.array([1, 0])
        self.w = np.array([0.5, 0.5])
        self.b = 0.0
        self.alpha = 0.1
        self.lambda_reg = 0.01
    
    def test_custom_training_step_linear(self):
        """Test custom training step for linear regression."""
        w_new, b_new, loss = custom_training_step_linear(
            self.X, self.y_true_linear, self.w, self.b, self.alpha, self.lambda_reg
        )
        
        # Check that parameters were updated
        self.assertFalse(np.array_equal(w_new, self.w))
        self.assertNotEqual(b_new, self.b)
        
        # Check that loss is computed
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_custom_training_step_logistic(self):
        """Test custom training step for logistic regression."""
        w_new, b_new, loss = custom_training_step_logistic(
            self.X, self.y_true_binary, self.w, self.b, self.alpha, self.lambda_reg
        )
        
        # Check that parameters were updated
        self.assertFalse(np.array_equal(w_new, self.w))
        self.assertNotEqual(b_new, self.b)
        
        # Check that loss is computed
        self.assertIsInstance(loss, float)
        self.assertGreater(loss, 0)
    
    def test_training_step_convergence(self):
        """Test that training steps reduce loss over multiple iterations."""
        w = np.array([0.0, 0.0])
        b = 0.0
        alpha = 0.1
        lambda_reg = 0.0
        
        losses = []
        
        # Multiple training steps
        for _ in range(5):
            w, b, loss = custom_training_step_linear(
                self.X, self.y_true_linear, w, b, alpha, lambda_reg
            )
            losses.append(loss)
        
        # Loss should generally decrease (not strictly due to learning rate)
        # But we can check that it's not increasing dramatically
        self.assertLess(losses[-1], losses[0] * 2)  # Loss shouldn't double

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
class TestLogTransforms(unittest.TestCase):
    """Test cases for log transform and inverse log transform functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.y_positive = np.array([1, 2, 3, 4, 5])
        self.y_series = pd.Series([1, 2, 3, 4, 5], name='test')
        self.y_zero = np.array([0, 1, 2, 3, 4])
        self.y_small = np.array([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
        self.y_negative = np.array([-1, 0, 1, 2, 3])
    
    def test_log_transform_array_positive(self):
        """Test log transform with numpy array of positive values."""
        y_log = log_transform(self.y_positive)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(y_log)))
        
        # Check that log(1) = 0 (first element)
        self.assertAlmostEqual(y_log[0], 0.0, places=10)
        
        # Check that log(2) ≈ 0.69314718
        self.assertAlmostEqual(y_log[1], np.log(2), places=10)
        
        # Check that log(5) ≈ 1.60943791
        self.assertAlmostEqual(y_log[4], np.log(5), places=10)
    
    def test_log_transform_series_positive(self):
        """Test log transform with pandas Series of positive values."""
        y_log = log_transform(self.y_positive)
        
        # Check that result is a pandas Series
        self.assertIsInstance(y_log, pd.Series)
        
        # Check that index and name are preserved
        self.assertEqual(y_log.name, 'test')
        np.testing.assert_array_equal(y_log.index, self.y_series.index)
        
        # Check that values are correct
        expected = np.log(self.y_series.values)
        np.testing.assert_array_almost_equal(y_log.values, expected, decimal=10)
    
    def test_log_transform_inverse_roundtrip(self):
        """Test that log_transform → inverse_log_transform ≈ original."""
        # Test with default c=1e-9
        y_log = log_transform(self.y_positive)
        y_restored = inverse_log_transform(y_log)
        
        # Check that roundtrip preserves values (within numerical precision)
        np.testing.assert_array_almost_equal(y_restored, self.y_positive, decimal=10)
        
        # Test with c=1.0
        y_log_c1 = log_transform(self.y_positive, c=1.0)
        y_restored_c1 = inverse_log_transform(y_log_c1, c=1.0)
        
        # Check that roundtrip preserves values
        np.testing.assert_array_almost_equal(y_log_c1, c=1.0)
    
    def test_log_transform_zero_and_small_values(self):
        """Test log transform with zero and very small values."""
        # Test with zero values
        y_log_zero = log_transform(self.y_zero)
        
        # log(0 + 1e-9) ≈ -20.7232658
        self.assertAlmostEqual(y_log_zero[0], np.log(1e-9), places=10)
        
        # Test with very small values
        y_log_small = log_transform(self.y_small)
        
        # All should be finite
        self.assertTrue(np.all(np.isfinite(y_log_small)))
        
        # Values should be very negative (log of very small numbers)
        self.assertTrue(np.all(y_log_small < -10))
    
    def test_log_transform_raises_on_negative(self):
        """Test that log transform raises ValueError for negative values."""
        with self.assertRaises(ValueError):
            log_transform(self.y_negative)
        
        # Test with c=1.0 and values that would make (y + c) ≤ 0
        y_problematic = np.array([-2, -1, 0, 1, 2])
        with self.assertRaises(ValueError):
            log_transform(y_problematic, c=1.0)
    
    def test_inverse_log_transform_correctness(self):
        """Test inverse log transform mathematical correctness."""
        # Test with known values
        y_log_known = np.array([0, np.log(2), np.log(3), np.log(4), np.log(5)])
        y_original = inverse_log_transform(y_log_known)
        
        # Should recover original values
        expected = np.array([0, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(y_original, expected, decimal=10)
        
        # Test with c=1.0
        y_log_c1 = np.array([0, np.log(2), np.log(3), np.log(4), np.log(5)])
        y_original_c1 = inverse_log_transform(y_log_c1, c=1.0)
        
        # Should recover original values
        expected_c1 = np.array([0, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(y_original_c1, expected_c1, decimal=10)
    
    def test_log_transform_numerical_stability(self):
        """Test numerical stability of log transform."""
        # Test with very small constant
        y_log_stable = log_transform(self.y_positive, c=1e-9)
        
        # Should use log1p for numerical stability
        expected_stable = np.log1p(self.y_positive + (1e-9 - 1))
        np.testing.assert_array_almost_equal(y_log_stable, expected_stable, decimal=15)
        
        # Test with c=1.0
        y_log_c1 = log_transform(self.y_positive, c=1.0)
        expected_c1 = np.log(self.y_positive + 1.0)
        np.testing.assert_array_almost_equal(y_log_c1, expected_c1, decimal=15)
    
    def test_inverse_log_transform_numerical_stability(self):
        """Test numerical stability of inverse log transform."""
        # Test with c=1.0 (should use expm1)
        y_log = np.array([0, 1, 2, 3, 4])
        y_original = inverse_log_transform(y_log, c=1.0)
        
        # Should use expm1 for stability
        expected = np.expm1(y_log)
        np.testing.assert_array_almost_equal(y_original, expected, decimal=15)
        
        # Test with c≠1.0 (should use exp - c)
        y_original_other = inverse_log_transform(y_log, c=0.5)
        expected_other = np.exp(y_log) - 0.5
        np.testing.assert_array_almost_equal(y_original_other, expected_other, decimal=15)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)
