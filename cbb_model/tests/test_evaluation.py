#!/usr/bin/env python3
"""
Unit tests for evaluation metrics in Phase 3.

Tests all exact formulas from requirements:
- Classification metrics: Accuracy, Precision, Recall, F1-Score
- Regression metrics: RMSE, MAE, R²
"""

import sys
import os
import unittest
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import ModelEvaluator

class TestEvaluationMetrics(unittest.TestCase):
    """
    Test cases for evaluation metrics using exact formulas.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = ModelEvaluator(random_state=42)
        
        # Test data for classification
        self.tp, self.tn, self.fp, self.fn = 80, 70, 20, 30
        
        # Test data for regression
        self.y_true = np.array([1, 2, 3, 4, 5])
        self.y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    def test_accuracy_formula(self):
        """Test accuracy formula: (tp + tn) / (tp + tn + fp + fn)"""
        expected = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)
        actual = self.evaluator.accuracy(self.tp, self.tn, self.fp, self.fn)
        
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(actual, 0.75, places=2)  # (80+70)/(80+70+20+30) = 150/200 = 0.75
    
    def test_precision_formula(self):
        """Test precision formula: tp / (tp + fp)"""
        expected = self.tp / (self.tp + self.fp)
        actual = self.evaluator.precision(self.tp, self.fp)
        
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(actual, 0.8, places=2)  # 80/(80+20) = 80/100 = 0.8
    
    def test_recall_formula(self):
        """Test recall formula: tp / (tp + fn)"""
        expected = self.tp / (self.tp + self.fn)
        actual = self.evaluator.recall(self.tp, self.fn)
        
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(actual, 0.727, places=3)  # 80/(80+30) = 80/110 ≈ 0.727
    
    def test_f1_score_formula(self):
        """Test F1-score formula: 2 * (precision * recall) / (precision + recall)"""
        precision = self.evaluator.precision(self.tp, self.fp)
        recall = self.evaluator.recall(self.tp, self.fn)
        expected = 2 * (precision * recall) / (precision + recall)
        actual = self.evaluator.f1_score(self.tp, self.fp, self.fn)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # F1 = 2 * (0.8 * 0.727) / (0.8 + 0.727) ≈ 0.762
    
    def test_rmse_formula(self):
        """Test RMSE formula: sqrt(mean((y_true - y_pred)²))"""
        expected = np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))
        actual = self.evaluator.rmse(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # RMSE = sqrt(mean([0.1², 0.1², 0.1², 0.1², 0.1²])) = sqrt(0.01) = 0.1
    
    def test_mae_formula(self):
        """Test MAE formula: mean(|y_true - y_pred|)"""
        expected = np.mean(np.abs(self.y_true - self.y_pred))
        actual = self.evaluator.mae(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # MAE = mean([0.1, 0.1, 0.1, 0.1, 0.1]) = 0.1
    
    def test_r2_formula(self):
        """Test R² formula: 1 - (ss_res / ss_tot)"""
        ss_res = np.sum((self.y_true - self.y_pred) ** 2)
        ss_tot = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        expected = 1 - (ss_res / ss_tot)
        actual = self.evaluator.r2(self.y_true, self.y_pred)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # R² should be very close to 1 for this near-perfect prediction
    
    def test_edge_cases(self):
        """Test edge cases for classification metrics."""
        # Test with zero values
        self.assertEqual(self.evaluator.precision(0, 0), 0.0)  # tp=0, fp=0
        self.assertEqual(self.evaluator.recall(0, 0), 0.0)     # tp=0, fn=0
        self.assertEqual(self.evaluator.f1_score(0, 0, 0), 0.0)  # tp=0, fp=0, fn=0
        
        # Test with zero true positives
        self.assertEqual(self.evaluator.precision(0, 10), 0.0)  # tp=0, fp=10
        self.assertEqual(self.evaluator.recall(0, 10), 0.0)     # tp=0, fn=10
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        # Perfect predictions should give RMSE=0, MAE=0, R²=1
        self.assertAlmostEqual(self.evaluator.rmse(y_true, y_pred), 0.0, places=10)
        self.assertAlmostEqual(self.evaluator.mae(y_true, y_pred), 0.0, places=10)
        self.assertAlmostEqual(self.evaluator.r2(y_true, y_pred), 1.0, places=10)
    
    def test_terrible_predictions(self):
        """Test with terrible predictions."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([10, 20, 30, 40, 50])  # Completely wrong
        
        # Terrible predictions should give high RMSE, high MAE, low R²
        self.assertGreater(self.evaluator.rmse(y_true, y_pred), 10.0)
        self.assertGreater(self.evaluator.mae(y_true, y_pred), 10.0)
        self.assertLess(self.evaluator.r2(y_true, y_pred), 0.0)  # Negative R²
    
    def test_random_state_consistency(self):
        """Test that random state ensures consistent results."""
        evaluator1 = ModelEvaluator(random_state=42)
        evaluator2 = ModelEvaluator(random_state=42)
        
        # Test with same inputs
        result1 = evaluator1.accuracy(self.tp, self.tn, self.fp, self.fn)
        result2 = evaluator2.accuracy(self.tp, self.tn, self.fp, self.fn)
        
        self.assertEqual(result1, result2)
    
    def test_formula_verification(self):
        """Verify that our formulas match the exact requirements."""
        # Test accuracy formula from requirements
        # accuracy = (tp + tn) / (tp + tn + fp + fp)
        tp, tn, fp, fn = 100, 80, 20, 10
        expected_accuracy = (tp + tn) / (tp + tn + fp + fp)
        actual_accuracy = self.evaluator.accuracy(tp, tn, fp, fn)
        self.assertAlmostEqual(actual_accuracy, expected_accuracy, places=10)
        
        # Test precision formula from requirements
        # precision = tp / (tp + fp)
        expected_precision = tp / (tp + fp)
        actual_precision = self.evaluator.precision(tp, fp)
        self.assertAlmostEqual(actual_precision, expected_precision, places=10)
        
        # Test recall formula from requirements
        # recall = tp / (tp + fn)
        expected_recall = tp / (tp + fn)
        actual_recall = self.evaluator.recall(tp, fn)
        self.assertAlmostEqual(actual_recall, expected_recall, places=10)
        
        # Test F1 formula from requirements
        # f1 = 2 * (precision * recall) / (precision + recall)
        expected_f1 = 2 * (expected_precision * expected_recall) / (expected_precision + expected_recall)
        actual_f1 = self.evaluator.f1_score(tp, fp, fn)
        self.assertAlmostEqual(actual_f1, expected_f1, places=10)
        
        # Test RMSE formula from requirements
        # rmse = sqrt(mean((y_true - y_pred)²))
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 1.9, 3.1])
        expected_rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        actual_rmse = self.evaluator.rmse(y_true, y_pred)
        self.assertAlmostEqual(actual_rmse, expected_rmse, places=10)
        
        # Test MAE formula from requirements
        # mae = mean(|y_true - y_pred|)
        expected_mae = np.mean(np.abs(y_true - y_pred))
        actual_mae = self.evaluator.mae(y_true, y_pred)
        self.assertAlmostEqual(actual_mae, expected_mae, places=10)
        
        # Test R² formula from requirements
        # r2 = 1 - (ss_res / ss_tot)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        expected_r2 = 1 - (ss_res / ss_tot)
        actual_r2 = self.evaluator.r2(y_true, y_pred)
        self.assertAlmostEqual(actual_r2, expected_r2, places=10)

class TestROISimulator(unittest.TestCase):
    """
    Test cases for ROI simulator formulas.
    """
    
    def setUp(self):
        """Set up test fixtures."""
        from roi_simulator import ROISimulator
        self.simulator = ROISimulator()
    
    def test_edge_formula(self):
        """Test edge formula: Edge = p̂ ⋅ (O - 1) - (1 - p̂)"""
        p = 0.6  # Model prediction
        odds = 2.0  # Decimal odds
        
        expected = p * (odds - 1) - (1 - p)
        actual = self.simulator.edge(p, odds)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # Edge = 0.6 * (2.0 - 1) - (1 - 0.6) = 0.6 * 1.0 - 0.4 = 0.6 - 0.4 = 0.2
    
    def test_roi_formula(self):
        """Test ROI formula: ROI = Total Profit / Total Stakes"""
        total_profit = 500
        total_stakes = 1000
        
        expected = total_profit / total_stakes
        actual = self.simulator.roi(total_profit, total_stakes)
        
        self.assertAlmostEqual(actual, expected, places=10)
        self.assertAlmostEqual(actual, 0.5, places=2)  # 500/1000 = 0.5
    
    def test_kelly_formula(self):
        """Test Kelly formula: f* = (bp - q) / b, where b = O - 1, p = p̂, q = 1 - p"""
        p = 0.6  # Model prediction
        odds = 2.0  # Decimal odds
        
        b = odds - 1
        q = 1 - p
        expected = (b * p - q) / b
        actual = self.simulator.kelly_fraction(p, odds)
        
        self.assertAlmostEqual(actual, expected, places=10)
        # Kelly = (1.0 * 0.6 - 0.4) / 1.0 = (0.6 - 0.4) / 1.0 = 0.2 / 1.0 = 0.2
    
    def test_edge_cases_roi(self):
        """Test edge cases for ROI calculations."""
        # Zero stakes should return 0 ROI
        self.assertEqual(self.simulator.roi(100, 0), 0.0)
        
        # Negative Kelly should be capped at 0
        p = 0.3  # Low probability
        odds = 1.5  # Low odds
        kelly = self.simulator.kelly_fraction(p, odds)
        self.assertEqual(kelly, 0.0)  # Should be capped at 0

if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)