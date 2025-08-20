#!/usr/bin/env python3
"""
Unit tests for feature_utils.py

Tests all utility functions including the new log_transform function.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.feature_utils import (
    log_transform, 
    normalize, 
    scale, 
    handle_missing,
    encode_categorical,
    remove_outliers
)

class TestLogTransform(unittest.TestCase):
    """Test cases for log_transform function."""
    
    def test_log_transform_pandas_series(self):
        """Test log transform with pandas Series."""
        # Test with positive values
        series = pd.Series([1, 2, 3, 4, 5])
        result = log_transform(series)
        
        # Check that result is pandas Series
        self.assertIsInstance(result, pd.Series)
        
        # Check that result has same index
        pd.testing.assert_index_equal(result.index, series.index)
        
        # Check mathematical correctness: log(x + 1)
        expected = np.log1p([1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(result.values, expected)
    
    def test_log_transform_numpy_array(self):
        """Test log transform with numpy array."""
        # Test with positive values
        arr = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
        result = log_transform(arr)
        
        # Check that result is numpy array
        self.assertIsInstance(result, np.ndarray)
        
        # Check mathematical correctness
        expected = np.log1p([0.1, 0.5, 1.0, 2.0, 10.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_log_transform_edge_cases(self):
        """Test log transform with edge cases."""
        # Test with zero values
        series = pd.Series([0, 1, 2])
        result = log_transform(series)
        expected = np.log1p([0, 1, 2])
        np.testing.assert_array_almost_equal(result.values, expected)
        
        # Test with very small positive values
        series = pd.Series([1e-10, 1e-8, 1e-6])
        result = log_transform(series)
        expected = np.log1p([1e-10, 1e-8, 1e-6])
        np.testing.assert_array_almost_equal(result.values, expected)
    
    def test_log_transform_negative_values_error(self):
        """Test that log transform raises error for negative values."""
        # Test with pandas Series containing negative values
        series = pd.Series([1, -2, 3])
        with self.assertRaises(ValueError) as context:
            log_transform(series)
        self.assertIn("non-negative", str(context.exception))
        
        # Test with numpy array containing negative values
        arr = np.array([0.1, -0.5, 1.0])
        with self.assertRaises(ValueError) as context:
            log_transform(arr)
        self.assertIn("non-negative", str(context.exception))
    
    def test_log_transform_numerical_stability(self):
        """Test that log1p is used for numerical stability."""
        # Test with very small values where log1p(x) != log(1+x) due to precision
        series = pd.Series([1e-15, 1e-10, 1e-5])
        result = log_transform(series)
        
        # Verify using log1p (should be very close to the input for very small values)
        expected_log1p = np.log1p([1e-15, 1e-10, 1e-5])
        expected_log_plus_1 = np.log([1e-15 + 1, 1e-10 + 1, 1e-5 + 1])
        
        # log1p should be more accurate for very small values
        np.testing.assert_array_almost_equal(result.values, expected_log1p)
        
        # The difference should be minimal but log1p should be more accurate
        diff_log1p = np.abs(result.values - expected_log1p)
        diff_log_plus_1 = np.abs(result.values - expected_log_plus_1)
        
        # log1p should be at least as accurate as log(1+x)
        self.assertTrue(np.all(diff_log1p <= diff_log_plus_1 + 1e-16))

class TestNormalizeWithLog(unittest.TestCase):
    """Test cases for normalize function with log method."""
    
    def test_normalize_log_method(self):
        """Test that normalize function works with log method."""
        series = pd.Series([1, 2, 3, 4, 5])
        result = normalize(series, method="log")
        
        # Should return same result as direct log_transform
        expected = log_transform(series)
        pd.testing.assert_series_equal(result, expected)
    
    def test_normalize_log_method_negative_error(self):
        """Test that normalize with log method raises error for negative values."""
        series = pd.Series([1, -2, 3])
        with self.assertRaises(ValueError):
            normalize(series, method="log")

class TestScaleWithLog(unittest.TestCase):
    """Test cases for scale function with log method."""
    
    def test_scale_log_method(self):
        """Test that scale function works with log method."""
        df = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [0.1, 0.5, 1.0, 2.0, 10.0]
        })
        
        result = scale(df, method="log")
        
        # Check that both columns are log transformed
        expected_col1 = log_transform(df['col1'])
        expected_col2 = log_transform(df['col2'])
        
        pd.testing.assert_series_equal(result['col1'], expected_col1)
        pd.testing.assert_series_equal(result['col2'], expected_col2)
    
    def test_scale_log_method_negative_error(self):
        """Test that scale with log method raises error for negative values."""
        df = pd.DataFrame({
            'col1': [1, -2, 3],
            'col2': [0.1, 0.5, 1.0]
        })
        
        with self.assertRaises(ValueError):
            scale(df, method="log")

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)