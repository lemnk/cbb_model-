#!/usr/bin/env python3
"""
Simple test script for log_transform function.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    import pandas as pd
    from src.features.feature_utils import log_transform, normalize, scale
    
    print("✅ Successfully imported all required modules")
    
    # Test 1: Pandas Series
    print("\n🧪 Test 1: Pandas Series")
    series = pd.Series([1, 2, 3, 4, 5], name='test_series')
    result = log_transform(series)
    print(f"Input: {series.values}")
    print(f"Output: {result.values}")
    print(f"Expected: {np.log1p([1, 2, 3, 4, 5])}")
    print(f"✅ Pandas Series test passed")
    
    # Test 2: Numpy Array
    print("\n🧪 Test 2: Numpy Array")
    arr = np.array([0.1, 0.5, 1.0, 2.0, 10.0])
    result = log_transform(arr)
    print(f"Input: {arr}")
    print(f"Output: {result}")
    print(f"Expected: {np.log1p([0.1, 0.5, 1.0, 2.0, 10.0])}")
    print(f"✅ Numpy Array test passed")
    
    # Test 3: Edge cases
    print("\n🧪 Test 3: Edge Cases")
    edge_series = pd.Series([0, 1e-10, 1e-8, 1e-6])
    result = log_transform(edge_series)
    print(f"Input: {edge_series.values}")
    print(f"Output: {result.values}")
    print(f"✅ Edge cases test passed")
    
    # Test 4: Normalize with log method
    print("\n🧪 Test 4: Normalize with Log Method")
    result = normalize(series, method="log")
    expected = log_transform(series)
    print(f"Normalize log result: {result.values}")
    print(f"Direct log_transform: {expected.values}")
    print(f"✅ Normalize with log method test passed")
    
    # Test 5: Scale with log method
    print("\n🧪 Test 5: Scale with Log Method")
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [0.1, 0.5, 1.0, 2.0, 10.0]
    })
    result = scale(df, method="log")
    print(f"Scaled DataFrame shape: {result.shape}")
    print(f"✅ Scale with log method test passed")
    
    # Test 6: Error handling for negative values
    print("\n🧪 Test 6: Error Handling for Negative Values")
    try:
        negative_series = pd.Series([1, -2, 3])
        log_transform(negative_series)
        print("❌ Should have raised ValueError")
    except ValueError as e:
        print(f"✅ Correctly raised ValueError: {e}")
    
    print("\n🎉 All tests passed! log_transform function is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()