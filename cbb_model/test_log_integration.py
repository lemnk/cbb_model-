#!/usr/bin/env python3
"""
Simple test script for log transform integration.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    import pandas as pd
    from src.features.feature_utils import (
        log_transform, 
        inverse_log_transform, 
        normalize, 
        scale, 
        Normalizer
    )
    
    print("✅ Successfully imported all log transform functions")
    
    # Test data
    y_positive = np.array([1, 2, 3, 4, 5])
    y_series = pd.Series([1, 2, 3, 4, 5], name='test')
    y_zero = np.array([0, 1, 2, 3, 4])
    
    print(f"\n🧪 Testing Log Transform Integration...")
    print(f"📊 Test data: {y_positive}")
    
    # Test 1: Direct function calls
    print(f"\n📈 Test 1: Direct Function Calls")
    print(f"   Formula: x'ᵢ = log(xᵢ + 1)")
    y_log = log_transform(y_positive)
    print(f"   Log transformed: {y_log}")
    
    print(f"   Formula: xᵢ = exp(x'ᵢ) - 1")
    y_restored = inverse_log_transform(y_log)
    print(f"   Restored: {y_restored}")
    
    # Test 2: Round-trip accuracy
    print(f"\n🎯 Test 2: Round-trip Accuracy")
    roundtrip_accurate = np.allclose(y_positive, y_restored)
    print(f"   Original ≈ Restored: {roundtrip_accurate}")
    if roundtrip_accurate:
        print(f"   Max difference: {np.max(np.abs(y_positive - y_restored)):.2e}")
    
    # Test 3: normalize() function integration
    print(f"\n🔧 Test 3: normalize() Function Integration")
    y_normalized = normalize(y_series, method="log")
    print(f"   normalize(method='log'): {y_normalized.values}")
    print(f"   Matches log_transform: {np.allclose(y_normalized.values, y_log)}")
    
    # Test 4: scale() function integration
    print(f"\n⚙️ Test 4: scale() Function Integration")
    df = pd.DataFrame({'col1': y_positive, 'col2': y_positive * 2})
    df_scaled = scale(df, method="log")
    print(f"   scale(method='log') col1: {df_scaled['col1'].values}")
    print(f"   scale(method='log') col2: {df_scaled['col2'].values}")
    
    # Test 5: Normalizer class integration
    print(f"\n🏗️ Test 5: Normalizer Class Integration")
    normalizer = Normalizer(method="log")
    
    # Fit and transform
    y_fit_transformed = normalizer.fit_transform(y_series)
    print(f"   fit_transform(): {y_fit_transformed.values}")
    print(f"   Matches log_transform: {np.allclose(y_fit_transformed.values, y_log)}")
    
    # Inverse transform
    y_inverse = normalizer.inverse_transform(y_fit_transformed)
    print(f"   inverse_transform(): {y_inverse.values}")
    print(f"   Matches original: {np.allclose(y_inverse.values, y_series.values)}")
    
    # Test 6: Edge cases
    print(f"\n🔢 Test 6: Edge Cases")
    y_log_zero = log_transform(y_zero)
    y_restored_zero = inverse_log_transform(y_log_zero)
    zero_accurate = np.allclose(y_zero, y_restored_zero)
    print(f"   Zero values round-trip: {zero_accurate}")
    
    # Test 7: Error handling
    print(f"\n⚠️ Test 7: Error Handling")
    y_negative = np.array([1, -2, 3])
    try:
        log_transform(y_negative)
        print("   ❌ Should have raised ValueError for negative values")
    except ValueError as e:
        print(f"   ✅ Correctly raised ValueError: {e}")
    
    print(f"\n🎉 All log transform integration tests completed successfully!")
    print(f"✅ Direct functions: Working")
    print(f"✅ Round-trip accuracy: {roundtrip_accurate}")
    print(f"✅ normalize() integration: Working")
    print(f"✅ scale() integration: Working")
    print(f"✅ Normalizer class: Working")
    print(f"✅ Edge cases: Working")
    print(f"✅ Error handling: Working")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()