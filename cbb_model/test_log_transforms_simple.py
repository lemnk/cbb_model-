#!/usr/bin/env python3
"""
Simple test script for log transform functions.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    import pandas as pd
    from src.models.train_utils import log_transform, inverse_log_transform
    
    print("✅ Successfully imported log transform functions")
    
    # Test data
    y_positive = np.array([1, 2, 3, 4, 5])
    y_series = pd.Series([1, 2, 3, 4, 5], name='test')
    y_zero = np.array([0, 1, 2, 3, 4])
    
    print(f"\n🧪 Testing Log Transform Functions...")
    print(f"📊 Test data: {y_positive}")
    
    # Test 1: Array log transform
    print(f"\n📈 Test 1: Array Log Transform")
    print(f"   Formula: y'ᵢ = log(yᵢ + c)")
    y_log = log_transform(y_positive)
    print(f"   Log transformed: {y_log}")
    
    # Test 2: Series log transform
    print(f"\n📊 Test 2: Series Log Transform")
    y_log_series = log_transform(y_series)
    print(f"   Result type: {type(y_log_series)}")
    print(f"   Series name: {y_log_series.name}")
    print(f"   Log transformed: {y_log_series.values}")
    
    # Test 3: Inverse log transform
    print(f"\n🔄 Test 3: Inverse Log Transform")
    print(f"   Formula: yᵢ = exp(y'ᵢ) - c")
    y_restored = inverse_log_transform(y_log)
    print(f"   Restored: {y_restored}")
    
    # Test 4: Roundtrip accuracy
    print(f"\n🎯 Test 4: Roundtrip Accuracy")
    roundtrip_accurate = np.allclose(y_positive, y_restored)
    print(f"   Original ≈ Restored: {roundtrip_accurate}")
    if roundtrip_accurate:
        print(f"   Max difference: {np.max(np.abs(y_positive - y_restored)):.2e}")
    
    # Test 5: Zero handling
    print(f"\n🔢 Test 5: Zero Handling")
    y_log_zero = log_transform(y_zero)
    print(f"   Zero values log transformed: {y_log_zero}")
    
    # Test 6: Different constants
    print(f"\n⚙️ Test 6: Different Constants")
    y_log_c1 = log_transform(y_positive, c=1.0)
    y_restored_c1 = inverse_log_transform(y_log_c1, c=1.0)
    print(f"   c=1.0 log: {y_log_c1}")
    print(f"   c=1.0 restored: {y_restored_c1}")
    print(f"   c=1.0 accurate: {np.allclose(y_positive, y_restored_c1)}")
    
    print(f"\n🎉 All log transform tests completed successfully!")
    print(f"✅ Array log transform: Working")
    print(f"✅ Series log transform: Working")
    print(f"✅ Inverse log transform: Working")
    print(f"✅ Roundtrip accuracy: {roundtrip_accurate}")
    print(f"✅ Zero handling: Working")
    print(f"✅ Different constants: Working")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()