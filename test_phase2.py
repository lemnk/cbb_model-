#!/usr/bin/env python3
"""
Simple test script for Phase 2: Feature Engineering
Tests the structure and basic functionality without requiring external dependencies.
"""

import os
import sys

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("Testing Phase 2 directory structure...")
    
    required_dirs = [
        'src/features',
        'notebooks'
    ]
    
    required_files = [
        'src/features/__init__.py',
        'src/features/team_features.py',
        'src/features/dynamic_features.py',
        'src/features/player_features.py',
        'src/features/market_features.py',
        'src/features/feature_pipeline.py',
        'src/features/feature_utils.py',
        'notebooks/feature_exploration.py',
        'PHASE2_SUMMARY.md'
    ]
    
    # Test directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    # Test files
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    return True

def test_file_contents():
    """Test that key files have expected content."""
    print("\nTesting file contents...")
    
    # Test PHASE2_SUMMARY.md
    if os.path.exists('PHASE2_SUMMARY.md'):
        with open('PHASE2_SUMMARY.md', 'r') as f:
            content = f.read()
            if 'Phase 2: Feature Engineering - COMPLETED ✅' in content:
                print("✅ PHASE2_SUMMARY.md has correct content")
            else:
                print("❌ PHASE2_SUMMARY.md missing expected content")
                return False
    else:
        print("❌ PHASE2_SUMMARY.md not found")
        return False
    
    # Test features __init__.py
    if os.path.exists('src/features/__init__.py'):
        with open('src/features/__init__.py', 'r') as f:
            content = f.read()
            if 'Feature Engineering Package' in content:
                print("✅ src/features/__init__.py has correct content")
            else:
                print("❌ src/features/__init__.py missing expected content")
                return False
    else:
        print("❌ src/features/__init__.py not found")
        return False
    
    return True

def test_feature_modules():
    """Test that feature modules have expected structure."""
    print("\nTesting feature module structure...")
    
    feature_modules = [
        'team_features.py',
        'dynamic_features.py', 
        'player_features.py',
        'market_features.py',
        'feature_pipeline.py',
        'feature_utils.py'
    ]
    
    for module in feature_modules:
        file_path = f'src/features/{module}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                # Check for key class definitions
                if 'class ' in content and 'def ' in content:
                    print(f"✅ {module} has classes and methods")
                else:
                    print(f"❌ {module} missing expected structure")
                    return False
        else:
            print(f"❌ {module} not found")
            return False
    
    return True

def test_notebook_script():
    """Test that the feature exploration script exists."""
    print("\nTesting feature exploration script...")
    
    script_path = 'notebooks/feature_exploration.py'
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            content = f.read()
            if 'def main()' in content and 'create_feature_pipeline' in content:
                print("✅ Feature exploration script has correct structure")
                return True
            else:
                print("❌ Feature exploration script missing expected content")
                return False
    else:
        print("❌ Feature exploration script not found")
        return False

def main():
    """Run all Phase 2 tests."""
    print("🏀 Phase 2: Feature Engineering - Structure Test")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_file_contents,
        test_feature_modules,
        test_notebook_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Phase 2 Structure Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 structure tests passed!")
        print("✅ Feature engineering system is properly structured")
        print("✅ Ready for dependency installation and testing")
    else:
        print("❌ Some Phase 2 structure tests failed")
        print("Please check the missing components above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)#!/usr/bin/env python3
"""
Simple test script for Phase 2: Feature Engineering
Tests the structure and basic functionality without requiring external dependencies.
"""

import os
import sys

def test_directory_structure():
    """Test that all required directories and files exist."""
    print("Testing Phase 2 directory structure...")
    
    required_dirs = [
        'src/features',
        'notebooks'
    ]
    
    required_files = [
        'src/features/__init__.py',
        'src/features/team_features.py',
        'src/features/dynamic_features.py',
        'src/features/player_features.py',
        'src/features/market_features.py',
        'src/features/feature_pipeline.py',
        'src/features/feature_utils.py',
        'notebooks/feature_exploration.py',
        'PHASE2_SUMMARY.md'
    ]
    
    # Test directories
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"✅ Directory exists: {dir_path}")
        else:
            print(f"❌ Directory missing: {dir_path}")
            return False
    
    # Test files
    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"✅ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")
            return False
    
    return True

def test_file_contents():
    """Test that key files have expected content."""
    print("\nTesting file contents...")
    
    # Test PHASE2_SUMMARY.md
    if os.path.exists('PHASE2_SUMMARY.md'):
        with open('PHASE2_SUMMARY.md', 'r') as f:
            content = f.read()
            if 'Phase 2: Feature Engineering - COMPLETED ✅' in content:
                print("✅ PHASE2_SUMMARY.md has correct content")
            else:
                print("❌ PHASE2_SUMMARY.md missing expected content")
                return False
    else:
        print("❌ PHASE2_SUMMARY.md not found")
        return False
    
    # Test features __init__.py
    if os.path.exists('src/features/__init__.py'):
        with open('src/features/__init__.py', 'r') as f:
            content = f.read()
            if 'Feature Engineering Package' in content:
                print("✅ src/features/__init__.py has correct content")
            else:
                print("❌ src/features/__init__.py missing expected content")
                return False
    else:
        print("❌ src/features/__init__.py not found")
        return False
    
    return True

def test_feature_modules():
    """Test that feature modules have expected structure."""
    print("\nTesting feature module structure...")
    
    feature_modules = [
        'team_features.py',
        'dynamic_features.py', 
        'player_features.py',
        'market_features.py',
        'feature_pipeline.py',
        'feature_utils.py'
    ]
    
    for module in feature_modules:
        file_path = f'src/features/{module}'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                content = f.read()
                # Check for key class definitions
                if 'class ' in content and 'def ' in content:
                    print(f"✅ {module} has classes and methods")
                else:
                    print(f"❌ {module} missing expected structure")
                    return False
        else:
            print(f"❌ {module} not found")
            return False
    
    return True

def test_notebook_script():
    """Test that the feature exploration script exists."""
    print("\nTesting feature exploration script...")
    
    script_path = 'notebooks/feature_exploration.py'
    if os.path.exists(script_path):
        with open(script_path, 'r') as f:
            content = f.read()
            if 'def main()' in content and 'create_feature_pipeline' in content:
                print("✅ Feature exploration script has correct structure")
            else:
                print("❌ Feature exploration script missing expected content")
                return False
    else:
        print("❌ Feature exploration script not found")
        return False
    
    return True

def main():
    """Run all Phase 2 tests."""
    print("🏀 Phase 2: Feature Engineering - Structure Test")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_file_contents,
        test_feature_modules,
        test_notebook_script
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ Test failed: {test.__name__}")
        except Exception as e:
            print(f"❌ Test error in {test.__name__}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Phase 2 Structure Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 2 structure tests passed!")
        print("✅ Feature engineering system is properly structured")
        print("✅ Ready for dependency installation and testing")
    else:
        print("❌ Some Phase 2 structure tests failed")
        print("Please check the missing components above")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)