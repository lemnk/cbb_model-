#!/usr/bin/env python3
"""
Simple test script for CBB Betting ML System.

This script tests basic functionality without requiring external dependencies.
"""

import os
import sys

def test_project_structure():
    """Test that the project structure is correct."""
    print("Testing project structure...")
    
    # Check directories exist
    required_dirs = [
        'src',
        'data',
        'data/raw',
        'data/processed',
        'data/backup',
        'notebooks'
    ]
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ Directory exists: {directory}")
        else:
            print(f"❌ Directory missing: {directory}")
            return False
    
    return True


def test_source_files():
    """Test that source files exist."""
    print("\nTesting source files...")
    
    required_files = [
        'src/__init__.py',
        'src/utils.py',
        'src/db.py',
        'src/scrape_games.py',
        'src/scrape_odds.py',
        'src/etl.py',
        'src/features.py',
        'src/train.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ File exists: {file}")
        else:
            print(f"❌ File missing: {file}")
            return False
    
    return True


def test_config_files():
    """Test that configuration files exist."""
    print("\nTesting configuration files...")
    
    required_files = [
        'config.yaml',
        'requirements.txt',
        'README.md',
        'setup.py',
        '.gitignore'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ File exists: {file}")
        else:
            print(f"❌ File missing: {file}")
            return False
    
    return True


def test_file_contents():
    """Test that key files have content."""
    print("\nTesting file contents...")
    
    # Check requirements.txt has content
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            content = f.read().strip()
            if len(content) > 0:
                print("✅ requirements.txt has content")
            else:
                print("❌ requirements.txt is empty")
                return False
    else:
        print("❌ requirements.txt not found")
        return False
    
    # Check config.yaml has content
    if os.path.exists('config.yaml'):
        with open('config.yaml', 'r') as f:
            content = f.read().strip()
            if len(content) > 0:
                print("✅ config.yaml has content")
            else:
                print("❌ config.yaml is empty")
                return False
    else:
        print("❌ config.yaml not found")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🏀 CBB Betting ML System - Simple Structure Test")
    print("=" * 50)
    
    tests = [
        ("Project Structure", test_project_structure),
        ("Source Files", test_source_files),
        ("Config Files", test_config_files),
        ("File Contents", test_file_contents),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All structure tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Configure database in config.yaml")
        print("3. Run full system test: python test_system.py")
        return 0
    else:
        print("⚠️  Some structure tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)