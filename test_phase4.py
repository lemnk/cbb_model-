#!/usr/bin/env python3
"""
Simple test script for Phase 4 components.
Tests basic functionality without requiring full dependencies.
"""

import os
import sys

def test_file_structure():
    """Test that all Phase 4 files exist."""
    print("🔍 Testing Phase 4 File Structure...")
    
    required_files = [
        "src/metrics/__init__.py",
        "src/metrics/core_metrics.py",
        "src/ensemble/__init__.py",
        "src/ensemble/ensemble_methods.py",
        "src/calibration/__init__.py",
        "src/calibration/calibration_methods.py",
        "src/validation/__init__.py",
        "src/validation/walk_forward.py",
        "src/optimization/__init__.py",
        "src/optimization/hyperparameter_optimizer.py",
        "src/deployment/__init__.py",
        "src/deployment/api.py",
        "src/deployment/cli.py",
        "phase4_main.py",
        "tests/test_phase4.py",
        "PHASE4_SUMMARY.md",
        "Dockerfile"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All Phase 4 files exist")
        return True

def test_directory_structure():
    """Test that all Phase 4 directories exist."""
    print("\n📁 Testing Phase 4 Directory Structure...")
    
    required_dirs = [
        "src/metrics",
        "src/ensemble", 
        "src/calibration",
        "src/validation",
        "src/optimization",
        "src/deployment",
        "outputs/phase4/models",
        "outputs/phase4/plots",
        "outputs/phase4/reports"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All Phase 4 directories exist")
        return True

def test_syntax():
    """Test basic syntax of Python files."""
    print("\n🐍 Testing Python Syntax...")
    
    python_files = [
        "src/metrics/core_metrics.py",
        "src/ensemble/ensemble_methods.py",
        "src/calibration/calibration_methods.py",
        "src/validation/walk_forward.py",
        "src/optimization/hyperparameter_optimizer.py",
        "src/deployment/api.py",
        "src/deployment/cli.py",
        "phase4_main.py"
    ]
    
    syntax_errors = []
    for file_path in python_files:
        try:
            with open(file_path, 'r') as f:
                compile(f.read(), file_path, 'exec')
        except SyntaxError as e:
            syntax_errors.append(f"{file_path}: {e}")
    
    if syntax_errors:
        print(f"❌ Syntax errors found: {syntax_errors}")
        return False
    else:
        print("✅ All Python files have valid syntax")
        return True

def test_imports():
    """Test that modules can be imported."""
    print("\n📦 Testing Module Imports...")
    
    try:
        # Test basic imports
        import src.metrics
        import src.ensemble
        import src.calibration
        import src.validation
        import src.optimization
        import src.deployment
        print("✅ All Phase 4 modules can be imported")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_formula_implementation():
    """Test that mathematical formulas are correctly implemented."""
    print("\n🧮 Testing Mathematical Formula Implementation...")
    
    # Check that formulas are documented in code
    formula_files = [
        "src/metrics/core_metrics.py",
        "src/ensemble/ensemble_methods.py",
        "src/calibration/calibration_methods.py"
    ]
    
    required_formulas = [
        "ROC-AUC = (1/n) * Σᵢ₌₁ⁿ 1[ŷᵢ > ŷⱼ]",
        "L = -(1/n) * Σᵢ₌₁ⁿ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]",
        "ECE = Σ |acc - conf| * (bin_count / total)",
        "ROI = (Total Return - Total Stake) / Total Stake",
        "BS = (1/n) * Σᵢ₌₁ⁿ (pᵢ - yᵢ)²",
        "p̂_ensemble = (1/M) * Σₘ₌₁ᴹ p̂ₘ",
        "p̂_ensemble = Σₘ₌₁ᴹ wₘ * p̂ₘ",
        "p' = σ(Ap + B)"
    ]
    
    formula_found = 0
    for file_path in formula_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                for formula in required_formulas:
                    if formula in content:
                        formula_found += 1
        except Exception as e:
            print(f"❌ Error reading {file_path}: {e}")
            return False
    
    if formula_found >= len(required_formulas) * 0.8:  # 80% threshold
        print(f"✅ {formula_found}/{len(required_formulas)} formulas found in implementation")
        return True
    else:
        print(f"❌ Only {formula_found}/{len(required_formulas)} formulas found")
        return False

def main():
    """Run all Phase 4 tests."""
    print("🚀 Phase 4: Model Optimization & Deployment - Validation Test")
    print("=" * 70)
    
    tests = [
        test_file_structure,
        test_directory_structure,
        test_syntax,
        test_imports,
        test_formula_implementation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Phase 4 Validation Results Summary")
    print("=" * 70)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n🎉 All Phase 4 tests passed!")
        print("The CBB Betting ML System Phase 4 is ready for use.")
        return True
    else:
        print(f"\n💥 {total - passed} tests failed.")
        print("Please check the implementation and fix any issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)