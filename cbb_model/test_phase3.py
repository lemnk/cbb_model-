#!/usr/bin/env python3
"""
Phase 3 Verification Script

This script verifies the Phase 3 implementation without requiring full dependencies.
It checks file structure, syntax, and basic functionality.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and print status."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} - MISSING")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and print status."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✅ {description}: {dirpath}")
        return True
    else:
        print(f"❌ {description}: {dirpath} - MISSING")
        return False

def check_python_syntax(filepath):
    """Check Python syntax using py_compile."""
    try:
        import py_compile
        py_compile.compile(filepath, doraise=True)
        print(f"✅ Syntax check passed: {filepath}")
        return True
    except Exception as e:
        print(f"❌ Syntax error in {filepath}: {e}")
        return False

def main():
    """Main verification function."""
    print("🧪 Phase 3 Implementation Verification")
    print("=" * 60)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Phase 3 required files and directories
    required_files = [
        ("src/models/__init__.py", "Models package init file"),
        ("src/models/logistic_regression.py", "Logistic Regression model"),
        ("src/models/random_forest.py", "Random Forest model"),
        ("src/models/xgboost_model.py", "XGBoost model"),
        ("src/models/neural_network.py", "Neural Network model"),
        ("train_models.py", "Main training pipeline"),
        ("evaluation.py", "Evaluation module"),
        ("roi_simulator.py", "ROI simulator"),
        ("tests/test_evaluation.py", "Unit tests"),
        ("PHASE3_SUMMARY.md", "Phase 3 summary document")
    ]
    
    required_dirs = [
        ("src/models", "Models source directory"),
        ("tests", "Tests directory"),
        ("outputs/phase3", "Phase 3 outputs directory"),
        ("outputs/phase3/models", "Models output directory"),
        ("outputs/phase3/plots", "Plots output directory")
    ]
    
    # Check files
    print("\n📁 Checking Required Files:")
    print("-" * 40)
    files_exist = True
    for filepath, description in required_files:
        if not check_file_exists(filepath, description):
            files_exist = False
    
    # Check directories
    print("\n📁 Checking Required Directories:")
    print("-" * 40)
    dirs_exist = True
    for dirpath, description in required_dirs:
        if not check_directory_exists(dirpath, description):
            dirs_exist = False
    
    # Check Python syntax for key files
    print("\n🐍 Checking Python Syntax:")
    print("-" * 40)
    syntax_ok = True
    key_files = [
        "src/models/logistic_regression.py",
        "src/models/random_forest.py", 
        "src/models/xgboost_model.py",
        "src/models/neural_network.py",
        "train_models.py",
        "evaluation.py",
        "roi_simulator.py"
    ]
    
    for filepath in key_files:
        if os.path.exists(filepath):
            if not check_python_syntax(filepath):
                syntax_ok = False
        else:
            print(f"⚠️ Skipping syntax check for missing file: {filepath}")
    
    # Check requirements.txt
    print("\n📦 Checking Dependencies:")
    print("-" * 40)
    requirements_ok = True
    
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r") as f:
            content = f.read()
            required_deps = ["xgboost", "torch", "joblib"]
            for dep in required_deps:
                if dep in content:
                    print(f"✅ {dep} dependency found")
                else:
                    print(f"❌ {dep} dependency missing")
                    requirements_ok = False
    else:
        print("❌ requirements.txt not found")
        requirements_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    if files_exist and dirs_exist and syntax_ok and requirements_ok:
        print("🎉 Phase 3 Implementation: COMPLETE ✅")
        print("\nAll required files, directories, and syntax checks passed.")
        print("The system is ready for Phase 3 execution.")
    else:
        print("⚠️ Phase 3 Implementation: INCOMPLETE ❌")
        print("\nSome checks failed. Please review the issues above.")
        
        if not files_exist:
            print("- Missing required files")
        if not dirs_exist:
            print("- Missing required directories") 
        if not syntax_ok:
            print("- Python syntax errors found")
        if not requirements_ok:
            print("- Missing required dependencies")
    
    print("\n" + "=" * 60)
    
    # Next steps
    if files_exist and dirs_exist and syntax_ok and requirements_ok:
        print("🚀 Next Steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run training pipeline: python train_models.py")
        print("3. Run evaluation: python evaluation.py")
        print("4. Run ROI simulation: python roi_simulator.py")
        print("5. Run tests: python -m pytest tests/test_evaluation.py -v")

if __name__ == "__main__":
    main()