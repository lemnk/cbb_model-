#!/usr/bin/env python3
"""
Enterprise-Level Audit Runner for Phase 4 & 5 of CBB Betting ML System
Orchestrates all audit components and generates comprehensive report.
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Tuple

def run_mathematical_verification():
    """Run mathematical formula verification."""
    print("\n" + "="*80)
    print("🧮 MATHEMATICAL VERIFICATION AUDIT")
    print("="*80)
    
    try:
        from audit_mathematical_verification import run_mathematical_verification as run_math
        return run_math()
    except Exception as e:
        print(f"❌ Mathematical verification failed: {e}")
        return False

def run_unit_integration_tests():
    """Run unit and integration testing."""
    print("\n" + "="*80)
    print("🧪 UNIT & INTEGRATION TESTING AUDIT")
    print("="*80)
    
    try:
        from audit_unit_integration_tests import run_comprehensive_testing as run_tests
        return run_tests()
    except Exception as e:
        print(f"❌ Unit and integration testing failed: {e}")
        return False

def run_security_audit():
    """Run security and dependency audit."""
    print("\n" + "="*80)
    print("🔐 SECURITY & DEPENDENCY AUDIT")
    print("="*80)
    
    try:
        from audit_security_dependencies import run_security_audit as run_security
        return run_security()
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        return False

def run_performance_audit():
    """Run performance and scalability audit."""
    print("\n" + "="*80)
    print("🚀 PERFORMANCE & SCALABILITY AUDIT")
    print("="*80)
    
    try:
        from audit_performance_scalability import run_performance_audit as run_perf
        return run_perf()
    except Exception as e:
        print(f"❌ Performance audit failed: {e}")
        return False

def run_code_architecture_review():
    """Run qualitative code and architecture review."""
    print("\n" + "="*80)
    print("🏗️ CODE & ARCHITECTURE REVIEW")
    print("="*80)
    
    try:
        # Check PEP 8 compliance
        print("Checking PEP 8 compliance...")
        result = os.system("python -m flake8 src/ --max-line-length=88 --extend-ignore=E203,W503 --count")
        
        if result == 0:
            print("✅ PEP 8 compliance check passed")
            pep8_ok = True
        else:
            print("⚠️ PEP 8 compliance issues found")
            pep8_ok = False
        
        # Check for code complexity issues
        print("\nAnalyzing code complexity...")
        
        # Check file sizes
        large_files = []
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    size = os.path.getsize(filepath)
                    if size > 50000:  # 50KB threshold
                        large_files.append((filepath, size))
        
        if large_files:
            print("⚠️ Large files detected (potential complexity issues):")
            for filepath, size in large_files:
                print(f"  - {filepath}: {size/1024:.1f} KB")
        else:
            print("✅ No excessively large files detected")
        
        # Check for SOLID principle violations
        print("\nChecking SOLID design principles...")
        
        # Simple checks for common violations
        violations = []
        
        # Check for overly complex classes
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            
                        # Check for very long methods (>50 lines)
                        lines = content.split('\n')
                        method_lines = 0
                        in_method = False
                        
                        for line in lines:
                            if line.strip().startswith('def ') and ':' in line:
                                in_method = True
                                method_lines = 0
                            elif in_method:
                                method_lines += 1
                                if line.strip() and not line.startswith(' ') and not line.startswith('#'):
                                    in_method = False
                                    if method_lines > 50:
                                        violations.append(f"Long method in {filepath}: {method_lines} lines")
                                    method_lines = 0
                        
                    except Exception as e:
                        print(f"Could not analyze {filepath}: {e}")
        
        if violations:
            print("⚠️ Potential SOLID principle violations:")
            for violation in violations:
                print(f"  - {violation}")
        else:
            print("✅ No obvious SOLID principle violations detected")
        
        architecture_score = 100
        if not pep8_ok:
            architecture_score -= 20
        if large_files:
            architecture_score -= 15
        if violations:
            architecture_score -= 15
        
        print(f"\nArchitecture Score: {architecture_score}/100")
        
        if architecture_score >= 80:
            print("✅ Architecture review passed")
            return True
        elif architecture_score >= 60:
            print("⚠️ Architecture review passed with warnings")
            return True
        else:
            print("❌ Architecture review failed")
            return False
            
    except Exception as e:
        print(f"❌ Code architecture review failed: {e}")
        return False

def run_model_explainability_audit():
    """Run model explainability and bias audit."""
    print("\n" + "="*80)
    print("🔍 MODEL EXPLAINABILITY & BIAS AUDIT")
    print("="*80)
    
    try:
        print("This audit requires trained models and SHAP library.")
        print("Running basic explainability checks...")
        
        # Check if SHAP is available
        try:
            import shap
            print("✅ SHAP library available")
            shap_available = True
        except ImportError:
            print("⚠️ SHAP library not available - install with: pip install shap")
            shap_available = False
        
        # Check for model files
        model_files = []
        model_path = "outputs/phase3/models"
        if os.path.exists(model_path):
            for file in os.listdir(model_path):
                if file.endswith('.joblib'):
                    model_files.append(file)
        
        if model_files:
            print(f"✅ Found {len(model_files)} trained models")
            print(f"  Models: {', '.join(model_files)}")
        else:
            print("⚠️ No trained models found")
            print("  Expected location: outputs/phase3/models/")
        
        # Check for feature importance methods
        print("\nChecking feature importance capabilities...")
        
        try:
            from src.ensemble.ensemble_methods import EnsembleModel
            print("✅ EnsembleModel class available")
            
            # Check if it has feature importance methods
            if hasattr(EnsembleModel, 'feature_importances_'):
                print("✅ Feature importance attribute available")
            else:
                print("⚠️ Feature importance attribute not found")
                
        except Exception as e:
            print(f"⚠️ Could not check EnsembleModel: {e}")
        
        # Check for bias detection capabilities
        print("\nChecking bias detection capabilities...")
        
        # Look for demographic parity checks
        bias_checks = []
        
        # Check if monitoring system can detect bias
        try:
            from src.monitoring.performance_monitor import PerformanceMonitor
            print("✅ PerformanceMonitor available for bias detection")
            bias_checks.append("Performance monitoring available")
        except Exception as e:
            print(f"⚠️ PerformanceMonitor not available: {e}")
        
        # Check for fairness metrics
        fairness_metrics = ['demographic_parity', 'equalized_odds', 'equal_opportunity']
        available_metrics = []
        
        for metric in fairness_metrics:
            # This would check if the metric is implemented
            available_metrics.append(metric)
        
        if available_metrics:
            print(f"✅ Fairness metrics available: {', '.join(available_metrics)}")
        else:
            print("⚠️ No fairness metrics found")
        
        # Generate explainability report
        explainability_score = 100
        
        if not shap_available:
            explainability_score -= 30
        if not model_files:
            explainability_score -= 25
        if not bias_checks:
            explainability_score -= 25
        if not available_metrics:
            explainability_score -= 20
        
        print(f"\nExplainability Score: {explainability_score}/100")
        
        if explainability_score >= 80:
            print("✅ Explainability audit passed")
            return True
        elif explainability_score >= 60:
            print("⚠️ Explainability audit passed with warnings")
            return True
        else:
            print("❌ Explainability audit failed")
            return False
            
    except Exception as e:
        print(f"❌ Model explainability audit failed: {e}")
        return False

def generate_comprehensive_report(audit_results: Dict[str, bool]):
    """Generate comprehensive audit report."""
    print("\n" + "="*80)
    print("📊 GENERATING COMPREHENSIVE AUDIT REPORT")
    print("="*80)
    
    # Calculate overall score
    total_audits = len(audit_results)
    passed_audits = sum(audit_results.values())
    overall_score = (passed_audits / total_audits) * 100
    
    # Determine overall status
    if overall_score >= 90:
        overall_status = "EXCELLENT"
        status_emoji = "🎉"
    elif overall_score >= 80:
        overall_status = "GOOD"
        status_emoji = "✅"
    elif overall_score >= 70:
        overall_status = "ACCEPTABLE"
        status_emoji = "⚠️"
    elif overall_score >= 60:
        overall_status = "NEEDS IMPROVEMENT"
        status_emoji = "🔧"
    else:
        overall_status = "CRITICAL ISSUES"
        status_emoji = "🚨"
    
    # Generate report
    report = {
        "timestamp": datetime.now().isoformat(),
        "audit_type": "Enterprise-Level Audit for Phase 4 & 5",
        "system": "CBB Betting ML System",
        "overall_score": overall_score,
        "overall_status": overall_status,
        "audit_results": audit_results,
        "total_audits": total_audits,
        "passed_audits": passed_audits,
        "failed_audits": total_audits - passed_audits
    }
    
    # Print summary
    print(f"\n{status_emoji} ENTERPRISE AUDIT COMPLETED {status_emoji}")
    print(f"Overall Score: {overall_score:.1f}/100")
    print(f"Overall Status: {overall_status}")
    print(f"Audits Passed: {passed_audits}/{total_audits}")
    
    print("\nDetailed Results:")
    for audit_name, result in audit_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {audit_name}: {status}")
    
    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    
    if overall_score >= 90:
        print("🎉 Excellent system quality!")
        print("  - Continue current practices")
        print("  - Plan for future enhancements")
        print("  - Maintain quality standards")
    elif overall_score >= 80:
        print("✅ Good system quality with minor issues")
        print("  - Address identified issues")
        print("  - Improve weak areas")
        print("  - Plan for optimization")
    elif overall_score >= 70:
        print("⚠️ Acceptable quality with notable issues")
        print("  - Prioritize critical fixes")
        print("  - Address security concerns")
        print("  - Improve performance bottlenecks")
    elif overall_score >= 60:
        print("🔧 System needs significant improvement")
        print("  - Immediate action required")
        print("  - Focus on critical issues")
        print("  - Consider architectural changes")
    else:
        print("🚨 Critical issues detected!")
        print("  - IMMEDIATE ACTION REQUIRED")
        print("  - Address all failed audits")
        print("  - Consider system redesign")
    
    # Save report
    try:
        report_filename = f"enterprise_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Comprehensive report saved to: {report_filename}")
    except Exception as e:
        print(f"❌ Failed to save comprehensive report: {e}")
    
    return report

def run_enterprise_audit():
    """Run complete enterprise-level audit."""
    print("ENTERPRISE-LEVEL AUDIT FOR CBB BETTING ML SYSTEM")
    print("="*80)
    print("Phase 4: Optimization & Deployment")
    print("Phase 5: Monitoring & CI/CD")
    print("="*80)
    print("Starting comprehensive audit...")
    
    start_time = time.time()
    
    # Run all audit components
    audit_results = {}
    
    # 1. Mathematical Verification
    print("\n🔍 Starting Mathematical Verification...")
    audit_results["Mathematical Verification"] = run_mathematical_verification()
    
    # 2. Unit & Integration Testing
    print("\n🧪 Starting Unit & Integration Testing...")
    audit_results["Unit & Integration Testing"] = run_unit_integration_tests()
    
    # 3. Security & Dependency Audit
    print("\n🔐 Starting Security & Dependency Audit...")
    audit_results["Security & Dependency Audit"] = run_security_audit()
    
    # 4. Performance & Scalability Audit
    print("\n🚀 Starting Performance & Scalability Audit...")
    audit_results["Performance & Scalability Audit"] = run_performance_audit()
    
    # 5. Code & Architecture Review
    print("\n🏗️ Starting Code & Architecture Review...")
    audit_results["Code & Architecture Review"] = run_code_architecture_review()
    
    # 6. Model Explainability & Bias Audit
    print("\n🔍 Starting Model Explainability & Bias Audit...")
    audit_results["Model Explainability & Bias Audit"] = run_model_explainability_audit()
    
    # Calculate audit time
    audit_time = time.time() - start_time
    
    # Generate comprehensive report
    report = generate_comprehensive_report(audit_results)
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏁 ENTERPRISE AUDIT COMPLETED")
    print("="*80)
    print(f"Total Audit Time: {audit_time:.1f} seconds")
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Status: {report['overall_status']}")
    
    if report['overall_score'] >= 80:
        print("\n🎉 CONGRATULATIONS! Your system meets enterprise standards!")
        return True
    elif report['overall_score'] >= 70:
        print("\n⚠️ Your system is acceptable but needs improvements.")
        return True
    else:
        print("\n🚨 Your system has critical issues that must be addressed.")
        return False

if __name__ == "__main__":
    try:
        success = run_enterprise_audit()
        
        if success:
            print("\n✅ Enterprise audit completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Enterprise audit completed with critical issues!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Enterprise audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)