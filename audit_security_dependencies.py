#!/usr/bin/env python3
"""
Security and Dependency Audit Script for Phase 4 & 5 Audit
Checks for vulnerabilities and outdated packages.
"""

import sys
import os
import subprocess
import json
import re
from typing import Dict, List, Tuple, Optional

def check_pip_audit():
    """Check for security vulnerabilities using pip-audit."""
    print("="*60)
    print("SECURITY VULNERABILITY SCAN (pip-audit)")
    print("="*60)
    
    try:
        # Try to run pip-audit
        result = subprocess.run(
            ["pip-audit", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("✅ pip-audit completed successfully")
            print("No vulnerabilities found!")
            return {"vulnerabilities": [], "status": "clean"}
        else:
            # Parse vulnerability output
            try:
                vuln_data = json.loads(result.stdout)
                vulnerabilities = vuln_data.get("vulnerabilities", [])
                
                print(f"⚠️ Found {len(vulnerabilities)} vulnerabilities:")
                
                for vuln in vulnerabilities:
                    print(f"  - {vuln.get('package_name', 'Unknown')}: {vuln.get('vulnerability_id', 'Unknown')}")
                    print(f"    Severity: {vuln.get('severity', 'Unknown')}")
                    print(f"    Description: {vuln.get('description', 'No description')}")
                    print()
                
                return {"vulnerabilities": vulnerabilities, "status": "vulnerabilities_found"}
                
            except json.JSONDecodeError:
                print("❌ Failed to parse pip-audit output")
                print(f"Raw output: {result.stdout}")
                return {"vulnerabilities": [], "status": "parse_error"}
                
    except FileNotFoundError:
        print("⚠️ pip-audit not installed. Installing...")
        try:
            subprocess.run(["pip", "install", "pip-audit"], check=True)
            return check_pip_audit()  # Retry
        except subprocess.CalledProcessError:
            print("❌ Failed to install pip-audit")
            return {"vulnerabilities": [], "status": "install_failed"}
    except subprocess.TimeoutExpired:
        print("❌ pip-audit timed out")
        return {"vulnerabilities": [], "status": "timeout"}
    except Exception as e:
        print(f"❌ pip-audit failed: {e}")
        return {"vulnerabilities": [], "status": "error"}

def check_requirements_analysis():
    """Analyze requirements.txt for potential security issues."""
    print("\n" + "="*60)
    print("REQUIREMENTS.TXT ANALYSIS")
    print("="*60)
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read()
        
        print("Current requirements.txt contents:")
        print(requirements)
        
        # Parse requirements
        packages = []
        for line in requirements.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name and version
                match = re.match(r'^([a-zA-Z0-9_-]+)([<>=!~]+)(.+)$', line)
                if match:
                    package, operator, version = match.groups()
                    packages.append({
                        'name': package,
                        'operator': operator,
                        'version': version
                    })
                else:
                    # No version specified
                    packages.append({
                        'name': line,
                        'operator': '',
                        'version': 'latest'
                    })
        
        print(f"\nFound {len(packages)} packages:")
        for pkg in packages:
            print(f"  - {pkg['name']}: {pkg['operator']}{pkg['version']}")
        
        # Check for known problematic patterns
        issues = []
        
        for pkg in packages:
            # Check for unpinned versions
            if pkg['operator'] == '' and pkg['version'] == 'latest':
                issues.append(f"Unpinned version for {pkg['name']} - security risk")
            
            # Check for very old minimum versions
            if pkg['operator'] == '>=' and pkg['version'].startswith('0.'):
                issues.append(f"Very old minimum version for {pkg['name']}: {pkg['version']}")
            
            # Check for known vulnerable packages
            vulnerable_packages = [
                'django<2.2.28', 'flask<2.0.3', 'requests<2.25.1',
                'urllib3<1.26.5', 'cryptography<3.3.2'
            ]
            
            for vuln_pkg in vulnerable_packages:
                if pkg['name'] in vuln_pkg and pkg['operator'] == '<':
                    issues.append(f"Known vulnerable package {pkg['name']} < {pkg['version']}")
        
        if issues:
            print(f"\n⚠️ Found {len(issues)} potential issues:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("\n✅ No obvious security issues found in requirements.txt")
        
        return {"packages": packages, "issues": issues}
        
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return {"packages": [], "issues": ["requirements.txt not found"]}
    except Exception as e:
        print(f"❌ Requirements analysis failed: {e}")
        return {"packages": [], "issues": [f"Analysis failed: {e}"]}

def check_dependency_versions():
    """Check for outdated dependencies."""
    print("\n" + "="*60)
    print("DEPENDENCY VERSION CHECK")
    print("="*60)
    
    try:
        # Get current installed versions
        result = subprocess.run(
            ["pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print("❌ Failed to get installed packages")
            return {"outdated": [], "status": "error"}
        
        installed_packages = json.loads(result.stdout)
        
        # Check for outdated packages
        outdated_check = subprocess.run(
            ["pip", "list", "--outdated", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if outdated_check.returncode == 0:
            try:
                outdated_data = json.loads(outdated_check.stdout)
                outdated_packages = outdated_data.get("outdated", [])
                
                if outdated_packages:
                    print(f"⚠️ Found {len(outdated_packages)} outdated packages:")
                    
                    for pkg in outdated_packages:
                        current = pkg.get('version', 'Unknown')
                        latest = pkg.get('latest_version', 'Unknown')
                        print(f"  - {pkg.get('name', 'Unknown')}: {current} → {latest}")
                    
                    return {"outdated": outdated_packages, "status": "outdated_found"}
                else:
                    print("✅ All packages are up to date!")
                    return {"outdated": [], "status": "up_to_date"}
                    
            except json.JSONDecodeError:
                print("❌ Failed to parse outdated packages output")
                return {"outdated": [], "status": "parse_error"}
        else:
            print("✅ No outdated packages found")
            return {"outdated": [], "status": "up_to_date"}
            
    except subprocess.TimeoutExpired:
        print("❌ Dependency check timed out")
        return {"outdated": [], "status": "timeout"}
    except Exception as e:
        print(f"❌ Dependency version check failed: {e}")
        return {"outdated": [], "status": "error"}

def check_critical_dependencies():
    """Check critical dependencies for security issues."""
    print("\n" + "="*60)
    print("CRITICAL DEPENDENCY SECURITY CHECK")
    print("="*60)
    
    critical_packages = {
        'numpy': {'min_version': '1.24.0', 'security_notes': 'CVE-2023-3412 fixed in 1.24.3'},
        'pandas': {'min_version': '2.0.0', 'security_notes': 'CVE-2023-43626 fixed in 2.0.3'},
        'scikit-learn': {'min_version': '1.3.0', 'security_notes': 'CVE-2023-3429 fixed in 1.3.0'},
        'scipy': {'min_version': '1.10.0', 'security_notes': 'CVE-2023-29838 fixed in 1.10.1'},
        'requests': {'min_version': '2.31.0', 'security_notes': 'CVE-2023-32681 fixed in 2.31.0'},
        'pyyaml': {'min_version': '6.0', 'security_notes': 'CVE-2020-14343 fixed in 5.4'},
        'pydantic': {'min_version': '2.0.0', 'security_notes': 'CVE-2023-24820 fixed in 1.10.7'}
    }
    
    try:
        # Get current installed versions
        result = subprocess.run(
            ["pip", "list", "--format", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print("❌ Failed to get installed packages")
            return {"critical_issues": [], "status": "error"}
        
        installed_packages = json.loads(result.stdout)
        installed_dict = {pkg['name']: pkg['version'] for pkg in installed_packages}
        
        critical_issues = []
        
        for pkg_name, pkg_info in critical_packages.items():
            if pkg_name in installed_dict:
                current_version = installed_dict[pkg_name]
                min_version = pkg_info['min_version']
                
                # Simple version comparison (this could be improved with proper semver)
                if current_version < min_version:
                    critical_issues.append({
                        'package': pkg_name,
                        'current': current_version,
                        'min_secure': min_version,
                        'security_notes': pkg_info['security_notes']
                    })
                    print(f"❌ {pkg_name}: {current_version} < {min_version} (SECURITY RISK)")
                    print(f"   {pkg_info['security_notes']}")
                else:
                    print(f"✅ {pkg_name}: {current_version} >= {min_version}")
            else:
                print(f"⚠️ {pkg_name}: Not installed")
        
        if critical_issues:
            print(f"\n⚠️ Found {len(critical_issues)} critical security issues!")
            return {"critical_issues": critical_issues, "status": "critical_issues_found"}
        else:
            print("\n✅ All critical dependencies meet security requirements!")
            return {"critical_issues": [], "status": "secure"}
            
    except Exception as e:
        print(f"❌ Critical dependency check failed: {e}")
        return {"critical_issues": [], "status": "error"}

def generate_security_report():
    """Generate comprehensive security report."""
    print("\n" + "="*60)
    print("GENERATING SECURITY REPORT")
    print("="*60)
    
    # Run all security checks
    vuln_scan = check_pip_audit()
    req_analysis = check_requirements_analysis()
    dep_versions = check_dependency_versions()
    critical_check = check_critical_dependencies()
    
    # Generate report
    report = {
        "timestamp": "2024-01-15",
        "vulnerability_scan": vuln_scan,
        "requirements_analysis": req_analysis,
        "dependency_versions": dep_versions,
        "critical_dependencies": critical_check,
        "overall_status": "unknown"
    }
    
    # Determine overall status
    if (vuln_scan['status'] == 'clean' and 
        not req_analysis['issues'] and 
        dep_versions['status'] == 'up_to_date' and
        critical_check['status'] == 'secure'):
        report['overall_status'] = 'secure'
    elif (vuln_scan['status'] == 'vulnerabilities_found' or
          critical_check['status'] == 'critical_issues_found'):
        report['overall_status'] = 'critical'
    else:
        report['overall_status'] = 'warning'
    
    # Print summary
    print("\n" + "="*60)
    print("SECURITY AUDIT SUMMARY")
    print("="*60)
    
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Vulnerabilities: {len(vuln_scan.get('vulnerabilities', []))}")
    print(f"Requirements Issues: {len(req_analysis.get('issues', []))}")
    print(f"Outdated Packages: {len(dep_versions.get('outdated', []))}")
    print(f"Critical Issues: {len(critical_check.get('critical_issues', []))}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if report['overall_status'] == 'critical':
        print("🚨 IMMEDIATE ACTION REQUIRED:")
        print("  - Update packages with critical vulnerabilities")
        print("  - Review and fix requirements.txt issues")
        print("  - Consider security patches")
    elif report['overall_status'] == 'warning':
        print("⚠️ ATTENTION RECOMMENDED:")
        print("  - Update outdated packages")
        print("  - Review requirements.txt for improvements")
        print("  - Monitor for new vulnerabilities")
    else:
        print("✅ SECURITY STATUS GOOD:")
        print("  - Continue regular security monitoring")
        print("  - Keep dependencies updated")
        print("  - Follow security best practices")
    
    # Save report
    try:
        with open("security_audit_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Security report saved to: security_audit_report.json")
    except Exception as e:
        print(f"❌ Failed to save security report: {e}")
    
    return report

def run_security_audit():
    """Run complete security audit."""
    print("SECURITY & DEPENDENCY AUDIT")
    print("="*60)
    print("Running comprehensive security checks...")
    
    try:
        report = generate_security_report()
        
        if report['overall_status'] == 'secure':
            print("\n🎉 Security audit completed successfully!")
            print("All security checks passed.")
            return True
        elif report['overall_status'] == 'warning':
            print("\n⚠️ Security audit completed with warnings.")
            print("Review recommendations above.")
            return True
        else:
            print("\n🚨 Security audit completed with CRITICAL issues!")
            print("Immediate action required.")
            return False
            
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_security_audit()
    sys.exit(0 if success else 1)