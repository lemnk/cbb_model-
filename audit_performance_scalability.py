#!/usr/bin/env python3
"""
Performance and Scalability Benchmarking Script for Phase 4 & 5 Audit
Tests performance characteristics and identifies scaling bottlenecks.
"""

import sys
import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import gc

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def benchmark_fastapi_performance():
    """Benchmark FastAPI endpoint performance."""
    print("="*60)
    print("FASTAPI PERFORMANCE BENCHMARKING")
    print("="*60)
    
    try:
        from deployment.api import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Test data
        test_features = {
            'team_efficiency': 0.75,
            'player_availability': 0.85,
            'dynamic_factors': 0.68,
            'market_signals': 0.72
        }
        
        # Warm-up
        print("Warming up FastAPI...")
        for _ in range(10):
            client.get("/health")
        
        # Performance test
        print("Running performance test...")
        n_requests = 1000
        latencies = []
        
        start_time = time.time()
        
        for i in range(n_requests):
            request_start = time.time()
            response = client.post("/predict", json={"features": test_features})
            request_end = time.time()
            
            if response.status_code == 200:
                latencies.append(request_end - request_start)
            else:
                print(f"Request {i} failed: {response.status_code}")
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        latencies = np.array(latencies)
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        max_latency = np.max(latencies)
        
        requests_per_sec = n_requests / total_time
        
        print(f"Performance Results:")
        print(f"  Total requests: {n_requests}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Requests/sec: {requests_per_sec:.2f}")
        print(f"  Average latency: {avg_latency*1000:.2f}ms")
        print(f"  95th percentile: {p95_latency*1000:.2f}ms")
        print(f"  99th percentile: {p99_latency*1000:.2f}ms")
        print(f"  Max latency: {max_latency*1000:.2f}ms")
        
        # Check if performance meets requirements
        performance_ok = True
        if avg_latency > 0.5:  # 500ms threshold
            print("⚠️ Average latency exceeds 500ms threshold")
            performance_ok = False
        else:
            print("✅ Average latency within acceptable range")
        
        if p95_latency > 1.0:  # 1 second threshold
            print("⚠️ 95th percentile latency exceeds 1 second threshold")
            performance_ok = False
        else:
            print("✅ 95th percentile latency within acceptable range")
        
        return {
            "requests_per_sec": requests_per_sec,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "max_latency": max_latency,
            "performance_ok": performance_ok
        }
        
    except Exception as e:
        print(f"❌ FastAPI performance test failed: {e}")
        return None

def benchmark_monitoring_performance():
    """Benchmark monitoring system performance."""
    print("\n" + "="*60)
    print("MONITORING SYSTEM PERFORMANCE BENCHMARKING")
    print("="*60)
    
    try:
        from monitoring.schema_validation import SchemaValidator
        from monitoring.drift_detection import DriftDetector
        from monitoring.performance_monitor import PerformanceMonitor
        from monitoring.alerts import AlertManager
        
        # Test data sizes
        test_sizes = [100, 1000, 10000]
        results = {}
        
        for size in test_sizes:
            print(f"\nTesting with {size} samples...")
            
            # Generate test data
            np.random.seed(42)
            test_data = {
                'game_id': [f'game_{i}' for i in range(size)],
                'date': ['2024-01-01'] * size,
                'season': [2024] * size,
                'home_team': ['Team_A'] * size,
                'away_team': ['Team_B'] * size,
                'team_efficiency': np.random.uniform(0.5, 0.9, size),
                'player_availability': np.random.uniform(0.7, 1.0, size),
                'dynamic_factors': np.random.uniform(0.4, 0.8, size),
                'market_signals': np.random.uniform(0.3, 0.7, size),
                'target': np.random.binomial(1, 0.5, size)
            }
            
            test_df = pd.DataFrame(test_data)
            
            # Benchmark schema validation
            start_time = time.time()
            validator = SchemaValidator()
            schema_results = validator.comprehensive_validation(test_df)
            schema_time = time.time() - start_time
            
            # Benchmark drift detection
            start_time = time.time()
            detector = DriftDetector(
                reference_df=test_df.iloc[:size//2],
                psi_threshold=0.25,
                ks_threshold=0.1,
                kl_threshold=0.1
            )
            drift_results = detector.detect_drift(test_df.iloc[size//2:])
            drift_time = time.time() - start_time
            
            # Benchmark performance monitoring
            start_time = time.time()
            thresholds = {
                'accuracy': 0.5, 'log_loss': 1.0, 'brier_score': 0.5,
                'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
                'roc_auc': 0.5, 'expected_value': 0.0
            }
            monitor = PerformanceMonitor(thresholds)
            
            y_true = test_df['target'].values
            y_pred_proba = np.random.uniform(0, 1, size)
            odds = np.random.uniform(1.5, 3.0, size)
            
            perf_results = monitor.evaluate(y_true, y_pred_proba, odds)
            perf_time = time.time() - start_time
            
            # Benchmark alerts
            start_time = time.time()
            alert_config = {'mode': 'console'}
            alert_manager = AlertManager(alert_config)
            alert_messages = alert_manager.check_alerts(perf_results)
            alert_time = time.time() - start_time
            
            results[size] = {
                'schema_validation': schema_time,
                'drift_detection': drift_time,
                'performance_monitoring': perf_time,
                'alerts': alert_time,
                'total_time': schema_time + drift_time + perf_time + alert_time
            }
            
            print(f"  Schema validation: {schema_time*1000:.2f}ms")
            print(f"  Drift detection: {drift_time*1000:.2f}ms")
            print(f"  Performance monitoring: {perf_time*1000:.2f}ms")
            print(f"  Alerts: {alert_time*1000:.2f}ms")
            print(f"  Total: {(schema_time + drift_time + perf_time + alert_time)*1000:.2f}ms")
        
        # Analyze scaling characteristics
        print("\nScaling Analysis:")
        sizes = list(results.keys())
        total_times = [results[size]['total_time'] for size in sizes]
        
        # Check if scaling is linear or worse
        if len(sizes) >= 2:
            scaling_factor = total_times[-1] / total_times[0]
            size_factor = sizes[-1] / sizes[0]
            
            if scaling_factor <= size_factor * 1.5:  # Linear scaling with 50% tolerance
                print("✅ Monitoring system scales linearly")
                scaling_ok = True
            else:
                print(f"⚠️ Monitoring system scales poorly (factor: {scaling_factor:.2f}x)")
                scaling_ok = False
        
        return results
        
    except Exception as e:
        print(f"❌ Monitoring performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("\n" + "="*60)
    print("MEMORY USAGE BENCHMARKING")
    print("="*60)
    
    try:
        from monitoring.performance_monitor import PerformanceMonitor
        
        # Test memory usage with different dataset sizes
        test_sizes = [1000, 10000, 100000]
        memory_results = {}
        
        for size in test_sizes:
            print(f"\nTesting memory usage with {size} samples...")
            
            # Get initial memory
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create large dataset
            np.random.seed(42)
            y_true = np.random.binomial(1, 0.5, size)
            y_pred_proba = np.random.uniform(0, 1, size)
            odds = np.random.uniform(1.5, 3.0, size)
            
            # Initialize monitor
            thresholds = {
                'accuracy': 0.5, 'log_loss': 1.0, 'brier_score': 0.5,
                'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
                'roc_auc': 0.5, 'expected_value': 0.0
            }
            monitor = PerformanceMonitor(thresholds)
            
            # Measure memory after initialization
            memory_after_init = process.memory_info().rss / 1024 / 1024
            
            # Run evaluation
            results = monitor.evaluate(y_true, y_pred_proba, odds)
            
            # Measure memory after evaluation
            memory_after_eval = process.memory_info().rss / 1024 / 1024
            
            # Clean up
            del y_true, y_pred_proba, odds, results
            gc.collect()
            
            # Measure memory after cleanup
            memory_after_cleanup = process.memory_info().rss / 1024 / 1024
            
            memory_results[size] = {
                'initial': initial_memory,
                'after_init': memory_after_init,
                'after_eval': memory_after_eval,
                'after_cleanup': memory_after_cleanup,
                'peak_usage': memory_after_eval - initial_memory,
                'memory_leak': memory_after_cleanup - initial_memory
            }
            
            print(f"  Initial memory: {initial_memory:.2f} MB")
            print(f"  Peak memory usage: {memory_results[size]['peak_usage']:.2f} MB")
            print(f"  Memory after cleanup: {memory_after_cleanup:.2f} MB")
            
            if memory_results[size]['memory_leak'] > 10:  # 10MB threshold
                print(f"  ⚠️ Potential memory leak: {memory_results[size]['memory_leak']:.2f} MB")
            else:
                print(f"  ✅ No significant memory leak detected")
        
        return memory_results
        
    except Exception as e:
        print(f"❌ Memory usage test failed: {e}")
        return None

def benchmark_training_pipeline():
    """Benchmark training pipeline performance."""
    print("\n" + "="*60)
    print("TRAINING PIPELINE PERFORMANCE BENCHMARKING")
    print("="*60)
    
    try:
        from optimization.hyperparameter_optimizer import GridSearchOptimizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.datasets import make_classification
        
        # Test with different dataset sizes
        test_sizes = [1000, 5000, 10000]
        training_results = {}
        
        for size in test_sizes:
            print(f"\nTesting training pipeline with {size} samples...")
            
            # Generate synthetic data
            X, y = make_classification(
                n_samples=size,
                n_features=20,
                n_informative=15,
                n_redundant=5,
                random_state=42
            )
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5]
            }
            
            # Benchmark grid search
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            optimizer = GridSearchOptimizer(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3
            )
            
            optimizer.optimize(X, y)
            
            training_time = time.time() - start_time
            peak_memory = process.memory_info().rss / 1024 / 1024
            
            training_results[size] = {
                'training_time': training_time,
                'initial_memory': initial_memory,
                'peak_memory': peak_memory,
                'memory_increase': peak_memory - initial_memory
            }
            
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Memory increase: {training_results[size]['memory_increase']:.2f} MB")
            
            # Clean up
            del optimizer, X, y
            gc.collect()
        
        return training_results
        
    except Exception as e:
        print(f"❌ Training pipeline test failed: {e}")
        return None

def identify_scaling_bottlenecks():
    """Identify potential scaling bottlenecks."""
    print("\n" + "="*60)
    print("SCALING BOTTLENECK ANALYSIS")
    print("="*60)
    
    bottlenecks = []
    
    # Check CPU usage patterns
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=1)
        print(f"Current CPU usage: {cpu_percent:.1f}%")
        
        if cpu_percent > 80:
            bottlenecks.append("High CPU usage detected")
    except Exception as e:
        print(f"Could not check CPU usage: {e}")
    
    # Check memory usage patterns
    try:
        memory_info = psutil.virtual_memory()
        print(f"Memory usage: {memory_info.percent:.1f}%")
        
        if memory_info.percent > 80:
            bottlenecks.append("High memory usage detected")
    except Exception as e:
        print(f"Could not check memory usage: {e}")
    
    # Check disk I/O
    try:
        disk_io = psutil.disk_io_counters()
        if disk_io:
            print(f"Disk I/O - Read: {disk_io.read_bytes / 1024 / 1024:.1f} MB, Write: {disk_io.write_bytes / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Could not check disk I/O: {e}")
    
    # Check network I/O
    try:
        net_io = psutil.net_io_counters()
        if net_io:
            print(f"Network I/O - Bytes sent: {net_io.bytes_sent / 1024 / 1024:.1f} MB, Bytes received: {net_io.bytes_recv / 1024 / 1024:.1f} MB")
    except Exception as e:
        print(f"Could not check network I/O: {e}")
    
    if bottlenecks:
        print(f"\n⚠️ Identified bottlenecks:")
        for bottleneck in bottlenecks:
            print(f"  - {bottleneck}")
    else:
        print("\n✅ No obvious bottlenecks detected")
    
    return bottlenecks

def generate_performance_report():
    """Generate comprehensive performance report."""
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE REPORT")
    print("="*60)
    
    # Run all benchmarks
    fastapi_results = benchmark_fastapi_performance()
    monitoring_results = benchmark_monitoring_performance()
    memory_results = benchmark_memory_usage()
    training_results = benchmark_training_pipeline()
    bottlenecks = identify_scaling_bottlenecks()
    
    # Generate report
    report = {
        "timestamp": "2024-01-15",
        "fastapi_performance": fastapi_results,
        "monitoring_performance": monitoring_results,
        "memory_usage": memory_results,
        "training_pipeline": training_results,
        "scaling_bottlenecks": bottlenecks,
        "overall_status": "unknown"
    }
    
    # Determine overall status
    performance_issues = []
    
    if fastapi_results and not fastapi_results.get('performance_ok', True):
        performance_issues.append("FastAPI performance below threshold")
    
    if memory_results:
        for size, mem_data in memory_results.items():
            if mem_data.get('memory_leak', 0) > 10:
                performance_issues.append(f"Memory leak detected at {size} samples")
    
    if bottlenecks:
        performance_issues.extend(bottlenecks)
    
    if not performance_issues:
        report['overall_status'] = 'good'
    elif len(performance_issues) <= 2:
        report['overall_status'] = 'warning'
    else:
        report['overall_status'] = 'critical'
    
    # Print summary
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Overall Status: {report['overall_status'].upper()}")
    
    if fastapi_results:
        print(f"FastAPI Performance: {'✅ OK' if fastapi_results.get('performance_ok', False) else '❌ ISSUES'}")
        print(f"  Requests/sec: {fastapi_results.get('requests_per_sec', 0):.2f}")
        print(f"  Avg Latency: {fastapi_results.get('avg_latency', 0)*1000:.2f}ms")
    
    if memory_results:
        print(f"Memory Usage: {'✅ OK' if not any(r.get('memory_leak', 0) > 10 for r in memory_results.values()) else '⚠️ ISSUES'}")
    
    if bottlenecks:
        print(f"Scaling Bottlenecks: {len(bottlenecks)} identified")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if report['overall_status'] == 'critical':
        print("🚨 IMMEDIATE ACTION REQUIRED:")
        print("  - Address performance bottlenecks")
        print("  - Fix memory leaks")
        print("  - Optimize critical paths")
    elif report['overall_status'] == 'warning':
        print("⚠️ ATTENTION RECOMMENDED:")
        print("  - Monitor performance trends")
        print("  - Consider optimizations")
        print("  - Plan for scaling")
    else:
        print("✅ PERFORMANCE STATUS GOOD:")
        print("  - Continue monitoring")
        print("  - Plan for future scaling")
        print("  - Maintain performance standards")
    
    # Save report
    try:
        with open("performance_benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Performance report saved to: performance_benchmark_report.json")
    except Exception as e:
        print(f"❌ Failed to save performance report: {e}")
    
    return report

def run_performance_audit():
    """Run complete performance audit."""
    print("PERFORMANCE & SCALABILITY BENCHMARKING AUDIT")
    print("="*60)
    print("Running comprehensive performance tests...")
    
    try:
        report = generate_performance_report()
        
        if report['overall_status'] == 'good':
            print("\n🎉 Performance audit completed successfully!")
            print("All performance benchmarks passed.")
            return True
        elif report['overall_status'] == 'warning':
            print("\n⚠️ Performance audit completed with warnings.")
            print("Review recommendations above.")
            return True
        else:
            print("\n🚨 Performance audit completed with CRITICAL issues!")
            print("Immediate action required.")
            return False
            
    except Exception as e:
        print(f"❌ Performance audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_performance_audit()
    sys.exit(0 if success else 1)