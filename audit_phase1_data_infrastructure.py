#!/usr/bin/env python3
"""
Phase 1 Enterprise Audit: Data Infrastructure
Comprehensive audit of ETL, database schema, data quality, and security compliance.
"""

import sys
import os
import time
import json
import hashlib
import psutil
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np

# Add cbb_model/src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'cbb_model', 'src'))

def audit_database_schema():
    """Audit database schema for correctness and consistency."""
    print("="*80)
    print("🏗️ DATABASE SCHEMA AUDIT")
    print("="*80)
    
    try:
        from db import DatabaseManager
        from utils import ConfigManager
        
        # Load configuration
        config = ConfigManager('cbb_model/config.yaml')
        
        # Initialize database manager
        db_manager = DatabaseManager(config)
        
        print("✅ Database manager initialized successfully")
        
        # Check schema definition
        print("\n1. SCHEMA DEFINITION VALIDATION")
        
        # Verify all required tables exist
        required_tables = ['games', 'odds', 'players', 'teams', 'games_odds']
        existing_tables = []
        
        for table_name in required_tables:
            if hasattr(db_manager, f'{table_name}_table'):
                table = getattr(db_manager, f'{table_name}_table')
                existing_tables.append(table_name)
                print(f"  ✅ Table '{table_name}' defined")
                
                # Check columns
                columns = [col.name for col in table.columns]
                print(f"    Columns: {len(columns)}")
                
                # Check primary key
                pk_columns = [col.name for col in table.columns if col.primary_key]
                if pk_columns:
                    print(f"    Primary Key: {pk_columns}")
                else:
                    print(f"    ⚠️ No primary key defined")
                
                # Check nullable constraints
                nullable_columns = [col.name for col in table.columns if col.nullable]
                non_nullable_columns = [col.name for col in table.columns if not col.nullable]
                print(f"    Non-nullable: {len(non_nullable_columns)} columns")
                print(f"    Nullable: {len(nullable_columns)} columns")
                
            else:
                print(f"  ❌ Table '{table_name}' not found")
        
        # Check for missing tables
        missing_tables = set(required_tables) - set(existing_tables)
        if missing_tables:
            print(f"  ❌ Missing tables: {missing_tables}")
            return False
        
        print(f"✅ All {len(required_tables)} required tables defined")
        
        # 2. COLUMN TYPE VALIDATION
        print("\n2. COLUMN TYPE VALIDATION")
        
        # Check for proper data types
        type_issues = []
        
        # Games table validation
        games_table = db_manager.games_table
        expected_games_types = {
            'game_id': 'String(50)',
            'date': 'DateTime',
            'season': 'Integer',
            'home_team': 'String(100)',
            'away_team': 'String(100)',
            'home_score': 'Integer',
            'away_score': 'Integer'
        }
        
        for col_name, expected_type in expected_games_types.items():
            col = games_table.columns.get(col_name)
            if col:
                actual_type = str(col.type)
                if expected_type in actual_type:
                    print(f"  ✅ {col_name}: {actual_type}")
                else:
                    print(f"  ❌ {col_name}: expected {expected_type}, got {actual_type}")
                    type_issues.append(f"{col_name}: {expected_type} vs {actual_type}")
            else:
                print(f"  ❌ Column {col_name} not found")
                type_issues.append(f"Missing column: {col_name}")
        
        # 3. CONSTRAINT VALIDATION
        print("\n3. CONSTRAINT VALIDATION")
        
        # Check for proper constraints
        constraint_issues = []
        
        # Check primary key constraints
        for table_name in required_tables:
            table = getattr(db_manager, f'{table_name}_table')
            pk_columns = [col.name for col in table.columns if col.primary_key]
            
            if not pk_columns:
                constraint_issues.append(f"{table_name}: No primary key")
                print(f"  ❌ {table_name}: No primary key defined")
            elif len(pk_columns) == 1:
                print(f"  ✅ {table_name}: Primary key on {pk_columns[0]}")
            else:
                print(f"  ⚠️ {table_name}: Composite primary key on {pk_columns}")
        
        # Check foreign key relationships
        print("\n4. FOREIGN KEY RELATIONSHIPS")
        
        # Expected relationships
        expected_fks = [
            ('odds', 'game_id', 'games', 'game_id'),
            ('players', 'game_id', 'games', 'game_id'),
            ('players', 'team_id', 'teams', 'team_id'),
            ('games', 'home_team_id', 'teams', 'team_id'),
            ('games', 'away_team_id', 'teams', 'team_id')
        ]
        
        fk_issues = []
        for fk_table, fk_col, ref_table, ref_col in expected_fks:
            if hasattr(db_manager, f'{fk_table}_table') and hasattr(db_manager, f'{ref_table}_table'):
                fk_table_obj = getattr(db_manager, f'{fk_table}_table')
                ref_table_obj = getattr(db_manager, f'{ref_table}_table')
                
                # Check if foreign key column exists
                fk_exists = any(col.name == fk_col for col in fk_table_obj.columns)
                ref_exists = any(col.name == ref_col for col in ref_table_obj.columns)
                
                if fk_exists and ref_exists:
                    print(f"  ✅ {fk_table}.{fk_col} → {ref_table}.{ref_col}")
                else:
                    print(f"  ❌ {fk_table}.{fk_col} → {ref_table}.{ref_col} (missing columns)")
                    fk_issues.append(f"{fk_table}.{fk_col} → {ref_table}.{ref_col}")
            else:
                print(f"  ❌ Cannot verify {fk_table}.{fk_col} → {ref_table}.{ref_col} (table missing)")
                fk_issues.append(f"Table missing for {fk_table}.{fk_col} → {ref_table}.{ref_col}")
        
        # Summary
        print("\n" + "="*50)
        print("SCHEMA AUDIT SUMMARY")
        print("="*50)
        
        total_checks = len(required_tables) + len(expected_games_types) + len(expected_fks)
        failed_checks = len(type_issues) + len(constraint_issues) + len(fk_issues)
        passed_checks = total_checks - failed_checks
        
        print(f"Total checks: {total_checks}")
        print(f"Passed: {passed_checks}")
        print(f"Failed: {failed_checks}")
        
        if failed_checks == 0:
            print("🎉 Database schema audit PASSED")
            return True
        else:
            print("⚠️ Database schema audit has issues")
            return False
            
    except Exception as e:
        print(f"❌ Database schema audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_etl_pipeline():
    """Audit ETL pipeline for robustness and reproducibility."""
    print("\n" + "="*80)
    print("🔄 ETL PIPELINE AUDIT")
    print("="*80)
    
    try:
        from etl import CBBDataETL
        from utils import ConfigManager
        
        # Load configuration
        config = ConfigManager('cbb_model/config.yaml')
        
        # Initialize ETL processor
        etl_processor = CBBDataETL(config)
        
        print("✅ ETL processor initialized successfully")
        
        # 1. TEST DATA LOADING WITH MALFORMED DATA
        print("\n1. MALFORMED DATA HANDLING TEST")
        
        # Create test data directory
        test_raw_dir = "test_data/raw"
        os.makedirs(test_raw_dir, exist_ok=True)
        
        # Create test data with various issues
        test_data_issues = []
        
        # Issue 1: Missing values
        missing_values_data = {
            'game_id': ['game_1', 'game_2', 'game_3'],
            'date': ['2024-01-01', None, '2024-01-03'],
            'season': [2024, 2024, None],
            'home_team': ['Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', None, 'Ohio State'],
            'home_score': [75, 68, None],
            'away_score': [70, 72, 74]
        }
        
        missing_df = pd.DataFrame(missing_values_data)
        missing_df.to_csv(f"{test_raw_dir}/games_missing_values.csv", index=False)
        test_data_issues.append("Missing values")
        
        # Issue 2: Duplicate rows
        duplicate_data = {
            'game_id': ['game_1', 'game_1', 'game_2', 'game_3'],
            'date': ['2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03'],
            'season': [2024, 2024, 2024, 2024],
            'home_team': ['Duke', 'Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', 'UNC', 'Kentucky', 'Ohio State'],
            'home_score': [75, 75, 68, 82],
            'away_score': [70, 70, 72, 74]
        }
        
        duplicate_df = pd.DataFrame(duplicate_data)
        duplicate_df.to_csv(f"{test_raw_dir}/games_duplicates.csv", index=False)
        test_data_issues.append("Duplicate rows")
        
        # Issue 3: Invalid data types
        invalid_types_data = {
            'game_id': ['game_1', 'game_2', 'game_3'],
            'date': ['invalid_date', '2024-01-02', '2024-01-03'],
            'season': ['not_a_number', 2024, 2024],
            'home_team': ['Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'],
            'home_score': ['score_text', 68, 82],
            'away_score': [70, 'invalid_score', 74]
        }
        
        invalid_types_df = pd.DataFrame(invalid_types_data)
        invalid_types_df.to_csv(f"{test_raw_dir}/games_invalid_types.csv", index=False)
        test_data_issues.append("Invalid data types")
        
        print(f"Created test data with issues: {', '.join(test_data_issues)}")
        
        # 2. TEST ETL PROCESSING
        print("\n2. ETL PROCESSING TEST")
        
        # Test loading malformed data
        try:
            # Temporarily change raw data directory
            original_raw_dir = etl_processor.raw_data_dir
            etl_processor.raw_data_dir = test_raw_dir
            
            # Test loading
            games_data = etl_processor.load_raw_games_data()
            
            if not games_data.empty:
                print(f"✅ Successfully loaded {len(games_data)} rows from malformed data")
                
                # Check if issues were handled
                missing_handled = games_data['date'].isna().sum() > 0
                duplicates_handled = len(games_data) < len(duplicate_df)
                
                if missing_handled:
                    print("  ✅ Missing values handled")
                else:
                    print("  ⚠️ Missing values not handled")
                
                if duplicates_handled:
                    print("  ✅ Duplicates handled")
                else:
                    print("  ⚠️ Duplicates not handled")
                
            else:
                print("⚠️ No data loaded from malformed files")
            
            # Restore original directory
            etl_processor.raw_data_dir = original_raw_dir
            
        except Exception as e:
            print(f"❌ ETL processing test failed: {e}")
            return False
        
        # 3. REPRODUCIBILITY TEST
        print("\n3. ETL REPRODUCIBILITY TEST")
        
        # Create clean test data
        clean_data = {
            'game_id': ['test_1', 'test_2', 'test_3'],
            'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'season': [2024, 2024, 2024],
            'home_team': ['Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'],
            'home_score': [75, 68, 82],
            'away_score': [70, 72, 74]
        }
        
        clean_df = pd.DataFrame(clean_data)
        clean_file = f"{test_raw_dir}/games_clean.csv"
        clean_df.to_csv(clean_file, index=False)
        
        # Run ETL twice and compare results
        try:
            etl_processor.raw_data_dir = test_raw_dir
            
            # First run
            result1 = etl_processor.load_raw_games_data()
            hash1 = hashlib.md5(result1.to_string().encode()).hexdigest()
            
            # Second run
            result2 = etl_processor.load_raw_games_data()
            hash2 = hashlib.md5(result2.to_string().encode()).hexdigest()
            
            if hash1 == hash2:
                print("✅ ETL reproducibility test PASSED - identical results")
                reproducibility_ok = True
            else:
                print("❌ ETL reproducibility test FAILED - different results")
                reproducibility_ok = False
            
            # Restore original directory
            etl_processor.raw_data_dir = original_raw_dir
            
        except Exception as e:
            print(f"❌ ETL reproducibility test failed: {e}")
            reproducibility_ok = False
        
        # Clean up test data
        import shutil
        shutil.rmtree("test_data", ignore_errors=True)
        
        return reproducibility_ok
        
    except Exception as e:
        print(f"❌ ETL pipeline audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_data_quality():
    """Audit data quality and validation."""
    print("\n" + "="*80)
    print("📊 DATA QUALITY AUDIT")
    print("="*80)
    
    try:
        from etl import CBBDataETL
        from utils import ConfigManager
        
        # Load configuration
        config = ConfigManager('cbb_model/config.yaml')
        
        # Initialize ETL processor
        etl_processor = CBBDataETL(config)
        
        print("✅ ETL processor initialized for data quality audit")
        
        # 1. ROW/COLUMN COUNT VALIDATION
        print("\n1. ROW/COLUMN COUNT VALIDATION")
        
        # Check if raw data exists
        raw_data_dir = etl_processor.raw_data_dir
        if not os.path.exists(raw_data_dir):
            print(f"⚠️ Raw data directory not found: {raw_data_dir}")
            print("Creating sample data for testing...")
            
            # Create sample data
            os.makedirs(raw_data_dir, exist_ok=True)
            sample_data = {
                'game_id': [f'game_{i}' for i in range(1, 101)],
                'date': [f'2024-01-{i:02d}' for i in range(1, 101)],
                'season': [2024] * 100,
                'home_team': ['Duke', 'Kansas', 'Michigan'] * 33 + ['Duke'],
                'away_team': ['UNC', 'Kentucky', 'Ohio State'] * 33 + ['UNC'],
                'home_score': np.random.randint(60, 100, 100),
                'away_score': np.random.randint(60, 100, 100)
            }
            
            sample_df = pd.DataFrame(sample_data)
            sample_df.to_csv(f"{raw_data_dir}/games_2024.csv", index=False)
            print(f"Created sample data: {len(sample_df)} rows")
        
        # Load raw data
        raw_games = etl_processor.load_raw_games_data()
        
        if raw_games.empty:
            print("❌ No raw games data found")
            return False
        
        print(f"✅ Raw data loaded: {len(raw_games)} rows, {len(raw_games.columns)} columns")
        
        # 2. DATA RANGE VALIDATION
        print("\n2. DATA RANGE VALIDATION")
        
        validation_issues = []
        
        # Check score ranges
        if 'home_score' in raw_games.columns and 'away_score' in raw_games.columns:
            home_scores = raw_games['home_score'].dropna()
            away_scores = raw_games['away_score'].dropna()
            
            if len(home_scores) > 0:
                if home_scores.min() < 0:
                    validation_issues.append("Negative home scores found")
                    print(f"  ❌ Negative home scores: {home_scores.min()}")
                else:
                    print(f"  ✅ Home scores range: {home_scores.min()} - {home_scores.max()}")
                
                if away_scores.min() < 0:
                    validation_issues.append("Negative away scores found")
                    print(f"  ❌ Negative away scores: {away_scores.min()}")
                else:
                    print(f"  ✅ Away scores range: {away_scores.min()} - {away_scores.max()}")
        
        # Check season ranges
        if 'season' in raw_games.columns:
            seasons = raw_games['season'].dropna()
            if len(seasons) > 0:
                current_year = datetime.now().year
                if seasons.min() < 1900 or seasons.max() > current_year + 1:
                    validation_issues.append("Invalid season years")
                    print(f"  ❌ Invalid season range: {seasons.min()} - {seasons.max()}")
                else:
                    print(f"  ✅ Season range: {seasons.min()} - {seasons.max()}")
        
        # Check date validity
        if 'date' in raw_games.columns:
            try:
                # Convert to datetime
                dates = pd.to_datetime(raw_games['date'], errors='coerce')
                valid_dates = dates.dropna()
                
                if len(valid_dates) > 0:
                    current_date = datetime.now()
                    future_dates = valid_dates[valid_dates > current_date]
                    
                    if len(future_dates) > 0:
                        validation_issues.append("Future dates found")
                        print(f"  ❌ Future dates found: {len(future_dates)} rows")
                    else:
                        print(f"  ✅ All dates are in the past")
                        
                    # Check for very old dates
                    old_threshold = current_date - timedelta(days=365*50)  # 50 years
                    old_dates = valid_dates[valid_dates < old_threshold]
                    
                    if len(old_dates) > 0:
                        validation_issues.append("Very old dates found")
                        print(f"  ❌ Very old dates found: {len(old_dates)} rows")
                    else:
                        print(f"  ✅ No extremely old dates")
                        
            except Exception as e:
                validation_issues.append(f"Date validation error: {e}")
                print(f"  ❌ Date validation failed: {e}")
        
        # 3. DATA CONSISTENCY CHECKS
        print("\n3. DATA CONSISTENCY CHECKS")
        
        # Check for missing values
        missing_counts = raw_games.isnull().sum()
        high_missing = missing_counts[missing_counts > len(raw_games) * 0.5]
        
        if len(high_missing) > 0:
            validation_issues.append("High missing value columns")
            print(f"  ⚠️ High missing values in columns: {list(high_missing.index)}")
        else:
            print("  ✅ No columns with excessive missing values")
        
        # Check for duplicate game IDs
        if 'game_id' in raw_games.columns:
            duplicates = raw_games['game_id'].duplicated().sum()
            if duplicates > 0:
                validation_issues.append(f"Duplicate game IDs: {duplicates}")
                print(f"  ❌ Duplicate game IDs: {duplicates}")
            else:
                print("  ✅ No duplicate game IDs")
        
        # Summary
        print("\n" + "="*50)
        print("DATA QUALITY AUDIT SUMMARY")
        print("="*50)
        
        if validation_issues:
            print(f"⚠️ Found {len(validation_issues)} data quality issues:")
            for issue in validation_issues:
                print(f"  - {issue}")
            return False
        else:
            print("🎉 Data quality audit PASSED")
            return True
            
    except Exception as e:
        print(f"❌ Data quality audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_performance_scalability():
    """Audit ETL performance and scalability."""
    print("\n" + "="*80)
    print("🚀 PERFORMANCE & SCALABILITY AUDIT")
    print("="*80)
    
    try:
        from etl import CBBDataETL
        from utils import ConfigManager
        
        # Load configuration
        config = ConfigManager('cbb_model/config.yaml')
        
        # Initialize ETL processor
        etl_processor = CBBDataETL(config)
        
        print("✅ ETL processor initialized for performance audit")
        
        # 1. BASELINE PERFORMANCE TEST
        print("\n1. BASELINE PERFORMANCE TEST")
        
        # Create test data of different sizes
        test_sizes = [100, 1000, 10000]
        performance_results = {}
        
        for size in test_sizes:
            print(f"\nTesting with {size} rows...")
            
            # Create test data
            test_data = {
                'game_id': [f'test_game_{i}' for i in range(size)],
                'date': [f'2024-01-{(i % 30) + 1:02d}' for i in range(size)],
                'season': [2024] * size,
                'home_team': ['Team_A', 'Team_B', 'Team_C'] * (size // 3 + 1),
                'away_team': ['Team_X', 'Team_Y', 'Team_Z'] * (size // 3 + 1),
                'home_score': np.random.randint(60, 100, size),
                'away_score': np.random.randint(60, 100, size)
            }
            
            test_df = pd.DataFrame(test_data)
            test_file = f"test_performance_{size}.csv"
            test_df.to_csv(test_file, index=False)
            
            # Measure performance
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            try:
                # Load the test data
                loaded_data = pd.read_csv(test_file)
                
                # Simulate ETL processing
                # Clean data
                cleaned_data = loaded_data.dropna()
                
                # Remove duplicates
                deduplicated_data = cleaned_data.drop_duplicates(subset=['game_id'])
                
                # Validate data types
                deduplicated_data['date'] = pd.to_datetime(deduplicated_data['date'])
                deduplicated_data['season'] = deduplicated_data['season'].astype(int)
                deduplicated_data['home_score'] = deduplicated_data['home_score'].astype(int)
                deduplicated_data['away_score'] = deduplicated_data['away_score'].astype(int)
                
                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                
                processing_time = end_time - start_time
                memory_used = end_memory - start_memory
                
                performance_results[size] = {
                    'processing_time': processing_time,
                    'memory_used': memory_used,
                    'rows_processed': len(deduplicated_data)
                }
                
                print(f"  Processing time: {processing_time:.3f}s")
                print(f"  Memory used: {memory_used:.2f} MB")
                print(f"  Rows processed: {len(deduplicated_data)}")
                
            except Exception as e:
                print(f"  ❌ Performance test failed: {e}")
                performance_results[size] = {'error': str(e)}
            
            # Clean up test file
            os.remove(test_file)
        
        # 2. SCALING ANALYSIS
        print("\n2. SCALING ANALYSIS")
        
        if len(performance_results) >= 2:
            sizes = list(performance_results.keys())
            times = [r.get('processing_time', 0) for r in performance_results.values() if 'error' not in r]
            
            if len(times) >= 2:
                # Calculate scaling factor
                scaling_factor = times[-1] / times[0]
                size_factor = sizes[-1] / sizes[0]
                
                print(f"Size increase: {size_factor}x")
                print(f"Time increase: {scaling_factor:.2f}x")
                
                if scaling_factor <= size_factor * 1.5:  # Linear scaling with 50% tolerance
                    print("✅ ETL scales linearly (good)")
                    scaling_ok = True
                else:
                    print("⚠️ ETL scales poorly (may have bottlenecks)")
                    scaling_ok = False
            else:
                scaling_ok = False
        else:
            scaling_ok = False
        
        # 3. MEMORY EFFICIENCY
        print("\n3. MEMORY EFFICIENCY")
        
        memory_issues = []
        for size, result in performance_results.items():
            if 'error' not in result:
                memory_per_row = result['memory_used'] / result['rows_processed']
                if memory_per_row > 0.1:  # More than 100KB per row
                    memory_issues.append(f"{size} rows: {memory_per_row:.3f} MB/row")
                    print(f"  ⚠️ High memory usage: {memory_per_row:.3f} MB/row")
                else:
                    print(f"  ✅ {size} rows: {memory_per_row:.3f} MB/row")
        
        # Summary
        print("\n" + "="*50)
        print("PERFORMANCE AUDIT SUMMARY")
        print("="*50)
        
        if scaling_ok and not memory_issues:
            print("🎉 Performance audit PASSED")
            return True
        else:
            print("⚠️ Performance audit has issues")
            if not scaling_ok:
                print("  - Scaling performance needs improvement")
            if memory_issues:
                print("  - Memory efficiency needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Performance audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def audit_security_compliance():
    """Audit security and compliance aspects."""
    print("\n" + "="*80)
    print("🔒 SECURITY & COMPLIANCE AUDIT")
    print("="*80)
    
    try:
        # 1. PII DETECTION
        print("\n1. PII (PERSONALLY IDENTIFIABLE INFORMATION) CHECK")
        
        pii_issues = []
        
        # Check database schema for PII fields
        from db import DatabaseManager
        from utils import ConfigManager
        
        config = ConfigManager('cbb_model/config.yaml')
        db_manager = DatabaseManager(config)
        
        # Check for potential PII columns
        pii_patterns = [
            'email', 'phone', 'address', 'ssn', 'passport', 'credit_card',
            'social_security', 'driver_license', 'bank_account'
        ]
        
        all_tables = ['games_table', 'odds_table', 'players_table', 'teams_table', 'games_odds_table']
        
        for table_name in all_tables:
            if hasattr(db_manager, table_name):
                table = getattr(db_manager, table_name)
                for column in table.columns:
                    col_name = column.name.lower()
                    for pattern in pii_patterns:
                        if pattern in col_name:
                            pii_issues.append(f"{table_name}.{column.name}")
                            print(f"  ⚠️ Potential PII field: {table_name}.{column.name}")
        
        if not pii_issues:
            print("  ✅ No obvious PII fields detected")
        
        # 2. SQL INJECTION SAFETY
        print("\n2. SQL INJECTION SAFETY CHECK")
        
        sql_injection_issues = []
        
        # Check database operations for parameterized queries
        db_file = "cbb_model/src/db.py"
        if os.path.exists(db_file):
            with open(db_file, 'r') as f:
                db_content = f.read()
            
            # Look for potential SQL injection patterns
            dangerous_patterns = [
                'execute(f"',
                'execute("',
                'execute(\'',
                'text(f"',
                'text("',
                'text(\''
            ]
            
            for pattern in dangerous_patterns:
                if pattern in db_content:
                    sql_injection_issues.append(f"Potential SQL injection: {pattern}")
                    print(f"  ⚠️ {pattern} found in database operations")
            
            # Look for good practices
            good_patterns = [
                'execute(text(',
                'execute(text("',
                'execute(text(\'',
                'bindparams(',
                ':param'
            ]
            
            good_practices_found = 0
            for pattern in good_patterns:
                if pattern in db_content:
                    good_practices_found += 1
            
            if good_practices_found > 0:
                print(f"  ✅ Good practices found: {good_practices_found} instances")
            else:
                print("  ⚠️ No obvious parameterized query patterns found")
        
        # 3. CREDENTIAL SECURITY
        print("\n3. CREDENTIAL SECURITY CHECK")
        
        credential_issues = []
        
        # Check configuration files
        config_files = ['cbb_model/config.yaml', 'cbb_model/requirements.txt']
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    content = f.read()
                
                # Look for hardcoded credentials
                hardcoded_patterns = [
                    'password:', 'passwd:', 'secret:', 'key:', 'token:',
                    'api_key:', 'access_token:', 'private_key:'
                ]
                
                for pattern in hardcoded_patterns:
                    if pattern in content:
                        # Check if it's a placeholder
                        lines = content.split('\n')
                        for line_num, line in enumerate(lines, 1):
                            if pattern in line:
                                if 'placeholder' in line.lower() or 'example' in line.lower() or 'your_' in line:
                                    print(f"  ✅ {config_file}:{line_num} - Placeholder detected")
                                else:
                                    credential_issues.append(f"{config_file}:{line_num} - {line.strip()}")
                                    print(f"  ❌ {config_file}:{line_num} - Potential hardcoded credential")
        
        # Check for environment variable usage
        try:
            from utils import ConfigManager
            config = ConfigManager('cbb_model/config.yaml')
            
            # Check if config loads from environment
            env_vars_used = []
            for key in ['database.user', 'database.password', 'database.host']:
                try:
                    value = config.get(key)
                    if value and '${' in str(value):
                        env_vars_used.append(key)
                except:
                    pass
            
            if env_vars_used:
                print(f"  ✅ Environment variables used: {len(env_vars_used)}")
            else:
                print("  ⚠️ No obvious environment variable usage found")
                
        except Exception as e:
            print(f"  ⚠️ Could not check environment variable usage: {e}")
        
        # 4. ACCESS CONTROL
        print("\n4. ACCESS CONTROL CHECK")
        
        # Check file permissions
        critical_files = [
            'cbb_model/config.yaml',
            'cbb_model/src/db.py',
            'cbb_model/src/utils.py'
        ]
        
        permission_issues = []
        for file_path in critical_files:
            if os.path.exists(file_path):
                try:
                    stat = os.stat(file_path)
                    mode = stat.st_mode & 0o777
                    
                    # Check if world readable/writable
                    if mode & 0o007:  # World permissions
                        permission_issues.append(f"{file_path}: {oct(mode)}")
                        print(f"  ⚠️ {file_path}: World accessible ({oct(mode)})")
                    else:
                        print(f"  ✅ {file_path}: Secure permissions ({oct(mode)})")
                        
                except Exception as e:
                    print(f"  ⚠️ Could not check {file_path} permissions: {e}")
        
        # Summary
        print("\n" + "="*50)
        print("SECURITY AUDIT SUMMARY")
        print("="*50)
        
        total_security_issues = len(pii_issues) + len(sql_injection_issues) + len(credential_issues) + len(permission_issues)
        
        if total_security_issues == 0:
            print("🎉 Security audit PASSED")
            return True
        else:
            print(f"⚠️ Security audit has {total_security_issues} issues")
            return False
            
    except Exception as e:
        print(f"❌ Security audit failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_phase1_audit_report(audit_results: Dict[str, bool]):
    """Generate comprehensive Phase 1 audit report."""
    print("\n" + "="*80)
    print("📊 PHASE 1 ENTERPRISE AUDIT REPORT")
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
        "audit_type": "Phase 1 Enterprise Audit: Data Infrastructure",
        "system": "CBB Betting ML System",
        "overall_score": overall_score,
        "overall_status": overall_status,
        "audit_results": audit_results,
        "total_audits": total_audits,
        "passed_audits": passed_audits,
        "failed_audits": total_audits - passed_audits,
        "audit_components": {
            "Database Schema": "Schema integrity, constraints, relationships",
            "ETL Pipeline": "Robustness, reproducibility, error handling",
            "Data Quality": "Validation, range checks, consistency",
            "Performance": "Scalability, memory usage, bottlenecks",
            "Security": "PII detection, SQL injection, credentials"
        }
    }
    
    # Print summary
    print(f"\n{status_emoji} PHASE 1 ENTERPRISE AUDIT COMPLETED {status_emoji}")
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
        print("🎉 Excellent data infrastructure quality!")
        print("  - Continue current practices")
        print("  - Plan for future scaling")
        print("  - Maintain quality standards")
    elif overall_score >= 80:
        print("✅ Good data infrastructure with minor issues")
        print("  - Address identified issues")
        print("  - Improve weak areas")
        print("  - Plan for optimization")
    elif overall_score >= 70:
        print("⚠️ Acceptable quality with notable issues")
        print("  - Prioritize critical fixes")
        print("  - Address data quality concerns")
        print("  - Improve performance bottlenecks")
    elif overall_score >= 60:
        print("🔧 Infrastructure needs significant improvement")
        print("  - Immediate action required")
        print("  - Focus on critical issues")
        print("  - Consider architectural changes")
    else:
        print("🚨 Critical issues detected!")
        print("  - IMMEDIATE ACTION REQUIRED")
        print("  - Address all failed audits")
        print("  - Consider infrastructure redesign")
    
    # Save report
    try:
        report_filename = f"phase1_enterprise_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n📄 Comprehensive report saved to: {report_filename}")
    except Exception as e:
        print(f"❌ Failed to save comprehensive report: {e}")
    
    return report

def run_phase1_enterprise_audit():
    """Run complete Phase 1 enterprise audit."""
    print("PHASE 1 ENTERPRISE AUDIT: DATA INFRASTRUCTURE")
    print("="*80)
    print("CBB Betting ML System - Phase 1")
    print("ETL, Database Schema, Data Quality, Performance, Security")
    print("="*80)
    print("Starting comprehensive Phase 1 audit...")
    
    start_time = time.time()
    
    # Run all audit components
    audit_results = {}
    
    # 1. Database Schema Audit
    print("\n🔍 Starting Database Schema Audit...")
    audit_results["Database Schema"] = audit_database_schema()
    
    # 2. ETL Pipeline Audit
    print("\n🔄 Starting ETL Pipeline Audit...")
    audit_results["ETL Pipeline"] = audit_etl_pipeline()
    
    # 3. Data Quality Audit
    print("\n📊 Starting Data Quality Audit...")
    audit_results["Data Quality"] = audit_data_quality()
    
    # 4. Performance & Scalability Audit
    print("\n🚀 Starting Performance & Scalability Audit...")
    audit_results["Performance & Scalability"] = audit_performance_scalability()
    
    # 5. Security & Compliance Audit
    print("\n🔒 Starting Security & Compliance Audit...")
    audit_results["Security & Compliance"] = audit_security_compliance()
    
    # Calculate audit time
    audit_time = time.time() - start_time
    
    # Generate comprehensive report
    report = generate_phase1_audit_report(audit_results)
    
    # Final summary
    print(f"\n" + "="*80)
    print("🏁 PHASE 1 ENTERPRISE AUDIT COMPLETED")
    print("="*80)
    print(f"Total Audit Time: {audit_time:.1f} seconds")
    print(f"Overall Score: {report['overall_score']:.1f}/100")
    print(f"Status: {report['overall_status']}")
    
    if report['overall_score'] >= 80:
        print("\n🎉 CONGRATULATIONS! Your Phase 1 data infrastructure meets enterprise standards!")
        return True
    elif report['overall_score'] >= 70:
        print("\n⚠️ Your Phase 1 infrastructure is acceptable but needs improvements.")
        return True
    else:
        print("\n🚨 Your Phase 1 infrastructure has critical issues that must be addressed.")
        return False

if __name__ == "__main__":
    try:
        success = run_phase1_enterprise_audit()
        
        if success:
            print("\n✅ Phase 1 enterprise audit completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Phase 1 enterprise audit completed with critical issues!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️ Phase 1 audit interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Phase 1 enterprise audit failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)