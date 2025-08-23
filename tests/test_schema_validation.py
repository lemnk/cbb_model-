#!/usr/bin/env python3
"""
Test script for Phase 5 Schema Validation module.
This script tests the schema validation functionality with various scenarios.
"""

import sys
import os
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_schema_validation():
    """Test the schema validation module."""
    try:
        # Import the monitoring module
        from monitoring.schema_validation import SchemaValidator, GameRecord
        
        print("✅ Successfully imported SchemaValidator and GameRecord")
        
        # Test Case 1: Valid GameRecord creation
        print("\n" + "="*60)
        print("TEST CASE 1: Valid GameRecord creation")
        print("="*60)
        
        try:
            valid_record = GameRecord(
                game_id="TEST001",
                date="2024-01-15",
                season=2024,
                home_team="Duke",
                away_team="UNC",
                team_efficiency=0.75,
                player_availability=0.85,
                dynamic_factors=0.68,
                market_signals=0.72,
                target=1
            )
            print("✅ Valid GameRecord created successfully")
            print(f"   Game ID: {valid_record.game_id}")
            print(f"   Teams: {valid_record.home_team} vs {valid_record.away_team}")
            print(f"   Target: {valid_record.target}")
        except Exception as e:
            print(f"❌ Valid GameRecord creation failed: {e}")
            return False
        
        # Test Case 2: Invalid GameRecord (date format)
        print("\n" + "="*60)
        print("TEST CASE 2: Invalid GameRecord (date format)")
        print("="*60)
        
        try:
            invalid_record = GameRecord(
                game_id="TEST002",
                date="2024/01/15",  # Wrong format
                season=2024,
                home_team="Kansas",
                away_team="Kentucky",
                team_efficiency=0.68,
                player_availability=0.92,
                dynamic_factors=0.71,
                market_signals=0.65,
                target=0
            )
            print("❌ Invalid date format should have failed validation")
            return False
        except Exception as e:
            print("✅ Invalid date format correctly rejected")
            print(f"   Error: {e}")
        
        # Test Case 3: Schema validation with valid DataFrame
        print("\n" + "="*60)
        print("TEST CASE 3: Schema validation with valid DataFrame")
        print("="*60)
        
        # Create valid DataFrame
        valid_data = {
            'game_id': ['TEST001', 'TEST002', 'TEST003'],
            'date': ['2024-01-15', '2024-01-16', '2024-01-17'],
            'season': [2024, 2024, 2024],
            'home_team': ['Duke', 'Kansas', 'Michigan'],
            'away_team': ['UNC', 'Kentucky', 'Ohio State'],
            'team_efficiency': [0.75, 0.68, 0.82],
            'player_availability': [0.85, 0.92, 0.78],
            'dynamic_factors': [0.68, 0.71, 0.74],
            'market_signals': [0.72, 0.65, 0.81],
            'target': [1, 0, 1]
        }
        
        valid_df = pd.DataFrame(valid_data)
        validator = SchemaValidator()
        
        # Test comprehensive validation
        validation_results = validator.comprehensive_validation(valid_df)
        
        if validation_results['is_valid']:
            print("✅ Valid DataFrame passed comprehensive validation")
            print(f"   Total rows: {validation_results['total_rows']}")
            print(f"   Valid rows: {validation_results['valid_rows']}")
            print(f"   Invalid rows: {validation_results['invalid_rows']}")
        else:
            print("❌ Valid DataFrame failed validation")
            print(f"   Errors: {validation_results['errors']}")
            return False
        
        # Test Case 4: Schema validation with invalid DataFrame
        print("\n" + "="*60)
        print("TEST CASE 4: Schema validation with invalid DataFrame")
        print("="*60)
        
        # Create invalid DataFrame (missing column)
        invalid_data = {
            'game_id': ['TEST001', 'TEST002'],
            'date': ['2024-01-15', '2024-01-16'],
            'season': [2024, 2024],
            'home_team': ['Duke', 'Kansas'],
            'away_team': ['UNC', 'Kentucky'],
            'team_efficiency': [0.75, 0.68],
            'player_availability': [0.85, 0.92],
            'dynamic_factors': [0.68, 0.71],
            # Missing market_signals column
            'target': [1, 0]
        }
        
        invalid_df = pd.DataFrame(invalid_data)
        validation_results = validator.comprehensive_validation(invalid_df)
        
        if not validation_results['is_valid']:
            print("✅ Invalid DataFrame correctly failed validation")
            print(f"   Errors: {validation_results['errors']}")
        else:
            print("❌ Invalid DataFrame should have failed validation")
            return False
        
        # Test Case 5: Row-by-row validation
        print("\n" + "="*60)
        print("TEST CASE 5: Row-by-row validation")
        print("="*60)
        
        # Test individual row validation
        valid_row = valid_df.iloc[0].to_dict()
        is_valid, error_msg = validator.validate_row(valid_row)
        
        if is_valid:
            print("✅ Individual row validation passed")
        else:
            print(f"❌ Individual row validation failed: {error_msg}")
            return False
        
        # Test Case 6: DataFrame validation method
        print("\n" + "="*60)
        print("TEST CASE 6: DataFrame validation method")
        print("="*60)
        
        # Test validate_dataframe method
        is_valid, errors = validator.validate_dataframe(valid_df)
        
        if is_valid and len(errors) == 0:
            print("✅ DataFrame validation method passed")
        else:
            print(f"❌ DataFrame validation method failed: {errors}")
            return False
        
        # Test Case 7: Validation summary
        print("\n" + "="*60)
        print("TEST CASE 7: Validation summary")
        print("="*60)
        
        # Test validation summary
        summary = validator.get_validation_summary(validation_results)
        print("✅ Validation summary generated:")
        print(f"   Summary: {summary}")
        
        # Test Case 8: Edge cases
        print("\n" + "="*60)
        print("TEST CASE 8: Edge cases")
        print("="*60)
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        empty_results = validator.comprehensive_validation(empty_df)
        
        if not empty_results['is_valid']:
            print("✅ Empty DataFrame correctly handled")
        else:
            print("❌ Empty DataFrame should have failed validation")
            return False
        
        # Test DataFrame with wrong types
        wrong_types_df = valid_df.copy()
        wrong_types_df['team_efficiency'] = 'invalid_string'  # Should be float
        
        wrong_types_results = validator.comprehensive_validation(wrong_types_df)
        
        if not wrong_types_results['is_valid']:
            print("✅ Wrong data types correctly detected")
        else:
            print("❌ Wrong data types should have failed validation")
            return False
        
        # Overall test summary
        print("\n" + "="*60)
        print("OVERALL TEST SUMMARY")
        print("="*60)
        
        total_tests = 8
        passed_tests = 8  # All tests passed if we got here
        
        print(f"Tests passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            print("🎉 All test cases passed! Schema validation is working correctly.")
            return True
        else:
            print(f"❌ {total_tests - passed_tests} test cases failed. Check the details above.")
            return False
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This usually means required packages are not installed.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Phase 5: Schema Validation Test")
    print("=" * 50)
    
    success = test_schema_validation()
    
    if success:
        print("\n✅ Schema Validation: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Schema Validation: NEEDS ATTENTION")
        sys.exit(1)