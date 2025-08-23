#!/usr/bin/env python3
"""
Test script for Phase 5 Schema Validation module.
This script tests the schema validation functionality without external dependencies.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_schema_validation():
    """Test the schema validation module."""
    try:
        # Import the monitoring module
        from monitoring.schema_validation import SchemaValidator, GameRecord
        
        print("✅ Successfully imported SchemaValidator and GameRecord")
        
        # Test GameRecord validation
        print("\nTesting GameRecord validation...")
        
        # Valid record
        valid_record = {
            'game_id': 'G001',
            'date': '2024-01-15',
            'season': 2024,
            'home_team': 'Team A',
            'away_team': 'Team C',
            'team_efficiency': 0.75,
            'player_availability': 0.90,
            'dynamic_factors': 0.60,
            'market_signals': 0.45,
            'target': 1
        }
        
        try:
            game_record = GameRecord(**valid_record)
            print("✅ Valid record created successfully")
        except Exception as e:
            print(f"❌ Valid record failed: {e}")
            return False
        
        # Test invalid record (wrong date format)
        invalid_record = valid_record.copy()
        invalid_record['date'] = '2024/01/15'  # Wrong format
        
        try:
            game_record = GameRecord(**invalid_record)
            print("❌ Invalid record should have failed")
            return False
        except Exception as e:
            print("✅ Invalid record correctly rejected")
        
        # Test SchemaValidator
        print("\nTesting SchemaValidator...")
        validator = SchemaValidator()
        print("✅ SchemaValidator created successfully")
        
        # Test with sample data
        import pandas as pd
        
        sample_data = {
            'game_id': ['G001', 'G002'],
            'date': ['2024-01-15', '2024-01-16'],
            'season': [2024, 2024],
            'home_team': ['Team A', 'Team B'],
            'away_team': ['Team C', 'Team D'],
            'team_efficiency': [0.75, 0.82],
            'player_availability': [0.90, 0.85],
            'dynamic_factors': [0.60, 0.70],
            'market_signals': [0.45, 0.55],
            'target': [1, 0]
        }
        
        df = pd.DataFrame(sample_data)
        print("✅ Sample DataFrame created")
        
        # Test comprehensive validation
        results = validator.comprehensive_validation(df)
        print("✅ Comprehensive validation completed")
        
        # Print results
        summary = validator.get_validation_summary(results)
        print("\n" + summary)
        
        if results['is_valid']:
            print("\n🎉 All tests passed! Schema validation is working correctly.")
            return True
        else:
            print("\n❌ Validation failed. Check the error details above.")
            return False
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This usually means pandas or pydantic is not installed.")
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
        print("\n✅ Schema Validation Module: READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("\n❌ Schema Validation Module: NEEDS ATTENTION")
        sys.exit(1)