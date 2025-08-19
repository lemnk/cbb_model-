#!/usr/bin/env python3
"""
Test script for CBB Betting ML System.

This script tests the basic functionality of all system components.
Run this to verify everything is working correctly.
"""

import os
import sys
import traceback

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from utils import ConfigManager, setup_logging, get_logger
        print("✅ utils module imported successfully")
    except Exception as e:
        print(f"❌ utils module import failed: {e}")
        return False
    
    try:
        from db import DatabaseManager, create_database_manager
        print("✅ db module imported successfully")
    except Exception as e:
        print(f"❌ db module import failed: {e}")
        return False
    
    try:
        from scrape_games import NCAAGamesScraper, create_games_scraper
        print("✅ scrape_games module imported successfully")
    except Exception as e:
        print(f"❌ scrape_games module import failed: {e}")
        return False
    
    try:
        from scrape_odds import OddsDataCollector, create_odds_collector
        print("✅ scrape_odds module imported successfully")
    except Exception as e:
        print(f"❌ scrape_odds module import failed: {e}")
        return False
    
    try:
        from etl import CBBDataETL, create_etl_processor
        print("✅ etl module imported successfully")
    except Exception as e:
        print(f"❌ etl module import failed: {e}")
        return False
    
    try:
        from features import CBBFeatureEngineer, create_feature_engineer
        print("✅ features module imported successfully")
    except Exception as e:
        print(f"❌ features module import failed: {e}")
        return False
    
    try:
        from train import CBBModelTrainer, create_model_trainer
        print("✅ train module imported successfully")
    except Exception as e:
        print(f"❌ train module import failed: {e}")
        return False
    
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from utils import ConfigManager
        
        # Test with default config
        config = ConfigManager("config.yaml")
        print("✅ Configuration loaded successfully")
        
        # Test some config values
        db_config = config.get('database', {})
        print(f"   Database host: {db_config.get('host', 'Not set')}")
        
        start_season = config.get('data_collection.start_season', 'Not set')
        print(f"   Start season: {start_season}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_logging():
    """Test logging setup."""
    print("\nTesting logging...")
    
    try:
        from utils import setup_logging, get_logger
        
        # Setup logging
        logger = setup_logging(log_level="INFO")
        print("✅ Logging setup successful")
        
        # Test logging
        test_logger = get_logger('test')
        test_logger.info("Test log message")
        print("✅ Logging test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        traceback.print_exc()
        return False


def test_utilities():
    """Test utility functions."""
    print("\nTesting utility functions...")
    
    try:
        from utils import safe_filename, get_season_dates, parse_date_range
        from datetime import datetime
        
        # Test safe_filename
        test_filename = "test file (2024).csv"
        safe_name = safe_filename(test_filename)
        print(f"✅ safe_filename: '{test_filename}' -> '{safe_name}'")
        
        # Test get_season_dates
        start_date, end_date = get_season_dates(2024)
        print(f"✅ Season 2024: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Test parse_date_range
        start, end = parse_date_range("2024-01-01", "2024-01-31")
        print(f"✅ Date range parsed: {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utility functions test failed: {e}")
        traceback.print_exc()
        return False


def test_scraper_creation():
    """Test scraper object creation."""
    print("\nTesting scraper creation...")
    
    try:
        from utils import ConfigManager
        from scrape_games import create_games_scraper
        from scrape_odds import create_odds_collector
        
        # Create scrapers
        games_scraper = create_games_scraper()
        print("✅ Games scraper created successfully")
        
        odds_collector = create_odds_collector()
        print("✅ Odds collector created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Scraper creation test failed: {e}")
        traceback.print_exc()
        return False


def test_etl_creation():
    """Test ETL processor creation."""
    print("\nTesting ETL processor creation...")
    
    try:
        from etl import create_etl_processor
        
        # Create ETL processor
        etl_processor = create_etl_processor()
        print("✅ ETL processor created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ ETL processor creation test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🏀 CBB Betting ML System - System Test")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Configuration", test_config),
        ("Logging", test_logging),
        ("Utility Functions", test_utilities),
        ("Scraper Creation", test_scraper_creation),
        ("ETL Creation", test_etl_creation),
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
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)