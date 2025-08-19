#!/usr/bin/env python3
"""
Example usage script for the CBB Betting ML System.

This script demonstrates the basic workflow for:
1. Setting up the system
2. Collecting data
3. Processing data
4. Basic analysis

Run this script to test the system functionality.
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils import setup_logging, ConfigManager, get_logger
from db import create_database_manager
from scrape_games import create_games_scraper
from scrape_odds import create_odds_collector
from etl import create_etl_processor


def main():
    """Main example workflow."""
    print("🏀 CBB Betting ML System - Example Usage")
    print("=" * 50)
    
    # Setup logging
    logger = setup_logging(log_level="INFO")
    logger.info("Starting CBB Betting ML System example")
    
    try:
        # 1. Load configuration
        print("\n1. Loading configuration...")
        config = ConfigManager("config.yaml")
        logger.info("Configuration loaded successfully")
        
        # 2. Test database connection (optional - skip if no DB)
        print("\n2. Testing database connection...")
        try:
            db_manager = create_database_manager()
            logger.info("Database connection successful")
            
            # Create tables
            db_manager.create_tables()
            logger.info("Database tables created")
            
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
            logger.info("Continuing without database (CSV only mode)")
            db_manager = None
        
        # 3. Test games scraper
        print("\n3. Testing games scraper...")
        try:
            games_scraper = create_games_scraper()
            
            # Test with limited teams for demonstration
            result = games_scraper.scrape_and_save_season(2024, max_teams=3)
            
            if result['success']:
                print(f"✅ Games scraping successful: {result['games_count']} games")
                print(f"   CSV saved to: {result['csv_path']}")
                if db_manager:
                    print(f"   Database rows: {result['db_rows']}")
            else:
                print(f"❌ Games scraping failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Games scraper test failed: {e}")
            print(f"❌ Games scraper test failed: {e}")
        
        # 4. Test odds collector
        print("\n4. Testing odds collector...")
        try:
            odds_collector = create_odds_collector()
            
            # Test with recent odds
            result = odds_collector.collect_and_save_odds(days_back=7)
            
            if result['success']:
                print(f"✅ Odds collection successful: {result['odds_count']} odds")
                print(f"   Books collected: {', '.join(result['books_collected'])}")
                print(f"   Files saved: {list(result['saved_files'].values())}")
            else:
                print(f"❌ Odds collection failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Odds collector test failed: {e}")
            print(f"❌ Odds collector test failed: {e}")
        
        # 5. Test ETL pipeline
        print("\n5. Testing ETL pipeline...")
        try:
            etl_processor = create_etl_processor()
            
            # Run ETL pipeline
            result = etl_processor.run_full_etl_pipeline(season=2024, days_back=7)
            
            if result['success']:
                print(f"✅ ETL pipeline successful:")
                print(f"   Games processed: {result['games_count']}")
                print(f"   Odds processed: {result['odds_count']}")
                print(f"   Merged records: {result['merged_count']}")
                print(f"   Features created: {result['features_count']}")
                print(f"   Merged file: {result['merged_file']}")
                print(f"   Features file: {result['features_file']}")
            else:
                print(f"❌ ETL pipeline failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"ETL pipeline test failed: {e}")
            print(f"❌ ETL pipeline test failed: {e}")
        
        # 6. Summary
        print("\n" + "=" * 50)
        print("🏁 Example workflow completed!")
        print("\nNext steps:")
        print("1. Review the generated CSV files in data/raw/ and data/processed/")
        print("2. Explore the data in Jupyter notebooks")
        print("3. Customize configuration in config.yaml")
        print("4. Implement Phase 2 (Feature Engineering)")
        print("5. Implement Phase 3 (Model Training)")
        
        if db_manager:
            db_manager.close()
        
    except Exception as e:
        logger.error(f"Example workflow failed: {e}")
        print(f"❌ Example workflow failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)