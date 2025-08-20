"""
Betting Odds Data Scraper for CBB Betting ML System.

This module scrapes betting odds data from:
- Pinnacle Sports
- DraftKings Sportsbook
- Other major sportsbooks

Note: This is a framework with stubs for actual implementation.
Real odds data collection may require:
- Official API access
- Selenium for dynamic content
- Rate limiting and anti-bot measures
"""

import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import json
import re

from .utils import get_logger, ConfigManager, safe_filename, parse_date_range
from .db import DatabaseManager


class OddsScraper:
    """Base class for odds scraping functionality."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize the odds scraper.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        self.config = config
        self.logger = get_logger('odds_scraper')
        self.db_manager = db_manager
        
        # Get configuration
        self.raw_data_dir = self.config.get('data_collection.raw_data_dir', 'data/raw')
        self.days_back = self.config.get('data_collection.odds.days_back', 30)
        
        # Create raw data directory if it doesn't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _make_request(self, url: str, retries: int = 3, delay: float = 1.0) -> Optional[requests.Response]:
        """Make HTTP request with retry logic and rate limiting.
        
        Args:
            url: URL to request
            retries: Number of retry attempts
            delay: Delay between requests in seconds
            
        Returns:
            Response object if successful, None otherwise
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(delay)
                
                return response
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None
        
        return None
    
    def save_odds_to_csv(self, odds_df: pd.DataFrame, book: str, date: str) -> str:
        """Save odds data to CSV file.
        
        Args:
            odds_df: DataFrame with odds data
            book: Sportsbook name (e.g., 'pinnacle', 'draftkings')
            date: Date string for filename
            
        Returns:
            Path to saved CSV file
        """
        filename = safe_filename(f"odds_{book}_{date}.csv")
        filepath = os.path.join(self.raw_data_dir, filename)
        
        odds_df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(odds_df)} odds records to {filepath}")
        
        return filepath
    
    def save_odds_to_db(self, odds_df: pd.DataFrame) -> int:
        """Save odds data to database.
        
        Args:
            odds_df: DataFrame with odds data
            
        Returns:
            Number of rows inserted/updated
        """
        if self.db_manager is None:
            self.logger.warning("No database manager provided, skipping database save")
            return 0
        
        try:
            rows_affected = self.db_manager.insert_or_update(
                'odds',
                odds_df,
                conflict_columns=['odds_id']
            )
            return rows_affected
            
        except Exception as e:
            self.logger.error(f"Error saving odds to database: {e}")
            return 0


class PinnacleOddsScraper(OddsScraper):
    """Scrapes odds data from Pinnacle Sports."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize Pinnacle scraper.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        super().__init__(config, db_manager)
        
        # Get Pinnacle-specific configuration
        self.base_url = self.config.get('apis.pinnacle.base_url')
        self.api_key = self.config.get('apis.pinnacle.api_key')
        self.rate_limit_delay = self.config.get('apis.pinnacle.rate_limit_delay', 0.5)
        
        if not self.api_key:
            self.logger.warning("No Pinnacle API key provided. Some functionality may be limited.")
    
    def get_ncaa_basketball_odds(self, date: datetime) -> pd.DataFrame:
        """Get NCAA basketball odds for a specific date.
        
        Args:
            date: Date to get odds for
            
        Returns:
            DataFrame with odds data
        """
        self.logger.info(f"Fetching Pinnacle odds for {date.strftime('%Y-%m-%d')}")
        
        # This is a stub - in practice, you would:
        # 1. Use Pinnacle's official API if you have access
        # 2. Parse the response to extract odds
        # 3. Handle different bet types (moneyline, spread, total)
        
        # For demonstration, we'll create sample data
        sample_odds = self._create_sample_odds(date, 'pinnacle')
        
        if sample_odds:
            return pd.DataFrame(sample_odds)
        else:
            self.logger.warning("No odds data available for the specified date")
            return pd.DataFrame()
    
    def _create_sample_odds(self, date: datetime, book: str) -> List[Dict[str, Any]]:
        """Create sample odds data for demonstration purposes.
        
        Args:
            date: Date for the odds
            book: Sportsbook name
            
        Returns:
            List of sample odds dictionaries
        """
        # Sample NCAA teams for demonstration
        sample_games = [
            {'home': 'Duke', 'away': 'North Carolina', 'home_id': 'duke', 'away_id': 'north-carolina'},
            {'home': 'Kentucky', 'away': 'Kansas', 'home_id': 'kentucky', 'away_id': 'kansas'},
            {'home': 'Michigan State', 'away': 'Indiana', 'home_id': 'michigan-state', 'away_id': 'indiana'},
        ]
        
        odds_data = []
        
        for game in sample_games:
            # Generate unique odds ID
            odds_id = f"{book}_{date.strftime('%Y%m%d')}_{game['home_id']}_{game['away_id']}"
            
            # Generate sample odds
            odds_record = {
                'odds_id': odds_id,
                'game_id': f"{date.year}_{game['home_id']}_{game['away_id']}_{date.strftime('%Y%m%d')}",
                'book': book,
                'open_time': date.replace(hour=9, minute=0, second=0, microsecond=0),
                'close_time': date.replace(hour=19, minute=0, second=0, microsecond=0),
                'open_moneyline_home': -110,
                'open_moneyline_away': -110,
                'close_moneyline_home': -105,
                'close_moneyline_away': -115,
                'open_spread': -2.5,
                'close_spread': -3.0,
                'open_total': 145.5,
                'close_total': 146.0,
                'spread_side': 'home'
            }
            
            odds_data.append(odds_record)
        
        return odds_data
    
    def scrape_historical_odds(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Scrape historical odds data for a date range.
        
        Args:
            start_date: Start date for odds collection
            end_date: End date for odds collection
            
        Returns:
            DataFrame with historical odds data
        """
        self.logger.info(f"Scraping Pinnacle odds from {start_date} to {end_date}")
        
        all_odds = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                daily_odds = self.get_ncaa_basketball_odds(current_date)
                if not daily_odds.empty:
                    all_odds.append(daily_odds)
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                self.logger.warning(f"Error scraping odds for {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        if all_odds:
            combined_odds = pd.concat(all_odds, ignore_index=True)
            self.logger.info(f"Collected {len(combined_odds)} odds records")
            return combined_odds
        else:
            self.logger.warning("No odds data collected")
            return pd.DataFrame()


class DraftKingsOddsScraper(OddsScraper):
    """Scrapes odds data from DraftKings Sportsbook."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize DraftKings scraper.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        super().__init__(config, db_manager)
        
        # Get DraftKings-specific configuration
        self.base_url = self.config.get('apis.draftkings.base_url')
        self.rate_limit_delay = self.config.get('apis.draftkings.rate_limit_delay', 1.0)
    
    def get_ncaa_basketball_odds(self, date: datetime) -> pd.DataFrame:
        """Get NCAA basketball odds for a specific date.
        
        Args:
            date: Date to get odds for
            
        Returns:
            DataFrame with odds data
        """
        self.logger.info(f"Fetching DraftKings odds for {date.strftime('%Y-%m-%d')}")
        
        # This is a stub - in practice, you would:
        # 1. Navigate to DraftKings NCAA basketball page
        # 2. Use Selenium if the content is dynamic
        # 3. Parse the odds from the page
        # 4. Handle different bet types and lines
        
        # For demonstration, we'll create sample data
        sample_odds = self._create_sample_odds(date, 'draftkings')
        
        if sample_odds:
            return pd.DataFrame(sample_odds)
        else:
            self.logger.warning("No odds data available for the specified date")
            return pd.DataFrame()
    
    def _create_sample_odds(self, date: datetime, book: str) -> List[Dict[str, Any]]:
        """Create sample odds data for demonstration purposes.
        
        Args:
            date: Date for the odds
            book: Sportsbook name
            
        Returns:
            List of sample odds dictionaries
        """
        # Sample NCAA teams for demonstration
        sample_games = [
            {'home': 'Duke', 'away': 'North Carolina', 'home_id': 'duke', 'away_id': 'north-carolina'},
            {'home': 'Kentucky', 'away': 'Kansas', 'home_id': 'kentucky', 'away_id': 'kansas'},
            {'home': 'Michigan State', 'away': 'Indiana', 'home_id': 'michigan-state', 'away_id': 'indiana'},
        ]
        
        odds_data = []
        
        for game in sample_games:
            # Generate unique odds ID
            odds_id = f"{book}_{date.strftime('%Y%m%d')}_{game['home_id']}_{game['away_id']}"
            
            # Generate sample odds (slightly different from Pinnacle for variety)
            odds_record = {
                'odds_id': odds_id,
                'game_id': f"{date.year}_{game['home_id']}_{game['away_id']}_{date.strftime('%Y%m%d')}",
                'book': book,
                'open_time': date.replace(hour=10, minute=0, second=0, microsecond=0),
                'close_time': date.replace(hour=20, minute=0, second=0, microsecond=0),
                'open_moneyline_home': -115,
                'open_moneyline_away': -105,
                'close_moneyline_home': -110,
                'close_moneyline_away': -110,
                'open_spread': -2.0,
                'close_spread': -2.5,
                'open_total': 146.0,
                'close_total': 145.5,
                'spread_side': 'home'
            }
            
            odds_data.append(odds_record)
        
        return odds_data
    
    def scrape_historical_odds(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Scrape historical odds data for a date range.
        
        Args:
            start_date: Start date for odds collection
            end_date: End date for odds collection
            
        Returns:
            DataFrame with historical odds data
        """
        self.logger.info(f"Scraping DraftKings odds from {start_date} to {end_date}")
        
        all_odds = []
        current_date = start_date
        
        while current_date <= end_date:
            try:
                daily_odds = self.get_ncaa_basketball_odds(current_date)
                if not daily_odds.empty:
                    all_odds.append(daily_odds)
                
                current_date += timedelta(days=1)
                
            except Exception as e:
                self.logger.warning(f"Error scraping odds for {current_date}: {e}")
                current_date += timedelta(days=1)
                continue
        
        if all_odds:
            combined_odds = pd.concat(all_odds, ignore_index=True)
            self.logger.info(f"Collected {len(combined_odds)} odds records")
            return combined_odds
        else:
            self.logger.warning("No odds data collected")
            return pd.DataFrame()


class OddsDataCollector:
    """Main class for collecting odds data from multiple sources."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize the odds data collector.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        self.config = config
        self.logger = get_logger('odds_collector')
        self.db_manager = db_manager
        
        # Initialize scrapers
        self.pinnacle_scraper = PinnacleOddsScraper(config, db_manager)
        self.draftkings_scraper = DraftKingsOddsScraper(config, db_manager)
    
    def collect_recent_odds(self, days_back: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Collect recent odds data from all sources.
        
        Args:
            days_back: Number of days back to collect (uses config default if None)
            
        Returns:
            Dictionary mapping book names to DataFrames
        """
        if days_back is None:
            days_back = self.config.get('data_collection.odds.days_back', 30)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        self.logger.info(f"Collecting odds data from {start_date} to {end_date}")
        
        results = {}
        
        # Collect from Pinnacle
        try:
            pinnacle_odds = self.pinnacle_scraper.scrape_historical_odds(start_date, end_date)
            if not pinnacle_odds.empty:
                results['pinnacle'] = pinnacle_odds
                self.logger.info(f"Collected {len(pinnacle_odds)} Pinnacle odds records")
        except Exception as e:
            self.logger.error(f"Error collecting Pinnacle odds: {e}")
        
        # Collect from DraftKings
        try:
            draftkings_odds = self.draftkings_scraper.scrape_historical_odds(start_date, end_date)
            if not draftkings_odds.empty:
                results['draftkings'] = draftkings_odds
                self.logger.info(f"Collected {len(draftkings_odds)} DraftKings odds records")
        except Exception as e:
            self.logger.error(f"Error collecting DraftKings odds: {e}")
        
        return results
    
    def save_all_odds(self, odds_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Save odds data from all sources to CSV and database.
        
        Args:
            odds_data: Dictionary mapping book names to DataFrames
            
        Returns:
            Dictionary mapping book names to saved CSV file paths
        """
        saved_files = {}
        
        for book, df in odds_data.items():
            if not df.empty:
                try:
                    # Save to CSV
                    date_str = datetime.now().strftime('%Y%m%d')
                    csv_path = self.pinnacle_scraper.save_odds_to_csv(df, book, date_str)
                    saved_files[book] = csv_path
                    
                    # Save to database
                    db_rows = self.pinnacle_scraper.save_odds_to_db(df)
                    self.logger.info(f"Saved {db_rows} {book} odds records to database")
                    
                except Exception as e:
                    self.logger.error(f"Error saving {book} odds: {e}")
        
        return saved_files
    
    def collect_and_save_odds(self, days_back: Optional[int] = None) -> Dict[str, Any]:
        """Complete workflow: collect odds and save to both CSV and database.
        
        Args:
            days_back: Number of days back to collect
            
        Returns:
            Dictionary with results summary
        """
        start_time = time.time()
        
        try:
            # Collect odds data
            odds_data = self.collect_recent_odds(days_back)
            
            if not odds_data:
                return {
                    'success': False,
                    'odds_count': 0,
                    'error': 'No odds data collected'
                }
            
            # Save data
            saved_files = self.save_all_odds(odds_data)
            
            # Count total records
            total_odds = sum(len(df) for df in odds_data.values())
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'odds_count': total_odds,
                'books_collected': list(odds_data.keys()),
                'saved_files': saved_files,
                'duration_seconds': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error in odds collection workflow: {e}")
            
            return {
                'success': False,
                'odds_count': 0,
                'error': str(e),
                'duration_seconds': duration
            }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'pinnacle_scraper'):
            self.pinnacle_scraper.session.close()
        if hasattr(self, 'draftkings_scraper'):
            self.draftkings_scraper.session.close()


def create_odds_collector(config_path: str = "config.yaml") -> OddsDataCollector:
    """Create and return an odds data collector instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        OddsDataCollector instance
    """
    config = ConfigManager(config_path)
    return OddsDataCollector(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the odds collector
    try:
        collector = create_odds_collector()
        
        # Collect recent odds (last 7 days for testing)
        result = collector.collect_and_save_odds(days_back=7)
        
        print(f"Odds collection result: {result}")
        
        collector.close()
        
    except Exception as e:
        print(f"Error: {e}")