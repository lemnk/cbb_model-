"""
ETL (Extract, Transform, Load) Pipeline for CBB Betting ML System.

This module handles:
- Loading raw CSV data
- Cleaning and normalizing data
- Merging games and odds data
- Feature engineering preparation
- Data validation and quality checks
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

from .utils import get_logger, ConfigManager, safe_filename, validate_dataframe
from .db import DatabaseManager


class CBBDataETL:
    """Main ETL class for processing CBB betting data."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize the ETL processor.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        self.config = config
        self.logger = get_logger('etl')
        self.db_manager = db_manager
        
        # Get configuration
        self.raw_data_dir = self.config.get('data_collection.raw_data_dir', 'data/raw')
        self.processed_data_dir = self.config.get('data_collection.processed_data_dir', 'data/processed')
        self.backup_dir = self.config.get('data_collection.backup_dir', 'data/backup')
        
        # Create directories if they don't exist
        for directory in [self.processed_data_dir, self.backup_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def load_raw_games_data(self, season: Optional[int] = None) -> pd.DataFrame:
        """Load raw games data from CSV files.
        
        Args:
            season: Specific season to load (if None, loads all available)
            
        Returns:
            DataFrame with games data
        """
        self.logger.info("Loading raw games data...")
        
        games_files = []
        
        # Find games CSV files
        for file in os.listdir(self.raw_data_dir):
            if file.startswith('games_') and file.endswith('.csv'):
                if season is None or f'games_{season}.csv' == file:
                    games_files.append(file)
        
        if not games_files:
            self.logger.warning("No games CSV files found")
            return pd.DataFrame()
        
        # Load and combine all games data
        all_games = []
        
        for file in games_files:
            try:
                filepath = os.path.join(self.raw_data_dir, file)
                games_df = pd.read_csv(filepath)
                
                # Add source file information
                games_df['source_file'] = file
                games_df['loaded_at'] = datetime.now()
                
                all_games.append(games_df)
                self.logger.info(f"Loaded {len(games_df)} games from {file}")
                
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                continue
        
        if not all_games:
            self.logger.warning("No games data loaded")
            return pd.DataFrame()
        
        # Combine all games data
        combined_games = pd.concat(all_games, ignore_index=True)
        
        # Remove duplicates based on game_id
        initial_count = len(combined_games)
        combined_games = combined_games.drop_duplicates(subset=['game_id'], keep='first')
        final_count = len(combined_games)
        
        if initial_count != final_count:
            self.logger.info(f"Removed {initial_count - final_count} duplicate games")
        
        self.logger.info(f"Total games loaded: {len(combined_games)}")
        return combined_games
    
    def load_raw_odds_data(self, days_back: Optional[int] = None) -> pd.DataFrame:
        """Load raw odds data from CSV files.
        
        Args:
            days_back: Only load odds from last N days (if None, loads all)
            
        Returns:
            DataFrame with odds data
        """
        self.logger.info("Loading raw odds data...")
        
        odds_files = []
        
        # Find odds CSV files
        for file in os.listdir(self.raw_data_dir):
            if file.startswith('odds_') and file.endswith('.csv'):
                odds_files.append(file)
        
        if not odds_files:
            self.logger.warning("No odds CSV files found")
            return pd.DataFrame()
        
        # Load and combine all odds data
        all_odds = []
        
        for file in odds_files:
            try:
                filepath = os.path.join(self.raw_data_dir, file)
                odds_df = pd.read_csv(filepath)
                
                # Add source file information
                odds_df['source_file'] = file
                odds_df['loaded_at'] = datetime.now()
                
                all_odds.append(odds_df)
                self.logger.info(f"Loaded {len(odds_df)} odds records from {file}")
                
            except Exception as e:
                self.logger.error(f"Error loading {file}: {e}")
                continue
        
        if not all_odds:
            self.logger.warning("No odds data loaded")
            return pd.DataFrame()
        
        # Combine all odds data
        combined_odds = pd.concat(all_odds, ignore_index=True)
        
        # Filter by date if specified
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            combined_odds['open_time'] = pd.to_datetime(combined_odds['open_time'])
            combined_odds = combined_odds[combined_odds['open_time'] >= cutoff_date]
            self.logger.info(f"Filtered to {len(combined_odds)} odds records from last {days_back} days")
        
        # Remove duplicates based on odds_id
        initial_count = len(combined_odds)
        combined_odds = combined_odds.drop_duplicates(subset=['odds_id'], keep='first')
        final_count = len(combined_odds)
        
        if initial_count != final_count:
            self.logger.info(f"Removed {initial_count - final_count} duplicate odds")
        
        self.logger.info(f"Total odds loaded: {len(combined_odds)}")
        return combined_odds
    
    def clean_games_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize games data.
        
        Args:
            games_df: Raw games DataFrame
            
        Returns:
            Cleaned games DataFrame
        """
        if games_df.empty:
            return games_df
        
        self.logger.info("Cleaning games data...")
        
        # Create a copy to avoid modifying original
        cleaned_games = games_df.copy()
        
        # Convert date column to datetime if it's not already
        if 'date' in cleaned_games.columns:
            cleaned_games['date'] = pd.to_datetime(cleaned_games['date'])
        
        # Clean team names - remove special characters and normalize
        for col in ['home_team', 'away_team']:
            if col in cleaned_games.columns:
                cleaned_games[col] = cleaned_games[col].astype(str).str.strip()
                # Remove common special characters but keep hyphens for team IDs
                cleaned_games[col] = cleaned_games[col].str.replace(r'[^\w\s-]', '', regex=True)
        
        # Clean numeric columns
        numeric_columns = ['home_score', 'away_score', 'attendance']
        for col in numeric_columns:
            if col in cleaned_games.columns:
                # Convert to numeric, coerce errors to NaN
                cleaned_games[col] = pd.to_numeric(cleaned_games[col], errors='coerce')
        
        # Clean boolean columns
        if 'overtime' in cleaned_games.columns:
            cleaned_games['overtime'] = cleaned_games['overtime'].astype(bool)
        
        # Clean location data
        if 'location' in cleaned_games.columns:
            cleaned_games['location'] = cleaned_games['location'].astype(str).str.strip()
            # Replace NaN/None with 'Unknown'
            cleaned_games['location'] = cleaned_games['location'].replace(['nan', 'None', ''], 'Unknown')
        
        # Generate missing team IDs if they don't exist
        if 'home_team_id' not in cleaned_games.columns:
            cleaned_games['home_team_id'] = cleaned_games['home_team'].str.lower().str.replace(' ', '-')
        if 'away_team_id' not in cleaned_games.columns:
            cleaned_games['away_team_id'] = cleaned_games['away_team'].str.lower().str.replace(' ', '-')
        
        # Remove rows with missing critical data
        critical_columns = ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
        missing_critical = cleaned_games[critical_columns].isnull().any(axis=1)
        
        if missing_critical.any():
            self.logger.warning(f"Removing {missing_critical.sum()} rows with missing critical data")
            cleaned_games = cleaned_games[~missing_critical]
        
        # Sort by date
        if 'date' in cleaned_games.columns:
            cleaned_games = cleaned_games.sort_values('date')
        
        self.logger.info(f"Games data cleaned: {len(cleaned_games)} rows remaining")
        return cleaned_games
    
    def clean_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize odds data.
        
        Args:
            odds_df: Raw odds DataFrame
            
        Returns:
            Cleaned odds DataFrame
        """
        if odds_df.empty:
            return odds_df
        
        self.logger.info("Cleaning odds data...")
        
        # Create a copy to avoid modifying original
        cleaned_odds = odds_df.copy()
        
        # Convert datetime columns
        datetime_columns = ['open_time', 'close_time']
        for col in datetime_columns:
            if col in cleaned_odds.columns:
                cleaned_odds[col] = pd.to_datetime(cleaned_odds[col], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = [
            'open_moneyline_home', 'open_moneyline_away',
            'close_moneyline_home', 'close_moneyline_away',
            'open_spread', 'close_spread',
            'open_total', 'close_total'
        ]
        
        for col in numeric_columns:
            if col in cleaned_odds.columns:
                cleaned_odds[col] = pd.to_numeric(cleaned_odds[col], errors='coerce')
        
        # Clean book names
        if 'book' in cleaned_odds.columns:
            cleaned_odds['book'] = cleaned_odds['book'].astype(str).str.lower().str.strip()
        
        # Clean spread_side
        if 'spread_side' in cleaned_odds.columns:
            cleaned_odds['spread_side'] = cleaned_odds['spread_side'].astype(str).str.lower().str.strip()
            # Validate spread_side values
            valid_sides = ['home', 'away']
            cleaned_odds['spread_side'] = cleaned_odds['spread_side'].apply(
                lambda x: x if x in valid_sides else 'home'
            )
        
        # Remove rows with missing critical data
        critical_columns = ['odds_id', 'game_id', 'book']
        missing_critical = cleaned_odds[critical_columns].isnull().any(axis=1)
        
        if missing_critical.any():
            self.logger.warning(f"Removing {missing_critical.sum()} odds rows with missing critical data")
            cleaned_odds = cleaned_odds[~missing_critical]
        
        # Sort by open_time
        if 'open_time' in cleaned_odds.columns:
            cleaned_odds = cleaned_odds.sort_values('open_time')
        
        self.logger.info(f"Odds data cleaned: {len(cleaned_odds)} rows remaining")
        return cleaned_odds
    
    def merge_games_and_odds(self, games_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Merge games and odds data into a unified dataset.
        
        Args:
            games_df: Cleaned games DataFrame
            odds_df: Cleaned odds DataFrame
            
        Returns:
            Merged DataFrame with games and odds data
        """
        if games_df.empty:
            self.logger.warning("No games data to merge")
            return pd.DataFrame()
        
        self.logger.info("Merging games and odds data...")
        
        # Start with games data
        merged_df = games_df.copy()
        
        if odds_df.empty:
            self.logger.warning("No odds data available, returning games data only")
            return merged_df
        
        # Pivot odds data to get one row per game with columns for each book
        odds_pivot = self._pivot_odds_data(odds_df)
        
        # Merge games with odds
        merged_df = merged_df.merge(
            odds_pivot,
            on='game_id',
            how='left',
            suffixes=('', '_odds')
        )
        
        # Fill missing odds columns with NaN
        odds_columns = [col for col in merged_df.columns if any(book in col for book in ['pinnacle', 'draftkings'])]
        for col in odds_columns:
            if col not in merged_df.columns:
                merged_df[col] = np.nan
        
        self.logger.info(f"Merged dataset created: {len(merged_df)} rows")
        return merged_df
    
    def _pivot_odds_data(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot odds data to create wide format with one row per game.
        
        Args:
            odds_df: Cleaned odds DataFrame
            
        Returns:
            Pivoted odds DataFrame
        """
        if odds_df.empty:
            return pd.DataFrame()
        
        # Create a copy for pivoting
        pivot_odds = odds_df.copy()
        
        # Create wide format columns for each book
        wide_odds = []
        
        for book in pivot_odds['book'].unique():
            book_odds = pivot_odds[pivot_odds['book'] == book].copy()
            
            # Rename columns to include book name
            book_odds = book_odds.rename(columns={
                'open_moneyline_home': f'{book}_open_ml_home',
                'open_moneyline_away': f'{book}_open_ml_away',
                'close_moneyline_home': f'{book}_close_ml_home',
                'close_moneyline_away': f'{book}_close_ml_away',
                'open_spread': f'{book}_open_spread',
                'close_spread': f'{book}_close_spread',
                'open_total': f'{book}_open_total',
                'close_total': f'{book}_close_total'
            })
            
            # Select only the columns we need
            columns_to_keep = ['game_id'] + [col for col in book_odds.columns if book in col]
            book_odds = book_odds[columns_to_keep]
            
            wide_odds.append(book_odds)
        
        if wide_odds:
            # Merge all books
            result = wide_odds[0]
            for book_df in wide_odds[1:]:
                result = result.merge(book_df, on='game_id', how='outer')
            
            return result
        else:
            return pd.DataFrame()
    
    def create_features_dataset(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Create features dataset for ML pipeline (Phase 2 preparation).
        
        Args:
            merged_df: Merged games and odds DataFrame
            
        Returns:
            Features DataFrame ready for ML
        """
        if merged_df.empty:
            return merged_df
        
        self.logger.info("Creating features dataset...")
        
        features_df = merged_df.copy()
        
        # Basic game features
        features_df['total_score'] = features_df['home_score'] + features_df['away_score']
        features_df['score_differential'] = features_df['home_score'] - features_df['away_score']
        features_df['home_win'] = (features_df['home_score'] > features_df['away_score']).astype(int)
        
        # Date features
        if 'date' in features_df.columns:
            features_df['day_of_week'] = features_df['date'].dt.dayofweek
            features_df['month'] = features_df['date'].dt.month
            features_df['day_of_year'] = features_df['date'].dt.dayofyear
        
        # Season features
        if 'season' in features_df.columns:
            features_df['season_phase'] = features_df['month'].apply(self._categorize_season_phase)
        
        # Odds-based features
        odds_columns = [col for col in features_df.columns if any(book in col for book in ['pinnacle', 'draftkings'])]
        
        for book in ['pinnacle', 'draftkings']:
            # Moneyline features
            open_ml_home = f'{book}_open_ml_home'
            close_ml_home = f'{book}_close_ml_home'
            
            if open_ml_home in features_df.columns and close_ml_home in features_df.columns:
                features_df[f'{book}_ml_movement'] = features_df[close_ml_home] - features_df[open_ml_home]
            
            # Spread features
            open_spread = f'{book}_open_spread'
            close_spread = f'{book}_close_spread'
            
            if open_spread in features_df.columns and close_spread in features_df.columns:
                features_df[f'{book}_spread_movement'] = features_df[close_spread] - features_df[open_spread]
            
            # Total features
            open_total = f'{book}_open_total'
            close_total = f'{book}_close_total'
            
            if open_total in features_df.columns and close_total in features_df.columns:
                features_df[f'{book}_total_movement'] = features_df[close_total] - features_df[open_total]
        
        self.logger.info(f"Features dataset created: {len(features_df)} rows, {len(features_df.columns)} columns")
        return features_df
    
    def _categorize_season_phase(self, month: int) -> str:
        """Categorize month into season phase.
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season phase string
        """
        if month in [11, 12]:
            return 'early_season'
        elif month in [1, 2]:
            return 'conference_play'
        elif month in [3, 4]:
            return 'postseason'
        else:
            return 'offseason'
    
    def validate_merged_data(self, merged_df: pd.DataFrame) -> Dict[str, Any]:
        """Validate the merged dataset for data quality.
        
        Args:
            merged_df: Merged DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        if merged_df.empty:
            return {'valid': False, 'errors': ['Dataset is empty']}
        
        validation_results = {
            'valid': True,
            'total_rows': len(merged_df),
            'total_columns': len(merged_df.columns),
            'missing_data': {},
            'data_types': {},
            'warnings': []
        }
        
        # Check for missing data
        for column in merged_df.columns:
            missing_count = merged_df[column].isnull().sum()
            missing_pct = (missing_count / len(merged_df)) * 100
            
            validation_results['missing_data'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Flag high missing data
            if missing_pct > 50:
                validation_results['warnings'].append(f"High missing data in {column}: {missing_pct:.1f}%")
        
        # Check data types
        for column in merged_df.columns:
            validation_results['data_types'][column] = str(merged_df[column].dtype)
        
        # Check for critical missing data
        critical_columns = ['game_id', 'date', 'home_team', 'away_team', 'home_score', 'away_score']
        critical_missing = merged_df[critical_columns].isnull().any(axis=1).sum()
        
        if critical_missing > 0:
            validation_results['valid'] = False
            validation_results['errors'] = [f"{critical_missing} rows missing critical data"]
        
        return validation_results
    
    def save_processed_data(self, processed_df: pd.DataFrame, filename: str) -> str:
        """Save processed data to CSV file.
        
        Args:
            processed_df: Processed DataFrame to save
            filename: Name of the file to save
            
        Returns:
            Path to saved file
        """
        safe_filename_str = safe_filename(filename)
        filepath = os.path.join(self.processed_data_dir, safe_filename_str)
        
        processed_df.to_csv(filepath, index=False)
        self.logger.info(f"Saved processed data to {filepath}")
        
        return filepath
    
    def run_full_etl_pipeline(self, season: Optional[int] = None, days_back: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete ETL pipeline.
        
        Args:
            season: Specific season to process (if None, processes all)
            days_back: Days back for odds data (if None, uses config default)
            
        Returns:
            Dictionary with pipeline results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("Starting full ETL pipeline...")
            
            # Extract: Load raw data
            games_df = self.load_raw_games_data(season)
            odds_df = self.load_raw_odds_data(days_back)
            
            if games_df.empty:
                return {
                    'success': False,
                    'error': 'No games data available',
                    'duration': datetime.now() - start_time
                }
            
            # Transform: Clean and merge data
            cleaned_games = self.clean_games_data(games_df)
            cleaned_odds = self.clean_odds_data(odds_df)
            
            merged_df = self.merge_games_and_odds(cleaned_games, cleaned_odds)
            
            # Validate merged data
            validation_results = self.validate_merged_data(merged_df)
            
            if not validation_results['valid']:
                self.logger.error(f"Data validation failed: {validation_results.get('errors', [])}")
                return {
                    'success': False,
                    'error': 'Data validation failed',
                    'validation_results': validation_results,
                    'duration': datetime.now() - start_time
                }
            
            # Create features dataset
            features_df = self.create_features_dataset(merged_df)
            
            # Load: Save processed data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save merged dataset
            merged_filename = f"games_odds_{timestamp}.csv"
            merged_path = self.save_processed_data(merged_df, merged_filename)
            
            # Save features dataset
            features_filename = f"features_{timestamp}.csv"
            features_path = self.save_processed_data(features_df, features_filename)
            
            # Save to database if available
            db_rows = 0
            if self.db_manager:
                try:
                    db_rows = self.db_manager.insert_or_update(
                        'games_odds',
                        merged_df,
                        conflict_columns=['game_id']
                    )
                    self.logger.info(f"Saved {db_rows} rows to database")
                except Exception as e:
                    self.logger.error(f"Error saving to database: {e}")
            
            duration = datetime.now() - start_time
            
            return {
                'success': True,
                'games_count': len(cleaned_games),
                'odds_count': len(cleaned_odds),
                'merged_count': len(merged_df),
                'features_count': len(features_df),
                'merged_file': merged_path,
                'features_file': features_path,
                'db_rows': db_rows,
                'validation_results': validation_results,
                'duration': duration
            }
            
        except Exception as e:
            duration = datetime.now() - start_time
            self.logger.error(f"ETL pipeline failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }


def create_etl_processor(config_path: str = "config.yaml") -> CBBDataETL:
    """Create and return an ETL processor instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        CBBDataETL instance
    """
    config = ConfigManager(config_path)
    return CBBDataETL(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the ETL pipeline
    try:
        etl_processor = create_etl_processor()
        
        # Run full pipeline
        result = etl_processor.run_full_etl_pipeline(season=2024, days_back=30)
        
        print(f"ETL pipeline result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")