"""
NCAA Basketball Games Data Scraper.

This module scrapes historical NCAA basketball data from sports-reference.com including:
- Game results and scores
- Team statistics
- Player performance data
- Advanced metrics (pace, efficiency, etc.)
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
import re

from .utils import get_logger, ConfigManager, safe_filename, get_season_dates
from .db import DatabaseManager


class NCAAGamesScraper:
    """Scrapes NCAA basketball games data from sports-reference.com."""
    
    def __init__(self, config: ConfigManager, db_manager: Optional[DatabaseManager] = None):
        """Initialize the scraper.
        
        Args:
            config: Configuration manager instance
            db_manager: Database manager instance (optional)
        """
        self.config = config
        self.logger = get_logger('games_scraper')
        self.db_manager = db_manager
        
        # Get configuration
        self.base_url = self.config.get('apis.sports_reference.base_url')
        self.rate_limit_delay = self.config.get('apis.sports_reference.rate_limit_delay', 1.0)
        self.raw_data_dir = self.config.get('data_collection.raw_data_dir', 'data/raw')
        
        # Create raw data directory if it doesn't exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def _make_request(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        """Make HTTP request with retry logic and rate limiting.
        
        Args:
            url: URL to request
            retries: Number of retry attempts
            
        Returns:
            BeautifulSoup object if successful, None otherwise
        """
        for attempt in range(retries):
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                
                return BeautifulSoup(response.content, 'html.parser')
                
            except requests.RequestException as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to fetch {url} after {retries} attempts")
                    return None
        
        return None
    
    def get_team_schedule_url(self, team_id: str, season: int) -> str:
        """Generate team schedule URL for sports-reference.com.
        
        Args:
            team_id: Team identifier (e.g., 'duke', 'kentucky')
            season: Season year
            
        Returns:
            Full URL for team schedule
        """
        return f"{self.base_url}/schools/{team_id}/{season}/schedule.html"
    
    def get_game_box_score_url(self, game_url: str) -> str:
        """Convert game URL to box score URL.
        
        Args:
            game_url: Game URL from schedule
            
        Returns:
            Box score URL
        """
        # Replace schedule with boxscore in URL
        return game_url.replace('schedule', 'boxscore')
    
    def parse_team_schedule(self, soup: BeautifulSoup, team_id: str, season: int) -> List[Dict[str, Any]]:
        """Parse team schedule page to extract game information.
        
        Args:
            soup: BeautifulSoup object of schedule page
            team_id: Team identifier
            season: Season year
            
        Returns:
            List of game dictionaries
        """
        games = []
        
        try:
            # Find the schedule table
            schedule_table = soup.find('table', {'id': 'schedule'})
            if not schedule_table:
                self.logger.warning(f"No schedule table found for {team_id} {season}")
                return games
            
            # Find all game rows
            game_rows = schedule_table.find('tbody').find_all('tr')
            
            for row in game_rows:
                # Skip rows without game data
                if not row.get('class') or 'thead' in row.get('class'):
                    continue
                
                try:
                    game_data = self._parse_game_row(row, team_id, season)
                    if game_data:
                        games.append(game_data)
                        
                except Exception as e:
                    self.logger.warning(f"Error parsing game row: {e}")
                    continue
            
            self.logger.info(f"Parsed {len(games)} games for {team_id} {season}")
            
        except Exception as e:
            self.logger.error(f"Error parsing schedule for {team_id} {season}: {e}")
        
        return games
    
    def _parse_game_row(self, row, team_id: str, season: int) -> Optional[Dict[str, Any]]:
        """Parse individual game row from schedule table.
        
        Args:
            row: BeautifulSoup row element
            team_id: Team identifier
            season: Season year
            
        Returns:
            Game data dictionary or None if invalid
        """
        try:
            cells = row.find_all('td')
            if len(cells) < 10:
                return None
            
            # Extract basic game info
            date_str = cells[0].get_text(strip=True)
            if not date_str or date_str == 'Date':
                return None
            
            # Parse date
            try:
                game_date = datetime.strptime(f"{date_str} {season}", "%b %d, %Y")
            except ValueError:
                # Handle season spanning dates (e.g., Nov 2023 is in 2024 season)
                if date_str.startswith(('Nov', 'Dec')):
                    game_date = datetime.strptime(f"{date_str} {season-1}", "%b %d, %Y")
                else:
                    game_date = datetime.strptime(f"{date_str} {season}", "%b %d, %Y")
            
            # Extract opponent and result
            opponent_cell = cells[1]
            opponent = opponent_cell.get_text(strip=True)
            
            # Check if it's a home or away game
            is_home = 'home' in opponent_cell.get('class', []) if opponent_cell.get('class') else False
            
            # Extract scores
            team_score = cells[2].get_text(strip=True)
            opponent_score = cells[3].get_text(strip=True)
            
            # Skip games without scores (future games)
            if not team_score or not opponent_score or team_score == 'W' or team_score == 'L':
                return None
            
            # Convert scores to integers
            try:
                team_score = int(team_score)
                opponent_score = int(opponent_score)
            except ValueError:
                return None
            
            # Determine home/away teams and scores
            if is_home:
                home_team = team_id
                away_team = opponent
                home_score = team_score
                away_score = opponent_score
            else:
                home_team = opponent
                away_team = team_id
                home_score = opponent_score
                away_score = team_score
            
            # Generate game ID
            game_id = f"{season}_{home_team}_{away_team}_{game_date.strftime('%Y%m%d')}"
            
            # Extract additional game info
            location = cells[4].get_text(strip=True) if len(cells) > 4 else None
            attendance = cells[5].get_text(strip=True) if len(cells) > 5 else None
            
            # Convert attendance to integer if possible
            if attendance and attendance != 'N/A':
                try:
                    attendance = int(attendance.replace(',', ''))
                except ValueError:
                    attendance = None
            
            # Check for overtime
            overtime = 'OT' in cells[2].get_text() or 'OT' in cells[3].get_text()
            
            game_data = {
                'game_id': game_id,
                'date': game_date,
                'season': season,
                'home_team': home_team,
                'away_team': away_team,
                'home_score': home_score,
                'away_score': away_score,
                'location': location,
                'attendance': attendance,
                'overtime': overtime,
                'home_team_id': home_team,
                'away_team_id': away_team
            }
            
            return game_data
            
        except Exception as e:
            self.logger.warning(f"Error parsing game row: {e}")
            return None
    
    def get_team_list(self, season: int) -> List[str]:
        """Get list of all NCAA teams for a given season.
        
        Args:
            season: Season year
            
        Returns:
            List of team identifiers
        """
        teams = []
        
        try:
            # Use the main teams page
            url = f"{self.base_url}/schools/"
            soup = self._make_request(url)
            
            if not soup:
                return teams
            
            # Find team links
            team_links = soup.find_all('a', href=re.compile(r'/schools/[^/]+/$'))
            
            for link in team_links:
                team_id = link['href'].split('/')[-2]
                if team_id and team_id != 'schools':
                    teams.append(team_id)
            
            self.logger.info(f"Found {len(teams)} teams for {season}")
            
        except Exception as e:
            self.logger.error(f"Error getting team list: {e}")
        
        return teams
    
    def scrape_season_games(self, season: int, max_teams: Optional[int] = None) -> pd.DataFrame:
        """Scrape all games for a given season.
        
        Args:
            season: Season year to scrape
            max_teams: Maximum number of teams to scrape (for testing)
            
        Returns:
            DataFrame with all games for the season
        """
        self.logger.info(f"Starting to scrape {season} season games...")
        
        # Get list of teams
        teams = self.get_team_list(season)
        
        if max_teams:
            teams = teams[:max_teams]
            self.logger.info(f"Limited to {max_teams} teams for testing")
        
        all_games = []
        
        # Scrape each team's schedule
        for team_id in tqdm(teams, desc=f"Scraping {season} season"):
            try:
                # Get team schedule
                schedule_url = self.get_team_schedule_url(team_id, season)
                soup = self._make_request(schedule_url)
                
                if soup:
                    team_games = self.parse_team_schedule(soup, team_id, season)
                    all_games.extend(team_games)
                
                # Small delay between teams
                time.sleep(0.5)
                
            except Exception as e:
                self.logger.warning(f"Error scraping {team_id}: {e}")
                continue
        
        # Convert to DataFrame and remove duplicates
        if all_games:
            df = pd.DataFrame(all_games)
            df = df.drop_duplicates(subset=['game_id'])
            
            # Sort by date
            df = df.sort_values('date')
            
            self.logger.info(f"Scraped {len(df)} unique games for {season} season")
            
            return df
        else:
            self.logger.warning(f"No games scraped for {season} season")
            return pd.DataFrame()
    
    def save_games_to_csv(self, games_df: pd.DataFrame, season: int) -> str:
        """Save games data to CSV file.
        
        Args:
            games_df: DataFrame with games data
            season: Season year
            
        Returns:
            Path to saved CSV file
        """
        filename = safe_filename(f"games_{season}.csv")
        filepath = os.path.join(self.raw_data_dir, filename)
        
        games_df.to_csv(filepath, index=False)
        self.logger.info(f"Saved {len(games_df)} games to {filepath}")
        
        return filepath
    
    def save_games_to_db(self, games_df: pd.DataFrame) -> int:
        """Save games data to database.
        
        Args:
            games_df: DataFrame with games data
            
        Returns:
            Number of rows inserted/updated
        """
        if self.db_manager is None:
            self.logger.warning("No database manager provided, skipping database save")
            return 0
        
        try:
            rows_affected = self.db_manager.insert_or_update(
                'games',
                games_df,
                conflict_columns=['game_id']
            )
            return rows_affected
            
        except Exception as e:
            self.logger.error(f"Error saving games to database: {e}")
            return 0
    
    def scrape_and_save_season(self, season: int, max_teams: Optional[int] = None) -> Dict[str, Any]:
        """Complete workflow: scrape season and save to both CSV and database.
        
        Args:
            season: Season year to scrape
            max_teams: Maximum number of teams to scrape (for testing)
            
        Returns:
            Dictionary with results summary
        """
        start_time = time.time()
        
        try:
            # Scrape games
            games_df = self.scrape_season_games(season, max_teams)
            
            if games_df.empty:
                return {
                    'success': False,
                    'season': season,
                    'games_count': 0,
                    'error': 'No games scraped'
                }
            
            # Save to CSV
            csv_path = self.save_games_to_csv(games_df, season)
            
            # Save to database
            db_rows = self.save_games_to_db(games_df)
            
            duration = time.time() - start_time
            
            return {
                'success': True,
                'season': season,
                'games_count': len(games_df),
                'csv_path': csv_path,
                'db_rows': db_rows,
                'duration_seconds': duration
            }
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"Error scraping {season} season: {e}")
            
            return {
                'success': False,
                'season': season,
                'games_count': 0,
                'error': str(e),
                'duration_seconds': duration
            }
    
    def close(self):
        """Clean up resources."""
        if self.session:
            self.session.close()


def create_games_scraper(config_path: str = "config.yaml") -> NCAAGamesScraper:
    """Create and return a games scraper instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        NCAAGamesScraper instance
    """
    config = ConfigManager(config_path)
    return NCAAGamesScraper(config)


# Example usage and testing
if __name__ == "__main__":
    # Test the scraper with a small sample
    try:
        scraper = create_games_scraper()
        
        # Test with just a few teams for the 2024 season
        result = scraper.scrape_and_save_season(2024, max_teams=5)
        
        print(f"Scraping result: {result}")
        
        scraper.close()
        
    except Exception as e:
        print(f"Error: {e}")