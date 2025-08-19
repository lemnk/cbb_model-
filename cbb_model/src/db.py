"""
Database management for the CBB Betting ML System.

This module handles:
- PostgreSQL connection management
- Database schema creation
- Data insertion and updates
- Connection pooling and optimization
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, Text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from .utils import get_logger, ConfigManager


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, config: ConfigManager):
        """Initialize database manager.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('database')
        self.engine = None
        self.metadata = MetaData()
        self.Session = None
        self._setup_connection()
        self._define_schema()
    
    def _setup_connection(self):
        """Set up database connection."""
        try:
            # Get database configuration
            db_config = self.config.get('database', {})
            
            # Build connection string
            connection_string = (
                f"postgresql://{db_config.get('user')}:{db_config.get('password')}"
                f"@{db_config.get('host')}:{db_config.get('port')}/{db_config.get('name')}"
            )
            
            # Create engine with connection pooling
            self.engine = create_engine(
                connection_string,
                poolclass=QueuePool,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False  # Set to True for SQL debugging
            )
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            self.logger.info("Database connection established successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def _define_schema(self):
        """Define database schema tables."""
        
        # Games table - stores NCAA basketball game data
        self.games_table = Table(
            'games',
            self.metadata,
            Column('game_id', String(50), primary_key=True),
            Column('date', DateTime, nullable=False, index=True),
            Column('season', Integer, nullable=False, index=True),
            Column('home_team', String(100), nullable=False),
            Column('away_team', String(100), nullable=False),
            Column('home_score', Integer),
            Column('away_score', Integer),
            Column('home_team_id', String(20)),
            Column('away_team_id', String(20)),
            Column('location', String(200)),
            Column('attendance', Integer),
            Column('overtime', Boolean, default=False),
            Column('home_pace', Float),
            Column('away_pace', Float),
            Column('home_efficiency', Float),
            Column('away_efficiency', Float),
            Column('home_possessions', Integer),
            Column('away_possessions', Integer),
            Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
            Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
        )
        
        # Odds table - stores betting odds data
        self.odds_table = Table(
            'odds',
            self.metadata,
            Column('odds_id', String(100), primary_key=True),
            Column('game_id', String(50), nullable=False, index=True),
            Column('book', String(50), nullable=False),  # Pinnacle, DraftKings, etc.
            Column('open_time', DateTime),
            Column('close_time', DateTime),
            Column('open_moneyline_home', Float),
            Column('open_moneyline_away', Float),
            Column('close_moneyline_home', Float),
            Column('close_moneyline_away', Float),
            Column('open_spread', Float),
            Column('close_spread', Float),
            Column('open_total', Float),
            Column('close_total', Float),
            Column('spread_side', String(10)),  # 'home' or 'away'
            Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
            Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
        )
        
        # Players table - stores player statistics
        self.players_table = Table(
            'players',
            self.metadata,
            Column('player_id', String(50), primary_key=True),
            Column('game_id', String(50), nullable=False, index=True),
            Column('team_id', String(20), nullable=False),
            Column('player_name', String(100), nullable=False),
            Column('minutes_played', Integer),
            Column('points', Integer),
            Column('rebounds', Integer),
            Column('assists', Integer),
            Column('steals', Integer),
            Column('blocks', Integer),
            Column('turnovers', Integer),
            Column('fouls', Integer),
            Column('field_goals_made', Integer),
            Column('field_goals_attempted', Integer),
            Column('three_pointers_made', Integer),
            Column('three_pointers_attempted', Integer),
            Column('free_throws_made', Integer),
            Column('free_throws_attempted', Integer),
            Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
            Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
        )
        
        # Teams table - stores team information
        self.teams_table = Table(
            'teams',
            self.metadata,
            Column('team_id', String(20), primary_key=True),
            Column('team_name', String(100), nullable=False),
            Column('conference', String(100)),
            Column('division', String(50)),
            Column('state', String(50)),
            Column('city', String(100)),
            Column('venue', String(200)),
            Column('venue_capacity', Integer),
            Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
            Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
        )
        
        # Games_odds table - merged view for ML features
        self.games_odds_table = Table(
            'games_odds',
            self.metadata,
            Column('game_id', String(50), primary_key=True),
            Column('date', DateTime, nullable=False, index=True),
            Column('season', Integer, nullable=False, index=True),
            Column('home_team', String(100), nullable=False),
            Column('away_team', String(100), nullable=False),
            Column('home_score', Integer),
            Column('away_score', Integer),
            Column('home_team_id', String(20)),
            Column('away_team_id', String(20)),
            Column('location', String(200)),
            Column('attendance', Integer),
            Column('overtime', Boolean, default=False),
            Column('home_pace', Float),
            Column('away_pace', Float),
            Column('home_efficiency', Float),
            Column('away_efficiency', Float),
            Column('home_possessions', Integer),
            Column('away_possessions', Integer),
            # Odds columns
            Column('pinnacle_open_ml_home', Float),
            Column('pinnacle_open_ml_away', Float),
            Column('pinnacle_close_ml_home', Float),
            Column('pinnacle_close_ml_away', Float),
            Column('pinnacle_open_spread', Float),
            Column('pinnacle_close_spread', Float),
            Column('pinnacle_open_total', Float),
            Column('pinnacle_close_total', Float),
            Column('draftkings_open_ml_home', Float),
            Column('draftkings_open_ml_away', Float),
            Column('draftkings_close_ml_home', Float),
            Column('draftkings_close_ml_away', Float),
            Column('draftkings_open_spread', Float),
            Column('draftkings_close_spread', Float),
            Column('draftkings_open_total', Float),
            Column('draftkings_close_total', Float),
            Column('created_at', DateTime, server_default=text('CURRENT_TIMESTAMP')),
            Column('updated_at', DateTime, server_default=text('CURRENT_TIMESTAMP'), onupdate=text('CURRENT_TIMESTAMP'))
        )
    
    def create_tables(self, drop_existing: bool = False):
        """Create all database tables.
        
        Args:
            drop_existing: Whether to drop existing tables before creating new ones
        """
        try:
            if drop_existing:
                self.logger.warning("Dropping existing tables...")
                self.metadata.drop_all(self.engine)
            
            self.logger.info("Creating database tables...")
            self.metadata.create_all(self.engine)
            self.logger.info("Database tables created successfully")
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    def insert_or_update(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame],
        conflict_columns: Optional[List[str]] = None
    ) -> int:
        """Insert or update data in the specified table.
        
        Args:
            table_name: Name of the table to insert/update
            data: Data to insert (dict, list of dicts, or DataFrame)
            conflict_columns: Columns to use for conflict resolution (ON CONFLICT)
            
        Returns:
            Number of rows affected
            
        Raises:
            ValueError: If table_name is invalid
            SQLAlchemyError: If database operation fails
        """
        # Validate table name
        valid_tables = ['games', 'odds', 'players', 'teams', 'games_odds']
        if table_name not in valid_tables:
            raise ValueError(f"Invalid table name: {table_name}. Must be one of {valid_tables}")
        
        # Convert DataFrame to list of dicts if needed
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        elif isinstance(data, dict):
            data = [data]
        
        if not data:
            self.logger.warning("No data to insert")
            return 0
        
        try:
            with self.Session() as session:
                rows_affected = 0
                
                for row in data:
                    # Handle conflict resolution if specified
                    if conflict_columns and table_name != 'games_odds':
                        # Use PostgreSQL's ON CONFLICT for upsert
                        table = getattr(self, f"{table_name}_table")
                        stmt = table.insert().values(**row)
                        
                        # Build conflict resolution
                        conflict_cols = [getattr(table.c, col) for col in conflict_columns if hasattr(table.c, col)]
                        if conflict_cols:
                            stmt = stmt.on_conflict_do_update(
                                index_elements=conflict_cols,
                                set_=row
                            )
                        
                        result = session.execute(stmt)
                        rows_affected += 1
                    else:
                        # Simple insert
                        table = getattr(self, f"{table_name}_table")
                        stmt = table.insert().values(**row)
                        session.execute(stmt)
                        rows_affected += 1
                
                session.commit()
                self.logger.info(f"Successfully inserted/updated {rows_affected} rows in {table_name}")
                return rows_affected
                
        except IntegrityError as e:
            self.logger.error(f"Integrity error while inserting data: {e}")
            session.rollback()
            raise
        except SQLAlchemyError as e:
            self.logger.error(f"Database error while inserting data: {e}")
            session.rollback()
            raise
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Execute a raw SQL query and return results as DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params=params)
                return result
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to execute query: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a table's structure.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table information
        """
        try:
            query = """
                SELECT 
                    column_name,
                    data_type,
                    is_nullable,
                    column_default
                FROM information_schema.columns 
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            
            result = self.execute_query(query, {'table_name': table_name})
            return result.to_dict('records')
            
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get table info: {e}")
            raise
    
    def get_row_count(self, table_name: str) -> int:
        """Get the number of rows in a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Number of rows
        """
        try:
            query = f"SELECT COUNT(*) as count FROM {table_name}"
            result = self.execute_query(query)
            return result['count'].iloc[0]
        except SQLAlchemyError as e:
            self.logger.error(f"Failed to get row count: {e}")
            raise
    
    def close(self):
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connections closed")


def create_database_manager(config_path: str = "config.yaml") -> DatabaseManager:
    """Create and return a database manager instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DatabaseManager instance
    """
    config = ConfigManager(config_path)
    return DatabaseManager(config)


# Example usage and testing
if __name__ == "__main__":
    # Test database connection and table creation
    try:
        db_manager = create_database_manager()
        db_manager.create_tables()
        
        # Test table info
        for table in ['games', 'odds', 'players', 'teams']:
            count = db_manager.get_row_count(table)
            print(f"Table {table}: {count} rows")
        
        db_manager.close()
        
    except Exception as e:
        print(f"Error: {e}")