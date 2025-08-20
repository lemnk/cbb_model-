"""
CBB Betting ML System - Source Package

This package contains the core functionality for the CBB betting ML system.
"""

__version__ = "1.0.0"
__author__ = "CBB Betting ML Team"

# Import main classes for easy access
from .utils import ConfigManager, setup_logging, get_logger
from .db import DatabaseManager, create_database_manager
from .scrape_games import NCAAGamesScraper, create_games_scraper
from .scrape_odds import OddsDataCollector, create_odds_collector
from .etl import CBBDataETL, create_etl_processor
from .features import CBBFeatureEngineer, create_feature_engineer
from .train import CBBModelTrainer, create_model_trainer

__all__ = [
    # Core utilities
    'ConfigManager',
    'setup_logging', 
    'get_logger',
    
    # Database management
    'DatabaseManager',
    'create_database_manager',
    
    # Data collection
    'NCAAGamesScraper',
    'create_games_scraper',
    'OddsDataCollector',
    'create_odds_collector',
    
    # Data processing
    'CBBDataETL',
    'create_etl_processor',
    
    # Feature engineering (Phase 2)
    'CBBFeatureEngineer',
    'create_feature_engineer',
    
    # Model training (Phase 3)
    'CBBModelTrainer',
    'create_model_trainer',
]