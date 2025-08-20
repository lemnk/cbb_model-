"""
Utility functions for the CBB Betting ML System.

This module provides shared functionality including:
- Configuration management
- Logging setup
- Date/time helpers
- Data validation utilities
"""

import os
import yaml
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd


class ConfigManager:
    """Manages configuration loading and access."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is malformed
        """
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Error parsing configuration file: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation like 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size_mb: int = 10,
    backup_count: int = 5
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size_mb: Maximum log file size in MB
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    # Configure logging
    logger = logging.getLogger('cbb_model')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name (optional)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f'cbb_model.{name}')
    return logging.getLogger('cbb_model')


def parse_date_range(
    start_date: Union[str, datetime],
    end_date: Union[str, datetime]
) -> tuple[datetime, datetime]:
    """Parse and validate date range.
    
    Args:
        start_date: Start date (string or datetime)
        end_date: End date (string or datetime)
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
        
    Raises:
        ValueError: If dates are invalid or start_date > end_date
    """
    # Convert strings to datetime if needed
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Validate date range
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    
    return start_date, end_date


def get_season_dates(season: int) -> tuple[datetime, datetime]:
    """Get start and end dates for an NCAA basketball season.
    
    Args:
        season: NCAA season year (e.g., 2024 for 2023-24 season)
        
    Returns:
        Tuple of (season_start, season_end) as datetime objects
    """
    # NCAA season typically runs from November to April
    season_start = datetime(season - 1, 11, 1)  # November 1st
    season_end = datetime(season, 4, 30)        # April 30th
    
    return season_start, season_end


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    data_types: Optional[Dict[str, str]] = None
) -> bool:
    """Validate DataFrame structure and data types.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        data_types: Dictionary mapping column names to expected data types
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If validation fails
    """
    # Check required columns
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check data types if specified
    if data_types:
        for col, expected_type in data_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if expected_type not in actual_type:
                    raise ValueError(
                        f"Column {col} has type {actual_type}, expected {expected_type}"
                    )
    
    return True


def safe_filename(filename: str) -> str:
    """Convert filename to safe filesystem name.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename for filesystem
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    return filename


def create_backup_dir(backup_dir: str) -> str:
    """Create backup directory with timestamp.
    
    Args:
        backup_dir: Base backup directory path
        
    Returns:
        Path to created backup directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"backup_{timestamp}")
    
    if not os.path.exists(backup_path):
        os.makedirs(backup_path)
    
    return backup_path


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# Convenience functions for common logging patterns
def log_info(message: str, logger: Optional[logging.Logger] = None):
    """Log info message."""
    if logger is None:
        logger = get_logger()
    logger.info(message)


def log_warning(message: str, logger: Optional[logging.Logger] = None):
    """Log warning message."""
    if logger is None:
        logger = get_logger()
    logger.warning(message)


def log_error(message: str, logger: Optional[logging.Logger] = None):
    """Log error message."""
    if logger is None:
        logger = get_logger()
    logger.error(message)


def log_debug(message: str, logger: Optional[logging.Logger] = None):
    """Log debug message."""
    if logger is None:
        logger = get_logger()
    logger.debug(message)