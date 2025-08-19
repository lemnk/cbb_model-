"""
Feature Engineering Utilities for CBB Betting ML System.

This module provides common utility functions for feature engineering:
- Rolling averages and trends
- Momentum calculations
- Run-length encoding
- Line drift calculations
- Feature validation and quality checks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging

from ..utils import get_logger, ConfigManager


class FeatureUtils:
    """Utility functions for feature engineering."""
    
    def __init__(self, config: ConfigManager):
        """Initialize the feature utilities.
        
        Args:
            config: Configuration manager instance
        """
        self.config = config
        self.logger = get_logger('feature_utils')
        
        # Get configuration
        self.rolling_windows = self.config.get('features.rolling_windows', [3, 5, 10, 20])
        self.momentum_alpha = self.config.get('features.momentum.alpha', 0.7)
        self.momentum_beta = self.config.get('features.momentum.beta', 0.3)
    
    def calculate_rolling_averages(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        group_by: Optional[str] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling averages for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to calculate rolling averages for
            group_by: Column to group by (e.g., 'team_id')
            windows: List of window sizes (uses config default if None)
            
        Returns:
            DataFrame with rolling average features added
        """
        if windows is None:
            windows = self.rolling_windows
        
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                self.logger.warning(f"Column {col} not found, skipping")
                continue
            
            for window in windows:
                feature_name = f"{col}_rolling_{window}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                else:
                    result_df[feature_name] = result_df[col].rolling(
                        window=window, min_periods=1
                    ).mean()
        
        return result_df
    
    def compute_momentum_index(
        self, 
        score_change: Union[float, pd.Series], 
        possession_change: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """Compute momentum index: M_t = α * Δscore_t + β * Δpossessions_t.
        
        Args:
            score_change: Change in score differential
            possession_change: Change in possession count
            
        Returns:
            Momentum index value(s)
        """
        return self.momentum_alpha * score_change + self.momentum_beta * possession_change
    
    def encode_run_lengths(self, series: pd.Series) -> pd.Series:
        """Encode run lengths for a boolean series.
        
        Args:
            series: Boolean series to encode
            
        Returns:
            Series with run lengths
        """
        # Group consecutive True values and count them
        groups = (series != series.shift()).cumsum()
        run_lengths = series.groupby(groups).cumsum()
        
        # Reset to 0 when False
        run_lengths = run_lengths.where(series, 0)
        
        return run_lengths
    
    def calculate_line_drift(
        self, 
        open_line: Union[float, pd.Series], 
        close_line: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """Calculate line drift between open and close.
        
        Args:
            open_line: Opening line value
            close_line: Closing line value
            
        Returns:
            Line drift value(s)
        """
        return close_line - open_line
    
    def compute_implied_probability_edge(
        self, 
        model_prob: Union[float, pd.Series], 
        market_prob: Union[float, pd.Series]
    ) -> Union[float, pd.Series]:
        """Compute implied probability edge between model and market.
        
        Args:
            model_prob: Model's predicted probability
            market_prob: Market's implied probability
            
        Returns:
            Probability edge (positive = overlay, negative = underlay)
        """
        return model_prob - market_prob
    
    def moneyline_to_probability(self, moneyline: Union[float, pd.Series]) -> Union[float, pd.Series]:
        """Convert moneyline odds to implied probability.
        
        Args:
            moneyline: Moneyline odds
            
        Returns:
            Implied probability
        """
        def convert_ml_to_prob(ml):
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        
        if isinstance(moneyline, pd.Series):
            return moneyline.apply(convert_ml_to_prob)
        else:
            return convert_ml_to_prob(moneyline)
    
    def calculate_rolling_trends(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        group_by: Optional[str] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling trend coefficients for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to calculate trends for
            group_by: Column to group by
            windows: List of window sizes
            
        Returns:
            DataFrame with trend features added
        """
        if windows is None:
            windows = self.rolling_windows
        
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
            
            for window in windows:
                feature_name = f"{col}_trend_{window}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].rolling(
                        window=window, min_periods=2
                    ).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    ).reset_index(0, drop=True)
                else:
                    result_df[feature_name] = result_df[col].rolling(
                        window=window, min_periods=2
                    ).apply(
                        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                    )
        
        return result_df
    
    def calculate_rolling_volatility(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        group_by: Optional[str] = None,
        windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Calculate rolling volatility (standard deviation) for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to calculate volatility for
            group_by: Column to group by
            windows: List of window sizes
            
        Returns:
            DataFrame with volatility features added
        """
        if windows is None:
            windows = self.rolling_windows
        
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
            
            for window in windows:
                feature_name = f"{col}_volatility_{window}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].rolling(
                        window=window, min_periods=1
                    ).std().reset_index(0, drop=True)
                else:
                    result_df[feature_name] = result_df[col].rolling(
                        window=window, min_periods=1
                    ).std()
                
                # Fill NaN values with 0
                result_df[feature_name] = result_df[feature_name].fillna(0)
        
        return result_df
    
    def create_lag_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        lags: List[int],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Create lag features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create lags for
            lags: List of lag periods
            group_by: Column to group by
            
        Returns:
            DataFrame with lag features added
        """
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
            
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].shift(lag)
                else:
                    result_df[feature_name] = result_df[col].shift(lag)
        
        return result_df
    
    def create_lead_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        leads: List[int],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Create lead features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create leads for
            leads: List of lead periods
            group_by: Column to group by
            
        Returns:
            DataFrame with lead features added
        """
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
            
            for lead in leads:
                feature_name = f"{col}_lead_{lead}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].shift(-lead)
                else:
                    result_df[feature_name] = result_df[col].shift(-lead)
        
        return result_df
    
    def calculate_percentile_features(
        self, 
        df: pd.DataFrame, 
        columns: List[str], 
        group_by: Optional[str] = None,
        percentiles: List[float] = [25, 50, 75]
    ) -> pd.DataFrame:
        """Calculate percentile-based features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to calculate percentiles for
            group_by: Column to group by
            percentiles: List of percentiles to calculate
            
        Returns:
            DataFrame with percentile features added
        """
        result_df = df.copy()
        
        for col in columns:
            if col not in result_df.columns:
                continue
            
            for p in percentiles:
                feature_name = f"{col}_p{p}"
                
                if group_by:
                    result_df[feature_name] = result_df.groupby(group_by)[col].transform(
                        lambda x: x.quantile(p/100)
                    )
                else:
                    result_df[feature_name] = result_df[col].quantile(p/100)
        
        return result_df
    
    def validate_feature_set(
        self, 
        df: pd.DataFrame, 
        required_columns: Optional[List[str]] = None,
        max_missing_pct: float = 0.5
    ) -> Dict[str, Any]:
        """Validate feature set for quality and completeness.
        
        Args:
            df: Feature DataFrame to validate
            required_columns: List of required columns
            max_missing_pct: Maximum allowed missing data percentage
            
        Returns:
            Dictionary with validation results
        """
        if df.empty:
            return {'valid': False, 'errors': ['DataFrame is empty']}
        
        validation_results = {
            'valid': True,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'warnings': [],
            'errors': []
        }
        
        # Check for missing data
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            
            validation_results['missing_data'][column] = {
                'count': missing_count,
                'percentage': missing_pct
            }
            
            # Flag high missing data
            if missing_pct > max_missing_pct * 100:
                validation_results['warnings'].append(
                    f"High missing data in {column}: {missing_pct:.1f}%"
                )
        
        # Check data types
        for column in df.columns:
            validation_results['data_types'][column] = str(df[column].dtype)
        
        # Check required columns if specified
        if required_columns:
            missing_required = set(required_columns) - set(df.columns)
            if missing_required:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Missing required columns: {missing_required}"
                )
        
        # Check for infinite values
        infinite_cols = []
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32']:
                if np.isinf(df[col]).any():
                    infinite_cols.append(col)
        
        if infinite_cols:
            validation_results['warnings'].append(
                f"Columns with infinite values: {infinite_cols}"
            )
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results['warnings'].append(
                f"Found {duplicate_count} duplicate rows"
            )
        
        return validation_results
    
    def remove_low_variance_features(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.01,
        exclude_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Remove features with low variance.
        
        Args:
            df: Input DataFrame
            threshold: Variance threshold (features below this will be removed)
            exclude_columns: Columns to exclude from variance check
            
        Returns:
            DataFrame with low variance features removed
        """
        if exclude_columns is None:
            exclude_columns = []
        
        result_df = df.copy()
        
        # Calculate variance for numeric columns
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        columns_to_check = [col for col in numeric_columns if col not in exclude_columns]
        
        low_variance_columns = []
        
        for col in columns_to_check:
            variance = result_df[col].var()
            if variance < threshold:
                low_variance_columns.append(col)
        
        if low_variance_columns:
            self.logger.info(f"Removing {len(low_variance_columns)} low variance features")
            result_df = result_df.drop(columns=low_variance_columns)
        
        return result_df
    
    def create_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a summary of all features in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with feature summary
        """
        summary_data = []
        
        for col in df.columns:
            col_data = {
                'feature_name': col,
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_values': df[col].nunique()
            }
            
            # Add numeric-specific statistics
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                col_data.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            summary_data.append(col_data)
        
        return pd.DataFrame(summary_data)


def create_feature_utils(config_path: str = "config.yaml") -> FeatureUtils:
    """Create and return a feature utils instance.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FeatureUtils instance
    """
    config = ConfigManager(config_path)
    return FeatureUtils(config)


# Standalone utility functions for easy import
def calculate_rolling_averages(
    df: pd.DataFrame, 
    columns: List[str], 
    group_by: Optional[str] = None,
    windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """Calculate rolling averages for specified columns."""
    utils = FeatureUtils(ConfigManager())
    return utils.calculate_rolling_averages(df, columns, group_by, windows)


def compute_momentum_index(
    score_change: Union[float, pd.Series], 
    possession_change: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Compute momentum index: M_t = α * Δscore_t + β * Δpossessions_t."""
    utils = FeatureUtils(ConfigManager())
    return utils.compute_momentum_index(score_change, possession_change)


def encode_run_lengths(series: pd.Series) -> pd.Series:
    """Encode run lengths for a boolean series."""
    utils = FeatureUtils(ConfigManager())
    return utils.encode_run_lengths(series)


def calculate_line_drift(
    open_line: Union[float, pd.Series], 
    close_line: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Calculate line drift between open and close."""
    utils = FeatureUtils(ConfigManager())
    return utils.calculate_line_drift(open_line, close_line)


def compute_implied_probability_edge(
    model_prob: Union[float, pd.Series], 
    market_prob: Union[float, pd.Series]
) -> Union[float, pd.Series]:
    """Compute implied probability edge between model and market."""
    utils = FeatureUtils(ConfigManager())
    return utils.compute_implied_probability_edge(model_prob, market_prob)


def validate_feature_set(
    df: pd.DataFrame, 
    required_columns: Optional[List[str]] = None,
    max_missing_pct: float = 0.5
) -> Dict[str, Any]:
    """Validate feature set for quality and completeness."""
    utils = FeatureUtils(ConfigManager())
    return utils.validate_feature_set(df, required_columns, max_missing_pct)


# Example usage and testing
if __name__ == "__main__":
    # Test the feature utilities
    try:
        utils = create_feature_utils()
        
        # Create sample data
        sample_df = pd.DataFrame({
            'team_id': ['team_1', 'team_1', 'team_1', 'team_2', 'team_2', 'team_2'],
            'score': [85, 88, 92, 78, 82, 85],
            'momentum': [0.5, 0.8, 1.2, -0.3, 0.1, 0.4]
        })
        
        # Test rolling averages
        result = utils.calculate_rolling_averages(sample_df, ['score', 'momentum'], 'team_id')
        print("Rolling averages result:")
        print(result)
        
        # Test momentum index
        momentum = utils.compute_momentum_index(2.0, 1.0)
        print(f"\nMomentum index: {momentum}")
        
        # Test validation
        validation = utils.validate_feature_set(result)
        print(f"\nValidation result: {validation['valid']}")
        
    except Exception as e:
        print(f"Error: {e}")