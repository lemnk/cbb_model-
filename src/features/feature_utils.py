"""
Feature Utilities for NCAA CBB Betting ML System.

This module provides common utility functions for feature engineering including:
- Data normalization and scaling
- Feature validation and quality assessment
- Common mathematical operations
- Data transformation helpers
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime, timedelta


def normalize_series(series, method="zscore"):
    """Normalize a Pandas Series by z-score or min-max.
    
    Args:
        series: Pandas Series to normalize
        method: Normalization method ('zscore' or 'minmax')
        
    Returns:
        Normalized Series
    """
    if method == "zscore":
        return (series - series.mean()) / series.std()
    elif method == "minmax":
        return (series - series.min()) / (series.max() - series.min())
    else:
        raise ValueError("Method must be 'zscore' or 'minmax'")


class FeatureUtils:
    """Utility class for common feature engineering operations."""
    
    def __init__(self):
        """Initialize the feature utilities."""
        pass
    
    @staticmethod
    def calculate_rolling_averages(df: pd.DataFrame, column: str, windows: list = [3, 5, 10]) -> pd.DataFrame:
        """Calculate rolling averages for a given column.
        
        Args:
            df: Input DataFrame
            column: Column to calculate rolling averages for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling average columns added
        """
        result_df = df.copy()
        
        for window in windows:
            col_name = f"{column}_rolling_{window}"
            result_df[col_name] = df[column].rolling(window=window, min_periods=1).mean()
        
        return result_df
    
    @staticmethod
    def compute_momentum_index(score_change: pd.Series, possession_change: pd.Series, 
                              alpha: float = 0.7, beta: float = 0.3) -> pd.Series:
        """Compute momentum index: M_t = α * Δscore_t + β * Δpossessions_t.
        
        Args:
            score_change: Series of score changes
            possession_change: Series of possession changes
            alpha: Weight for score change (default: 0.7)
            beta: Weight for possession change (default: 0.3)
            
        Returns:
            Series of momentum index values
        """
        return alpha * score_change + beta * possession_change
    
    @staticmethod
    def encode_run_lengths(series: pd.Series) -> pd.Series:
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
    
    @staticmethod
    def calculate_line_drift(open_line: pd.Series, close_line: pd.Series) -> pd.Series:
        """Calculate line drift between open and close.
        
        Args:
            open_line: Series of opening line values
            close_line: Series of closing line values
            
        Returns:
            Series of line drift values
        """
        return close_line - open_line
    
    @staticmethod
    def compute_implied_probability_edge(model_prob: pd.Series, market_prob: pd.Series) -> pd.Series:
        """Compute implied probability edge between model and market.
        
        Args:
            model_prob: Model's predicted probabilities
            market_prob: Market's implied probabilities
            
        Returns:
            Series of probability edges (positive = overlay, negative = underlay)
        """
        return model_prob - market_prob
    
    @staticmethod
    def moneyline_to_probability(moneyline: pd.Series) -> pd.Series:
        """Convert moneyline odds to implied probability.
        
        Args:
            moneyline: Series of moneyline odds
            
        Returns:
            Series of implied probabilities
        """
        def ml_to_prob(ml):
            if ml > 0:
                return 100 / (ml + 100)
            else:
                return abs(ml) / (abs(ml) + 100)
        
        return moneyline.apply(ml_to_prob)
    
    @staticmethod
    def calculate_rolling_trends(df: pd.DataFrame, column: str, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """Calculate rolling trend slopes for a given column.
        
        Args:
            df: Input DataFrame
            column: Column to calculate trends for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling trend columns added
        """
        result_df = df.copy()
        
        for window in windows:
            col_name = f"{column}_trend_{window}"
            result_df[col_name] = df[column].rolling(
                window=window, min_periods=2
            ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
        
        return result_df
    
    @staticmethod
    def calculate_rolling_volatility(df: pd.DataFrame, column: str, windows: list = [5, 10, 20]) -> pd.DataFrame:
        """Calculate rolling volatility (standard deviation) for a given column.
        
        Args:
            df: Input DataFrame
            column: Column to calculate volatility for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling volatility columns added
        """
        result_df = df.copy()
        
        for window in windows:
            col_name = f"{column}_volatility_{window}"
            result_df[col_name] = df[column].rolling(window=window, min_periods=1).std()
        
        return result_df
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: list, lags: list = [1, 2, 3]) -> pd.DataFrame:
        """Create lag features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create lags for
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features added
        """
        result_df = df.copy()
        
        for col in columns:
            for lag in lags:
                col_name = f"{col}_lag_{lag}"
                result_df[col_name] = df[col].shift(lag)
        
        return result_df
    
    @staticmethod
    def create_lead_features(df: pd.DataFrame, columns: list, leads: list = [1, 2, 3]) -> pd.DataFrame:
        """Create lead features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to create leads for
            leads: List of lead periods
            
        Returns:
            DataFrame with lead features added
        """
        result_df = df.copy()
        
        for col in columns:
            for lead in leads:
                col_name = f"{col}_lead_{lead}"
                result_df[col_name] = df[col].shift(-lead)
        
        return result_df
    
    @staticmethod
    def calculate_percentile_features(df: pd.DataFrame, columns: list, percentiles: list = [25, 50, 75]) -> pd.DataFrame:
        """Calculate percentile-based features for specified columns.
        
        Args:
            df: Input DataFrame
            columns: List of columns to calculate percentiles for
            percentiles: List of percentiles to calculate
            
        Returns:
            DataFrame with percentile features added
        """
        result_df = df.copy()
        
        for col in columns:
            for pct in percentiles:
                col_name = f"{col}_pct_{pct}"
                result_df[col_name] = df[col].quantile(pct / 100)
        
        return result_df
    
    @staticmethod
    def validate_feature_set(df: pd.DataFrame) -> dict:
        """Validate feature set quality and identify potential issues.
        
        Args:
            df: Feature DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_percentage': (df.duplicated().sum() / len(df)) * 100,
            'constant_columns': (df.nunique() == 1).sum(),
            'high_cardinality_columns': (df.nunique() > len(df) * 0.5).sum()
        }
        
        # Check for infinite values
        validation_results['infinite_values'] = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        
        # Check for extreme values (beyond 3 standard deviations)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_values = 0
        for col in numeric_cols:
            if df[col].std() > 0:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                extreme_values += (z_scores > 3).sum()
        
        validation_results['extreme_values'] = extreme_values
        
        return validation_results
    
    @staticmethod
    def remove_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance.
        
        Args:
            df: Input DataFrame
            threshold: Variance threshold (features below this will be removed)
            
        Returns:
            DataFrame with low variance features removed
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # Calculate variance for each numeric column
        variances = df[numeric_cols].var()
        
        # Keep columns above threshold
        keep_cols = variances[variances > threshold].index.tolist()
        
        # Add back non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        keep_cols.extend(non_numeric_cols)
        
        return df[keep_cols]
    
    @staticmethod
    def create_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
        """Create a comprehensive summary of the feature set.
        
        Args:
            df: Feature DataFrame to summarize
            
        Returns:
            DataFrame with feature summary
        """
        summary_data = []
        
        for col in df.columns:
            col_info = {
                'feature_name': col,
                'data_type': str(df[col].dtype),
                'missing_count': df[col].isnull().sum(),
                'missing_percentage': (df[col].isnull().sum() / len(df)) * 100,
                'unique_count': df[col].nunique()
            }
            
            # Add numeric-specific statistics
            if df[col].dtype in ['int64', 'float64']:
                col_info.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            else:
                col_info.update({
                    'mean': None,
                    'std': None,
                    'min': None,
                    'max': None,
                    'median': None
                })
            
            summary_data.append(col_info)
        
        return pd.DataFrame(summary_data)


# Standalone functions for direct import
def normalize_features(df: pd.DataFrame, columns: list = None, method: str = "zscore") -> pd.DataFrame:
    """Normalize specified columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of columns to normalize (if None, normalize all numeric columns)
        method: Normalization method ('zscore' or 'minmax')
        
    Returns:
        DataFrame with normalized columns
    """
    result_df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if col in df.columns:
            result_df[f"{col}_normalized"] = normalize_series(df[col], method)
    
    return result_df


def create_interaction_features(df: pd.DataFrame, columns: list, max_interactions: int = 10) -> pd.DataFrame:
    """Create interaction features between specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create interactions for
        max_interactions: Maximum number of interaction features to create
        
    Returns:
        DataFrame with interaction features added
    """
    result_df = df.copy()
    
    if len(columns) < 2:
        return result_df
    
    interaction_count = 0
    
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            if interaction_count >= max_interactions:
                break
                
            col1, col2 = columns[i], columns[j]
            
            # Only create interactions for numeric columns
            if df[col1].dtype in ['int64', 'float64'] and df[col2].dtype in ['int64', 'float64']:
                # Multiplication interaction
                result_df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                
                # Division interaction (avoid division by zero)
                if (df[col2] != 0).all():
                    result_df[f"{col1}_div_{col2}"] = df[col1] / df[col2]
                
                interaction_count += 1
    
    return result_df


def create_polynomial_features(df: pd.DataFrame, columns: list, degree: int = 2) -> pd.DataFrame:
    """Create polynomial features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: List of columns to create polynomial features for
        degree: Maximum polynomial degree
        
    Returns:
        DataFrame with polynomial features added
    """
    result_df = df.copy()
    
    for col in columns:
        if col in df.columns and df[col].dtype in ['int64', 'float64']:
            for d in range(2, degree + 1):
                col_name = f"{col}_pow_{d}"
                result_df[col_name] = df[col] ** d
    
    return result_df


def create_ratio_features(df: pd.DataFrame, numerator_cols: list, denominator_cols: list) -> pd.DataFrame:
    """Create ratio features between numerator and denominator columns.
    
    Args:
        df: Input DataFrame
        numerator_cols: List of numerator columns
        denominator_cols: List of denominator columns
        
    Returns:
        DataFrame with ratio features added
    """
    result_df = df.copy()
    
    for num_col in numerator_cols:
        for den_col in denominator_cols:
            if num_col in df.columns and den_col in df.columns:
                # Avoid division by zero
                if (df[den_col] != 0).all():
                    col_name = f"{num_col}_div_{den_col}"
                    result_df[col_name] = df[num_col] / df[den_col]
    
    return result_df