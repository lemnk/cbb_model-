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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def validate_keys(df, key="game_id", df_name="DataFrame"):
    """
    Validate that a DataFrame has the required key column without nulls.
    
    Args:
        df: DataFrame to validate
        key: Required key column name
        df_name: Name of DataFrame for error messages
        
    Returns:
        True if validation passes
        
    Raises:
        ValueError: If key is missing or contains nulls
    """
    if key not in df.columns:
        raise ValueError(f"Missing required key `{key}` in {df_name}")
    if df[key].isnull().any():
        raise ValueError(f"Null values found in `{key}` of {df_name}")
    return True

def standardize_team_names(df, team_col="team"):
    """
    Standardize team names to prevent merge mismatches.
    
    Args:
        df: DataFrame with team names
        team_col: Column containing team names
        
    Returns:
        DataFrame with standardized team names
    """
    df = df.copy()
    if team_col in df.columns:
        df[team_col] = df[team_col].astype(str).str.strip().str.lower()
    return df

class Normalizer:
    """
    Safe normalizer that prevents data leakage by fitting on training data only.
    """
    
    def __init__(self, method="minmax"):
        self.method = method
        self.params = {}
        self.is_fitted = False
    
    def fit(self, series):
        """
        Fit the normalizer on training data.
        
        Args:
            series: Training data series
        """
        if self.method == "minmax":
            self.params["min"] = series.min()
            self.params["max"] = series.max()
        elif self.method == "zscore":
            self.params["mean"] = series.mean()
            self.params["std"] = series.std()
        elif self.method == "robust":
            median = series.median()
            q75, q25 = series.quantile([0.75, 0.25])
            self.params["median"] = median
            self.params["iqr"] = q75 - q25
        elif self.method == "log":
            # Log transform doesn't need fitting parameters
            pass
        
        self.is_fitted = True
    
    def transform(self, series):
        """
        Transform data using fitted parameters.
        
        Args:
            series: Data to transform
            
        Returns:
            Transformed series
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        if self.method == "minmax":
            min_val = self.params["min"]
            max_val = self.params["max"]
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val + 1e-9)
        elif self.method == "zscore":
            mean_val = self.params["mean"]
            std_val = self.params["std"]
            if std_val == 0:
                return pd.Series(0, index=series.index)
            return (series - mean_val) / (std_val + 1e-9)
        elif self.method == "robust":
            median_val = self.params["median"]
            iqr_val = self.params["iqr"]
            if iqr_val == 0:
                return pd.Series(0, index=series.index)
            return (series - median_val) / (iqr_val + 1e-9)
        elif self.method == "log":
            return log_transform(series)
        else:
            return series
    
    def fit_transform(self, series):
        """
        Fit and transform in one step (for training data only).
        
        Args:
            series: Training data series
            
        Returns:
            Transformed series
        """
        self.fit(series)
        return self.transform(series)

def safe_fill(df, col, fill_value=0):
    """
    Safely fill missing values in a column, creating the column if it doesn't exist.
    
    Args:
        df: DataFrame to process
        col: Column name to fill
        fill_value: Value to use for filling
        
    Returns:
        DataFrame with filled column
    """
    df = df.copy()
    if col not in df.columns:
        df[col] = fill_value
    else:
        df[col] = df[col].fillna(fill_value)
    return df

def ensure_time_order(df, date_col="date", team_col="team"):
    """
    Ensure DataFrame is sorted by team and date to prevent data leakage.
    
    Args:
        df: DataFrame to sort
        date_col: Date column name
        team_col: Team column name
        
    Returns:
        Sorted DataFrame
    """
    df = df.copy()
    if date_col in df.columns and team_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values([team_col, date_col])
    return df

def normalize(series, method="minmax"):
    """
    Normalize a Pandas Series using specified method.
    
    Args:
        series: Pandas Series to normalize
        method: Normalization method ('minmax', 'zscore', 'robust')
        
    Returns:
        Normalized Pandas Series
    """
    if series.empty:
        return series
    
    if method == "minmax":
        scaler = MinMaxScaler()
        return pd.Series(
            scaler.fit_transform(series.values.reshape(-1, 1)).flatten(),
            index=series.index,
            name=series.name
        )
    elif method == "zscore":
        scaler = StandardScaler()
        return pd.Series(
            scaler.fit_transform(series.values.reshape(-1, 1)).flatten(),
            index=series.index,
            name=series.name
        )
    elif method == "robust":
        # Robust scaling using median and IQR
        median = series.median()
        q75, q25 = series.quantile([0.75, 0.25])
        iqr = q75 - q25
        if iqr == 0:
            return pd.Series(0, index=series.index, name=series.name)
        return (series - median) / iqr
    elif method == "log":
        return log_transform(series)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def log_transform(series):
    """
    Apply log transform: x'ᵢ = log(xᵢ + 1)
    
    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input data, must be non-negative.
        
    Returns
    -------
    transformed : same type as input
        Log-transformed values.
        
    Raises
    ------
    ValueError
        If input contains negative values.
        
    Notes
    -----
    Uses np.log1p for numerical stability instead of np.log(series + 1).
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(series, 'values'):
        # Pandas Series
        if (series < 0).any():
            raise ValueError("Log transform requires non-negative input values.")
        return pd.Series(np.log1p(series.values), index=series.index, name=series.name)
    else:
        # Numpy array
        if np.any(series < 0):
            raise ValueError("Log transform requires non-negative input values.")
        return np.log1p(series)

def scale(df, columns=None, method="zscore"):
    """
    Scale multiple columns in a DataFrame.
    
    Args:
        df: DataFrame to scale
        columns: List of column names to scale (None = all numeric)
        method: Scaling method ('minmax', 'zscore', 'robust')
        
    Returns:
        DataFrame with scaled columns
    """
    df_scaled = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    for col in columns:
        if col in df.columns and df[col].dtype in [np.number]:
            if method == "log":
                df_scaled[col] = log_transform(df[col])
            else:
                df_scaled[col] = normalize(df[col], method)
    
    return df_scaled

def handle_missing(df, strategy="zero", columns=None):
    """
    Handle missing values in DataFrame.
    
    Args:
        df: DataFrame to process
        strategy: Strategy for handling missing values
                 ('zero', 'mean', 'median', 'drop', 'forward_fill')
        columns: List of column names to process (None = all)
        
    Returns:
        DataFrame with missing values handled
    """
    df_processed = df.copy()
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if df[col].isnull().any():
            if strategy == "zero":
                df_processed[col] = df[col].fillna(0)
            elif strategy == "mean":
                if df[col].dtype in [np.number]:
                    df_processed[col] = df[col].fillna(df[col].mean())
                else:
                    df_processed[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
            elif strategy == "median":
                if df[col].dtype in [np.number]:
                    df_processed[col] = df[col].fillna(df[col].median())
                else:
                    df_processed[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "unknown")
            elif strategy == "drop":
                df_processed = df_processed.dropna(subset=[col])
            elif strategy == "forward_fill":
                df_processed[col] = df[col].fillna(method='ffill')
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
    
    return df_processed

def create_lag_features(df, columns, lags=[1, 2, 3, 5, 10]):
    """
    Create lag features for time series data.
    
    Args:
        df: DataFrame with time series data
        columns: List of column names to create lags for
        lags: List of lag periods
        
    Returns:
        DataFrame with lag features added
    """
    df_lagged = df.copy()
    
    # Ensure DataFrame is sorted by time
    if 'date' in df.columns:
        df_lagged = df_lagged.sort_values('date')
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for lag in lags:
            lag_col_name = f"{col}_lag_{lag}"
            df_lagged[lag_col_name] = df_lagged[col].shift(lag)
    
    return df_lagged

def create_rolling_features(df, columns, windows=[3, 5, 10], functions=['mean', 'std', 'min', 'max']):
    """
    Create rolling window features.
    
    Args:
        df: DataFrame to process
        columns: List of column names to create rolling features for
        windows: List of window sizes
        functions: List of aggregation functions
        
    Returns:
        DataFrame with rolling features added
    """
    df_rolling = df.copy()
    
    # Ensure DataFrame is sorted by time
    if 'date' in df.columns:
        df_rolling = df_rolling.sort_values('date')
    
    for col in columns:
        if col not in df.columns:
            continue
            
        for window in windows:
            for func in functions:
                if func == 'mean':
                    df_rolling[f"{col}_rolling_mean_{window}"] = df_rolling[col].rolling(window=window, min_periods=1).mean()
                elif func == 'std':
                    df_rolling[f"{col}_rolling_std_{window}"] = df_rolling[col].rolling(window=window, min_periods=1).std()
                elif func == 'min':
                    df_rolling[f"{col}_rolling_min_{window}"] = df_rolling[col].rolling(window=window, min_periods=1).min()
                elif func == 'max':
                    df_rolling[f"{col}_rolling_max_{window}"] = df_rolling[col].rolling(window=window, min_periods=1).max()
                elif func == 'sum':
                    df_rolling[f"{col}_rolling_sum_{window}"] = df_rolling[col].rolling(window=window, min_periods=1).sum()
    
    return df_rolling

def encode_categorical(df, columns=None, method="onehot"):
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame to process
        columns: List of categorical column names (None = auto-detect)
        method: Encoding method ('onehot', 'label', 'target')
        
    Returns:
        DataFrame with encoded categorical variables
    """
    df_encoded = df.copy()
    
    if columns is None:
        # Auto-detect categorical columns
        columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == "onehot":
            # One-hot encoding
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(columns=[col])
        elif method == "label":
            # Label encoding
            df_encoded[col] = df_encoded[col].astype('category').cat.codes
        elif method == "target":
            # Target encoding (mean encoding) - requires target variable
            if 'target' in df.columns:
                target_means = df.groupby(col)['target'].mean()
                df_encoded[f"{col}_target_encoded"] = df_encoded[col].map(target_means)
            else:
                # Fall back to label encoding if no target
                df_encoded[col] = df_encoded[col].astype('category').cat.codes
        else:
            raise ValueError(f"Unknown encoding method: {method}")
    
    return df_encoded

def remove_outliers(df, columns=None, method="iqr", threshold=1.5):
    """
    Remove outliers from DataFrame.
    
    Args:
        df: DataFrame to process
        columns: List of column names to check for outliers (None = all numeric)
        method: Outlier detection method ('iqr', 'zscore')
        threshold: Threshold for outlier detection
        
    Returns:
        DataFrame with outliers removed
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            col_outliers = z_scores > threshold
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        outlier_mask = outlier_mask | col_outliers
    
    return df_clean[~outlier_mask]

def feature_correlation_analysis(df, target_col=None, threshold=0.8):
    """
    Analyze feature correlations and identify highly correlated features.
    
    Args:
        df: DataFrame to analyze
        target_col: Target column name (if provided, include in analysis)
        threshold: Correlation threshold for identifying high correlations
        
    Returns:
        Dictionary with correlation analysis results
    """
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col and target_col in numeric_cols:
        # Calculate correlations with target
        target_correlations = df[numeric_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        target_correlations = target_correlations.drop(target_col) if target_col in target_correlations.index else target_correlations
    else:
        target_correlations = None
    
    # Calculate feature correlation matrix
    correlation_matrix = df[numeric_cols].corr()
    
    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            col1 = correlation_matrix.columns[i]
            col2 = correlation_matrix.columns[j]
            corr_value = correlation_matrix.iloc[i, j]
            if abs(corr_value) > threshold:
                high_corr_pairs.append({
                    'feature1': col1,
                    'feature2': col2,
                    'correlation': corr_value
                })
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'correlation_matrix': correlation_matrix,
        'target_correlations': target_correlations,
        'high_correlation_pairs': high_corr_pairs,
        'threshold': threshold
    }