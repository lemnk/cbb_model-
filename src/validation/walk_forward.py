"""
Walk-forward validation and backtesting for Phase 4.
Implements rolling window validation for time series data.
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, List, Dict, Any
from sklearn.metrics import accuracy_score, roc_auc_score
from src.metrics import log_loss, roi


def walk_forward_split(data, train_size, step_size):
    """
    Walk-Forward Split generator.
    
    Generates train/validation splits for time series data.
    
    Parameters:
    -----------
    data : array-like
        Data to split
    train_size : int
        Size of training window
    step_size : int
        Step size for moving window
        
    Yields:
    -------
    tuple : (train_data, val_data) for each split
    """
    for start in range(0, len(data) - train_size, step_size):
        end = start + train_size
        yield data[start:end], data[end:end + step_size]


class WalkForwardValidator:
    """
    Comprehensive walk-forward validation system for backtesting.
    """
    
    def __init__(self, train_size, step_size, metrics=['accuracy', 'roc_auc', 'log_loss']):
        """
        Initialize walk-forward validator.
        
        Parameters:
        -----------
        train_size : int
            Size of training window
        step_size : int
            Step size for moving window
        metrics : list
            List of metrics to compute
        """
        self.train_size = train_size
        self.step_size = step_size
        self.metrics = metrics
        self.results = []
        
    def validate(self, data, model_factory, feature_cols, target_col, 
                 date_col=None, threshold=0.5):
        """
        Perform walk-forward validation.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with features and target
        model_factory : callable
            Function that returns a fresh model instance
        feature_cols : list
            List of feature column names
        target_col : str
            Target column name
        date_col : str, optional
            Date column for time ordering
        threshold : float
            Classification threshold
            
        Returns:
        --------
        dict : Validation results summary
        """
        if date_col:
            data = data.sort_values(date_col).reset_index(drop=True)
        
        splits = list(walk_forward_split(data, self.train_size, self.step_size))
        self.results = []
        
        for i, (train_data, val_data) in enumerate(splits):
            # Skip if validation set is too small
            if len(val_data) < 2:
                continue
                
            # Train model
            model = model_factory()
            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            model.fit(X_train, y_train)
            
            # Make predictions
            X_val = val_data[feature_cols]
            y_val = val_data[target_col]
            
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                y_pred = (y_pred_proba >= threshold).astype(int)
            else:
                y_pred = model.predict(X_val)
                y_pred_proba = y_pred.astype(float)
            
            # Calculate metrics
            split_results = self._calculate_metrics(y_val, y_pred, y_pred_proba, i)
            self.results.append(split_results)
        
        return self._summarize_results()
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, split_idx):
        """
        Calculate metrics for a single split.
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred : array
            Predicted labels
        y_pred_proba : array
            Predicted probabilities
        split_idx : int
            Split index
            
        Returns:
        --------
        dict : Metrics for this split
        """
        results = {'split': split_idx}
        
        if 'accuracy' in self.metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'roc_auc' in self.metrics:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            except ValueError:
                results['roc_auc'] = np.nan
        
        if 'log_loss' in self.metrics:
            try:
                results['log_loss'] = log_loss(y_true, y_pred_proba)
            except ValueError:
                results['log_loss'] = np.nan
        
        # Calculate ROI simulation
        results['roi'] = self._calculate_split_roi(y_true, y_pred_proba)
        
        return results
    
    def _calculate_split_roi(self, y_true, y_pred_proba, stake=100):
        """
        Calculate ROI for a split using Kelly criterion.
        
        Parameters:
        -----------
        y_true : array
            True labels
        y_pred_proba : array
            Predicted probabilities
        stake : float
            Base stake amount
            
        Returns:
        --------
        float : ROI for this split
        """
        total_stake = 0
        total_return = 0
        
        for prob, actual in zip(y_pred_proba, y_true):
            if prob > 0.5:  # Bet on positive class
                bet_amount = stake * (2 * prob - 1)  # Kelly fraction
                total_stake += bet_amount
                
                if actual == 1:
                    total_return += bet_amount * 2  # Win
                else:
                    total_return += 0  # Lose
        
        if total_stake == 0:
            return 0.0
        
        return roi(total_return, total_stake)
    
    def _summarize_results(self):
        """
        Summarize validation results across all splits.
        
        Returns:
        --------
        dict : Summary statistics
        """
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        summary = {}
        
        for metric in self.metrics:
            if metric in df.columns:
                summary[f'{metric}_mean'] = df[metric].mean()
                summary[f'{metric}_std'] = df[metric].std()
                summary[f'{metric}_min'] = df[metric].min()
                summary[f'{metric}_max'] = df[metric].max()
        
        # ROI summary
        if 'roi' in df.columns:
            summary['roi_mean'] = df['roi'].mean()
            summary['roi_std'] = df['roi'].std()
            summary['cumulative_roi'] = df['roi'].sum()
        
        summary['n_splits'] = len(self.results)
        summary['total_samples'] = sum(len(split[1]) for split in 
                                     walk_forward_split(range(1000), self.train_size, self.step_size))
        
        return summary
    
    def get_results_dataframe(self):
        """
        Get results as a pandas DataFrame.
        
        Returns:
        --------
        pd.DataFrame : Results for each split
        """
        return pd.DataFrame(self.results)
    
    def plot_cumulative_roi(self, save_path=None):
        """
        Plot cumulative ROI over time.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if not self.results:
            print("No results to plot. Run validate() first.")
            return
        
        import matplotlib.pyplot as plt
        
        df = pd.DataFrame(self.results)
        cumulative_roi = df['roi'].cumsum()
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_roi, marker='o', linewidth=2, markersize=4)
        plt.title('Cumulative ROI Over Time')
        plt.xlabel('Split Index')
        plt.ylabel('Cumulative ROI')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()