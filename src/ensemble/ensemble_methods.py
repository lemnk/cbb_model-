"""
Ensemble methods implementation for Phase 4.
All formulas match exactly with the mathematical definitions provided.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def averaging_ensemble(preds_list):
    """
    Averaging Ensemble.
    
    Formula: p̂_ensemble = (1/M) * Σₘ₌₁ᴹ p̂ₘ
    
    Parameters:
    -----------
    preds_list : list of arrays
        List of prediction arrays from different models
        
    Returns:
    --------
    array : Ensemble predictions (simple average)
    """
    return np.mean(preds_list, axis=0)


def weighted_ensemble(preds_list, weights):
    """
    Weighted Ensemble.
    
    Formula: p̂_ensemble = Σₘ₌₁ᴹ wₘ * p̂ₘ, where Σwₘ = 1
    
    Parameters:
    -----------
    preds_list : list of arrays
        List of prediction arrays from different models
    weights : array-like
        Weights for each model (must sum to 1)
        
    Returns:
    --------
    array : Ensemble predictions (weighted average)
    """
    weights = np.array(weights)
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights)
    return np.average(preds_list, axis=0, weights=weights)


def stacked_ensemble(base_preds, y_true):
    """
    Stacked Ensemble.
    
    Uses Logistic Regression meta-learner to combine base predictions.
    
    Parameters:
    -----------
    base_preds : array-like
        Base model predictions (n_samples, n_models)
    y_true : array-like
        True labels for training the meta-learner
        
    Returns:
    --------
    LogisticRegression : Fitted meta-learner model
    """
    meta = LogisticRegression(random_state=42)
    meta.fit(base_preds, y_true)
    return meta


class EnsembleModel:
    """
    Comprehensive ensemble model that combines multiple base models.
    """
    
    def __init__(self, base_models, ensemble_method='weighted'):
        """
        Initialize ensemble model.
        
        Parameters:
        -----------
        base_models : list
            List of fitted base models
        ensemble_method : str
            'averaging', 'weighted', or 'stacked'
        """
        self.base_models = base_models
        self.ensemble_method = ensemble_method
        self.meta_learner = None
        self.weights = None
        
    def fit(self, X, y):
        """
        Fit the ensemble model.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
        """
        if self.ensemble_method == 'stacked':
            # Get base predictions for training meta-learner
            base_preds = np.column_stack([
                model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') 
                else model.predict(X) for model in self.base_models
            ])
            self.meta_learner = stacked_ensemble(base_preds, y)
        elif self.ensemble_method == 'weighted':
            # Use validation performance to set weights
            val_scores = []
            for model in self.base_models:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                # Use ROC-AUC as weight
                from src.metrics import roc_auc_score
                score = roc_auc_score(y, pred)
                val_scores.append(score)
            
            # Normalize weights
            self.weights = np.array(val_scores) / np.sum(val_scores)
        
        return self
    
    def predict_proba(self, X):
        """
        Get ensemble predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
            
        Returns:
        --------
        array : Ensemble probability predictions
        """
        base_preds = []
        for model in self.base_models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_preds.append(pred)
        
        if self.ensemble_method == 'averaging':
            return averaging_ensemble(base_preds)
        elif self.ensemble_method == 'weighted':
            return weighted_ensemble(base_preds, self.weights)
        elif self.ensemble_method == 'stacked':
            base_preds_array = np.column_stack(base_preds)
            return self.meta_learner.predict_proba(base_preds_array)[:, 1]
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
    def predict(self, X, threshold=0.5):
        """
        Get binary predictions.
        
        Parameters:
        -----------
        X : array-like
            Features to predict on
        threshold : float
            Classification threshold
            
        Returns:
        --------
        array : Binary predictions
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)