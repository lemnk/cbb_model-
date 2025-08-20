"""
Calibration methods implementation for Phase 4.
All formulas match exactly with the mathematical definitions provided.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def platt_scaling(probs, y_true):
    """
    Platt Scaling.
    
    Formula: p' = σ(Ap + B)
    
    Parameters:
    -----------
    probs : array-like
        Raw predicted probabilities
    y_true : array-like
        True binary labels
        
    Returns:
    --------
    array : Calibrated probabilities
    """
    lr = LogisticRegression(random_state=42)
    lr.fit(probs.reshape(-1, 1), y_true)
    return lr.predict_proba(probs.reshape(-1, 1))[:, 1]


def isotonic_calibration(probs, y_true):
    """
    Isotonic Regression Calibration.
    
    Non-parametric calibration using isotonic regression.
    
    Parameters:
    -----------
    probs : array-like
        Raw predicted probabilities
    y_true : array-like
        True binary labels
        
    Returns:
    --------
    array : Calibrated probabilities
    """
    iso = IsotonicRegression(out_of_bounds="clip")
    return iso.fit_transform(probs, y_true)


def brier_score(y_true, probs):
    """
    Brier Score.
    
    Formula: BS = (1/n) * Σᵢ₌₁ⁿ (pᵢ - yᵢ)²
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    probs : array-like
        Predicted probabilities
        
    Returns:
    --------
    float : Brier score (lower is better, range 0-1)
    """
    return np.mean((probs - y_true) ** 2)


class Calibrator:
    """
    Comprehensive probability calibration system.
    """
    
    def __init__(self, method='platt'):
        """
        Initialize calibrator.
        
        Parameters:
        -----------
        method : str
            'platt' or 'isotonic'
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
        
    def fit(self, probs, y_true):
        """
        Fit the calibrator.
        
        Parameters:
        -----------
        probs : array-like
            Raw predicted probabilities
        y_true : array-like
            True binary labels
        """
        if self.method == 'platt':
            self.calibrator = LogisticRegression(random_state=42)
            self.calibrator.fit(probs.reshape(-1, 1), y_true)
        elif self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(probs, y_true)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        return self
    
    def transform(self, probs):
        """
        Calibrate probabilities.
        
        Parameters:
        -----------
        probs : array-like
            Raw predicted probabilities
            
        Returns:
        --------
        array : Calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        if self.method == 'platt':
            return self.calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
        elif self.method == 'isotonic':
            return self.calibrator.transform(probs)
    
    def fit_transform(self, probs, y_true):
        """
        Fit and transform in one step.
        
        Parameters:
        -----------
        probs : array-like
            Raw predicted probabilities
        y_true : array-like
            True binary labels
            
        Returns:
        --------
        array : Calibrated probabilities
        """
        return self.fit(probs, y_true).transform(probs)


def evaluate_calibration(y_true, probs_raw, probs_calibrated, n_bins=10):
    """
    Evaluate calibration quality.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    probs_raw : array-like
        Raw predicted probabilities
    probs_calibrated : array-like
        Calibrated probabilities
    n_bins : int
        Number of bins for reliability diagram
        
    Returns:
    --------
    dict : Calibration metrics
    """
    from src.metrics import expected_calibration_error
    
    # Calculate metrics
    ece_raw = expected_calibration_error(probs_raw, y_true, n_bins)
    ece_calibrated = expected_calibration_error(probs_calibrated, y_true, n_bins)
    brier_raw = brier_score(y_true, probs_raw)
    brier_calibrated = brier_score(y_true, probs_calibrated)
    
    return {
        'ece_raw': ece_raw,
        'ece_calibrated': ece_calibrated,
        'brier_raw': brier_raw,
        'brier_calibrated': brier_calibrated,
        'ece_improvement': ece_raw - ece_calibrated,
        'brier_improvement': brier_raw - brier_calibrated
    }