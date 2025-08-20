"""
Core metrics implementation for Phase 4.
All formulas match exactly with the mathematical definitions provided.
"""

import numpy as np
from sklearn.metrics import roc_auc_score as sklearn_roc_auc
from sklearn.metrics import log_loss as sklearn_log_loss


def roc_auc_score(y_true, y_pred_proba):
    """
    ROC-AUC Score.
    
    Formula: ROC-AUC = (1/n) * Σᵢ₌₁ⁿ 1[ŷᵢ > ŷⱼ]
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
        
    Returns:
    --------
    float : ROC-AUC score between 0 and 1
    """
    return sklearn_roc_auc_score(y_true, y_pred_proba)


def log_loss(y_true, y_pred_proba):
    """
    Log Loss (Binary Cross-Entropy).
    
    Formula: L = -(1/n) * Σᵢ₌₁ⁿ [yᵢ log(pᵢ) + (1-yᵢ) log(1-pᵢ)]
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
        
    Returns:
    --------
    float : Log loss value (lower is better)
    """
    return sklearn_log_loss(y_true, y_pred_proba)


def expected_calibration_error(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE).
    
    Formula: ECE = Σ |acc - conf| * (bin_count / total)
    
    Parameters:
    -----------
    probs : array-like
        Predicted probabilities
    labels : array-like
        True binary labels
    n_bins : int, default=10
        Number of bins for probability calibration
        
    Returns:
    --------
    float : ECE value between 0 and 1 (lower is better)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    binids = np.digitize(probs, bins) - 1
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = binids == i
        if np.any(bin_mask):
            acc = np.mean(labels[bin_mask])
            conf = np.mean(probs[bin_mask])
            ece += np.abs(acc - conf) * np.sum(bin_mask) / len(probs)
    
    return ece


def roi(total_return, total_stake):
    """
    Return on Investment.
    
    Formula: ROI = (Total Return - Total Stake) / Total Stake
    
    Parameters:
    -----------
    total_return : float
        Total amount returned from investments
    total_stake : float
        Total amount staked/invested
        
    Returns:
    --------
    float : ROI as a decimal (e.g., 0.2 for 20%)
    """
    return (total_return - total_stake) / total_stake


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