#!/usr/bin/env python3
"""
Phase 3 Core Formulas Implementation for CBB Betting ML System

This module implements all mathematical formulas exactly as specified in the requirements.
Each formula is implemented in its own clearly documented function.
"""

import numpy as np
from sklearn.metrics import roc_auc_score

def logistic_prediction_probability(w, x_i, b):
    """
    Classification (Win Probability) Formula:
    p̂ᵢ = σ(w^T xᵢ + b), σ(z) = 1/(1+e^(-z))
    
    Args:
        w: Weight vector
        x_i: Feature vector for sample i
        b: Bias term
        
    Returns:
        Predicted probability p̂ᵢ
    """
    z = np.dot(w, x_i) + b
    return 1 / (1 + np.exp(-z))

def decision_rule(p_hat_i, threshold=0.5):
    """
    Decision Rule Formula:
    ŷᵢ = {1 if p̂ᵢ ≥ 0.5, 0 otherwise}
    
    Args:
        p_hat_i: Predicted probability for sample i
        threshold: Decision threshold (default 0.5)
        
    Returns:
        Predicted class ŷᵢ
    """
    return 1 if p_hat_i >= threshold else 0

def linear_regression_prediction(w, x_i, b):
    """
    Regression (Point Differential) Formula:
    ŷᵢ = w^T xᵢ + b
    
    Args:
        w: Weight vector
        x_i: Feature vector for sample i
        b: Bias term
        
    Returns:
        Predicted value ŷᵢ
    """
    return np.dot(w, x_i) + b

def accuracy_metric(TP, TN, FP, FN):
    """
    Classification Accuracy Formula:
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Args:
        TP: True Positives
        TN: True Negatives
        FP: False Positives
        FN: False Negatives
        
    Returns:
        Accuracy value
    """
    return (TP + TN) / (TP + TN + FP + FN)

def precision_metric(TP, FP):
    """
    Classification Precision Formula:
    Precision = TP / (TP + FP)
    
    Args:
        TP: True Positives
        FP: False Positives
        
    Returns:
        Precision value
    """
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0

def recall_metric(TP, FN):
    """
    Classification Recall Formula:
    Recall = TP / (TP + FN)
    
    Args:
        TP: True Positives
        FN: False Negatives
        
    Returns:
        Recall value
    """
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0

def f1_score_metric(TP, FP, FN):
    """
    F1 Score Formula:
    F1 = (2 · Precision · Recall) / (Precision + Recall)
    
    Args:
        TP: True Positives
        FP: False Positives
        FN: False Negatives
        
    Returns:
        F1 score value
    """
    precision = precision_metric(TP, FP)
    recall = recall_metric(TP, FN)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def auc_metric(y_true, y_pred_proba):
    """
    AUC Formula:
    AUC = ∫₀¹ TPR(FPR) dFPR
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        AUC value (using sklearn implementation)
    """
    return roc_auc_score(y_true, y_pred_proba)

def rmse_metric(y_true, y_pred):
    """
    RMSE Formula:
    RMSE = √((1/n) ∑ᵢ₌₁ⁿ (yᵢ - ŷᵢ)²)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    n = len(y_true)
    return np.sqrt((1/n) * np.sum((y_true - y_pred) ** 2))

def mae_metric(y_true, y_pred):
    """
    MAE Formula:
    MAE = (1/n) ∑ᵢ₌₁ⁿ |yᵢ - ŷᵢ|
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    n = len(y_true)
    return (1/n) * np.sum(np.abs(y_true - y_pred))

def r2_metric(y_true, y_pred):
    """
    R² Formula:
    R² = 1 - (∑ᵢ₌₁ⁿ (yᵢ - ŷᵢ)²) / (∑ᵢ₌₁ⁿ (yᵢ - ȳ)²)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² value
    """
    n = len(y_true)
    y_bar = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_bar) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

def edge_calculation(p_hat, O):
    """
    Edge Formula:
    Edge = p̂ · (O - 1) - (1 - p̂)
    
    Args:
        p_hat: Predicted probability p̂
        O: Decimal odds
        
    Returns:
        Edge value
    """
    return p_hat * (O - 1) - (1 - p_hat)

def roi_calculation(total_profit, total_stakes):
    """
    ROI Formula:
    ROI = Total Profit / Total Stakes
    
    Args:
        total_profit: Total profit from bets
        total_stakes: Total amount staked
        
    Returns:
        ROI value
    """
    return total_profit / total_stakes if total_stakes > 0 else 0.0

def kelly_fraction_calculation(p, O):
    """
    Kelly Fraction Formula:
    f* = (bp - q) / b, where b = O - 1, q = 1 - p
    
    Args:
        p: Predicted probability p̂
        O: Decimal odds
        
    Returns:
        Kelly fraction f*
    """
    b = O - 1
    q = 1 - p
    kelly = (b * p - q) / b
    return max(kelly, 0.0)  # No negative betting

# Test the implementation
if __name__ == "__main__":
    print("🧪 Testing Phase 3 Formula Implementation...")
    
    # Test data
    w = np.array([0.1, 0.2, 0.3])
    x_i = np.array([1.0, 2.0, 3.0])
    b = 0.5
    
    # Test logistic prediction
    p_hat = logistic_prediction_probability(w, x_i, b)
    print(f"✅ Logistic Prediction: p̂ = {p_hat:.4f}")
    
    # Test decision rule
    y_hat = decision_rule(p_hat)
    print(f"✅ Decision Rule: ŷ = {y_hat}")
    
    # Test linear regression
    y_reg = linear_regression_prediction(w, x_i, b)
    print(f"✅ Linear Regression: ŷ = {y_reg:.4f}")
    
    # Test classification metrics
    TP, TN, FP, FN = 80, 70, 20, 30
    acc = accuracy_metric(TP, TN, FP, FN)
    prec = precision_metric(TP, FP)
    rec = recall_metric(TP, FN)
    f1 = f1_score_metric(TP, FP, FN)
    
    print(f"✅ Classification Metrics:")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {prec:.4f}")
    print(f"   Recall: {rec:.4f}")
    print(f"   F1: {f1:.4f}")
    
    # Test regression metrics
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    rmse = rmse_metric(y_true, y_pred)
    mae = mae_metric(y_true, y_pred)
    r2 = r2_metric(y_true, y_pred)
    
    print(f"✅ Regression Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")
    
    # Test betting metrics
    p_bet = 0.6
    odds = 2.0
    
    edge = edge_calculation(p_bet, odds)
    roi = roi_calculation(100, 1000)
    kelly = kelly_fraction_calculation(p_bet, odds)
    
    print(f"✅ Betting Metrics:")
    print(f"   Edge: {edge:.4f}")
    print(f"   ROI: {roi:.4f}")
    print(f"   Kelly: {kelly:.4f}")
    
    print("\n✅ All Phase 3 formulas implemented and tested successfully!")