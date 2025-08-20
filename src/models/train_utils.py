#!/usr/bin/env python3
"""
Phase 2 Training Utilities for CBB Betting ML System

This module implements all missing Phase 2 training formulas exactly as specified:
- Linear Regression Loss (MSE)
- Logistic Regression Loss (Binary Cross-Entropy)
- Gradient Descent Updates
- L2 Regularization
- Log Transform and Inverse Log Transform
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple

def linear_regression_loss(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Mean Squared Error Loss: L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²
    
    Parameters
    ----------
    y_true : np.ndarray
        True target values, shape (n_samples,)
    y_pred : np.ndarray
        Predicted values, shape (n_samples,)
    w : np.ndarray
        Weight vector, shape (n_features,)
    b : float
        Bias term
        
    Returns
    -------
    float
        MSE loss value
        
    Notes
    -----
    This function computes the exact mathematical formula for MSE loss.
    The prediction y_pred should ideally be computed as wᵀxᵢ + b.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute MSE: L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²
    mse = (1/n) * np.sum((y_true - y_pred) ** 2)
    
    return float(mse)

def logistic_regression_loss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Binary Cross-Entropy Loss: L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]
    
    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1), shape (n_samples,)
    y_pred_proba : np.ndarray
        Predicted probabilities, shape (n_samples,)
        
    Returns
    -------
    float
        Binary cross-entropy loss value
        
    Notes
    -----
    This function computes the exact mathematical formula for binary cross-entropy loss.
    Uses epsilon (1e-15) to prevent log(0) which would result in -infinity.
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Clip probabilities to prevent log(0) or log(1)
    y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    
    # Compute binary cross-entropy: L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]
    bce = -(1/n) * np.sum(y_true * np.log(y_pred_proba) + 
                          (1 - y_true) * np.log(1 - y_pred_proba))
    
    return float(bce)

def gradient_descent_update_w(w: np.ndarray, alpha: float, grad_w: np.ndarray) -> np.ndarray:
    """
    Weight Update Rule: w := w - α ∂L/∂w
    
    Parameters
    ----------
    w : np.ndarray
        Current weight vector, shape (n_features,)
    alpha : float
        Learning rate (α)
    grad_w : np.ndarray
        Gradient of loss with respect to weights, shape (n_features,)
        
    Returns
    -------
    np.ndarray
        Updated weight vector
        
    Notes
    -----
    This function implements the exact gradient descent update rule for weights.
    The gradient grad_w should be computed as ∂L/∂w.
    """
    # Ensure inputs are numpy arrays
    w = np.asarray(w)
    grad_w = np.asarray(grad_w)
    
    # Weight update: w := w - α ∂L/∂w
    w_new = w - alpha * grad_w
    
    return w_new

def gradient_descent_update_b(b: float, alpha: float, grad_b: float) -> float:
    """
    Bias Update Rule: b := b - α ∂L/∂b
    
    Parameters
    ----------
    b : float
        Current bias term
    alpha : float
        Learning rate (α)
    grad_b : float
        Gradient of loss with respect to bias
        
    Returns
    -------
    float
        Updated bias term
        
    Notes
    -----
    This function implements the exact gradient descent update rule for bias.
    The gradient grad_b should be computed as ∂L/∂b.
    """
    # Bias update: b := b - α ∂L/∂b
    b_new = b - alpha * grad_b
    
    return float(b_new)

def l2_regularization_loss(loss: float, w: np.ndarray, lambda_reg: float) -> float:
    """
    L2 Regularized Loss: L_reg = L + λ ||w||²
    
    Parameters
    ----------
    loss : float
        Base loss value (e.g., MSE or cross-entropy)
    w : np.ndarray
        Weight vector, shape (n_features,)
    lambda_reg : float
        Regularization strength (λ)
        
    Returns
    -------
    float
        Regularized loss value
        
    Notes
    -----
    This function adds L2 regularization to the base loss.
    The L2 penalty is λ ||w||² = λ Σᵢ wᵢ²
    """
    # Ensure weight vector is numpy array
    w = np.asarray(w)
    
    # Compute L2 regularization penalty: λ ||w||² = λ Σᵢ wᵢ²
    l2_penalty = lambda_reg * np.sum(w ** 2)
    
    # Regularized loss: L_reg = L + λ ||w||²
    regularized_loss = loss + l2_penalty
    
    return float(regularized_loss)

def compute_gradients_linear_regression(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute gradients for linear regression: ∂L/∂w and ∂L/∂b
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y_true : np.ndarray
        True target values, shape (n_samples,)
    y_pred : np.ndarray
        Predicted values, shape (n_samples,)
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Gradient with respect to weights (∂L/∂w) and bias (∂L/∂b)
        
    Notes
    -----
    For MSE loss L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²:
    ∂L/∂w = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b)) xᵢ
    ∂L/∂b = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b))
    """
    n = len(y_true)
    if n == 0:
        return np.zeros(X.shape[1]), 0.0
    
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Compute residuals
    residuals = y_true - y_pred
    
    # Gradient with respect to weights: ∂L/∂w = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b)) xᵢ
    grad_w = -(2/n) * np.sum(residuals.reshape(-1, 1) * X, axis=0)
    
    # Gradient with respect to bias: ∂L/∂b = -(2/n) Σᵢ (yᵢ - (wᵀxᵢ + b))
    grad_b = -(2/n) * np.sum(residuals)
    
    return grad_w, float(grad_b)

def compute_gradients_logistic_regression(X: np.ndarray, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Compute gradients for logistic regression: ∂L/∂w and ∂L/∂b
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix, shape (n_samples, n_features)
    y_true : np.ndarray
        True binary labels (0 or 1), shape (n_samples,)
    y_pred_proba : np.ndarray
        Predicted probabilities, shape (n_samples,)
        
    Returns
    -------
    Tuple[np.ndarray, float]
        Gradient with respect to weights (∂L/∂w) and bias (∂L/∂b)
        
    Notes
    -----
    For binary cross-entropy loss L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]:
    ∂L/∂w = (1/n) Σᵢ (p̂ᵢ - yᵢ) xᵢ
    ∂L/∂b = (1/n) Σᵢ (p̂ᵢ - yᵢ)
    """
    n = len(y_true)
    if n == 0:
        return np.zeros(X.shape[1]), 0.0
    
    # Ensure inputs are numpy arrays
    X = np.asarray(X)
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    
    # Compute prediction error
    error = y_pred_proba - y_true
    
    # Gradient with respect to weights: ∂L/∂w = (1/n) Σᵢ (p̂ᵢ - yᵢ) xᵢ
    grad_w = (1/n) * np.sum(error.reshape(-1, 1) * X, axis=0)
    
    # Gradient with respect to bias: ∂L/∂b = (1/n) Σᵢ (p̂ᵢ - yᵢ)
    grad_b = (1/n) * np.sum(error)
    
    return grad_w, float(grad_b)

def custom_training_step_linear(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: float, 
                               alpha: float, lambda_reg: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """
    Perform one custom training step for linear regression using our formulas.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        True target values
    w : np.ndarray
        Current weight vector
    b : float
        Current bias term
    alpha : float
        Learning rate
    lambda_reg : float
        L2 regularization strength
        
    Returns
    -------
    Tuple[np.ndarray, float, float]
        Updated weights, updated bias, and loss value
    """
    # Forward pass
    y_pred = np.dot(X, w) + b
    
    # Compute loss
    loss = linear_regression_loss(y_true, y_pred, w, b)
    
    # Add regularization if specified
    if lambda_reg > 0:
        loss = l2_regularization_loss(loss, w, lambda_reg)
    
    # Compute gradients
    grad_w, grad_b = compute_gradients_linear_regression(X, y_true, y_pred)
    
    # Update parameters
    w_new = gradient_descent_update_w(w, alpha, grad_w)
    b_new = gradient_descent_update_b(b, alpha, grad_b)
    
    return w_new, b_new, loss

def custom_training_step_logistic(X: np.ndarray, y_true: np.ndarray, w: np.ndarray, b: float, 
                                 alpha: float, lambda_reg: float = 0.0) -> Tuple[np.ndarray, float, float]:
    """
    Perform one custom training step for logistic regression using our formulas.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y_true : np.ndarray
        True binary labels
    w : np.ndarray
        Current weight vector
    b : float
        Current bias term
    alpha : float
        Learning rate
    lambda_reg : float
        L2 regularization strength
        
    Returns
    -------
    Tuple[np.ndarray, float, float]
        Updated weights, updated bias, and loss value
    """
    # Forward pass
    z = np.dot(X, w) + b
    y_pred_proba = 1 / (1 + np.exp(-z))
    
    # Compute loss
    loss = logistic_regression_loss(y_true, y_pred_proba)
    
    # Add regularization if specified
    if lambda_reg > 0:
        loss = l2_regularization_loss(loss, w, lambda_reg)
    
    # Compute gradients
    grad_w, grad_b = compute_gradients_logistic_regression(X, y_true, y_pred_proba)
    
    # Update parameters
    w_new = gradient_descent_update_w(w, alpha, grad_w)
    b_new = gradient_descent_update_b(b, alpha, grad_b)
    
    return w_new, b_new, loss

def log_transform(y, c: float = 1e-9):
    """
    Apply log transform: y'ᵢ = log(yᵢ + c)
    
    Parameters
    ----------
    y : np.ndarray or pd.Series
        Input data, must satisfy (y + c) > 0 for all elements
    c : float, default=1e-9
        Small constant to add before taking log for numerical stability
        
    Returns
    -------
    transformed : same type as input
        Log-transformed values: log(y + c)
        
    Raises
    ------
    ValueError
        If any element satisfies (y + c) ≤ 0
        
    Notes
    -----
    Uses np.log1p(y + (c - 1)) for numerical stability when c=1e-9.
    This is equivalent to log(y + c) but more numerically stable.
    
    Examples
    --------
    >>> y = np.array([1, 2, 3, 4, 5])
    >>> y_log = log_transform(y)
    >>> print(y_log)
    [0.         0.69314718 1.09861229 1.38629436 1.60943791]
    
    >>> y_log = log_transform(y, c=1.0)
    >>> print(y_log)
    [0.69314718 1.09861229 1.38629436 1.60943791 1.79175947]
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(y, 'values'):
        # Pandas Series
        y_array = y.values
        if np.any(y_array + c <= 0):
            raise ValueError(f"Log transform requires (y + c) > 0 for all elements. Found values where (y + {c}) ≤ 0")
        
        if c == 1e-9:
            # Use log1p for numerical stability
            result = np.log1p(y_array + (c - 1))
        else:
            result = np.log(y_array + c)
        
        return pd.Series(result, index=y.index, name=y.name)
    else:
        # Numpy array
        y_array = np.asarray(y)
        if np.any(y_array + c <= 0):
            raise ValueError(f"Log transform requires (y + c) > 0 for all elements. Found values where (y + {c}) ≤ 0")
        
        if c == 1e-9:
            # Use log1p for numerical stability
            result = np.log1p(y_array + (c - 1))
        else:
            result = np.log(y_array + c)
        
        return result

def inverse_log_transform(y_log, c: float = 1e-9):
    """
    Apply inverse log transform: yᵢ = exp(y'ᵢ) - c
    
    Parameters
    ----------
    y_log : np.ndarray or pd.Series
        Log-transformed values
    c : float, default=1e-9
        Small constant that was added before taking log
        
    Returns
    -------
    original : same type as input
        Original values: exp(y_log) - c
        
    Notes
    -----
    Uses np.expm1 for stability when c=1.
    This is the inverse operation of log_transform.
    
    Examples
    --------
    >>> y_log = np.array([0, 0.69314718, 1.09861229, 1.38629436, 1.60943791])
    >>> y_original = inverse_log_transform(y_log)
    >>> print(y_original)
    [0. 1. 2. 3. 4.]
    
    >>> y_original = inverse_log_transform(y_log, c=1.0)
    >>> print(y_original)
    [0. 1. 2. 3. 4.]
    """
    # Handle both pandas Series and numpy arrays
    if hasattr(y_log, 'values'):
        # Pandas Series
        y_log_array = y_log.values
        
        if c == 1:
            # Use expm1 for stability
            result = np.expm1(y_log_array)
        else:
            result = np.exp(y_log_array) - c
        
        return pd.Series(result, index=y_log.index, name=y_log.name)
    else:
        # Numpy array
        y_log_array = np.asarray(y_log)
        
        if c == 1:
            # Use expm1 for stability
            result = np.expm1(y_log_array)
        else:
            result = np.exp(y_log_array) - c
        
        return result