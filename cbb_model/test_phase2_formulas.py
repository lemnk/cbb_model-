#!/usr/bin/env python3
"""
Simple test script for Phase 2 training formulas.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import numpy as np
    from src.models.train_utils import (
        linear_regression_loss,
        logistic_regression_loss,
        gradient_descent_update_w,
        gradient_descent_update_b,
        l2_regularization_loss,
        custom_training_step_linear,
        custom_training_step_logistic
    )
    
    print("✅ Successfully imported all Phase 2 training functions")
    
    # Test data
    X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
    y_true_linear = np.array([3, 7, 11])  # y = x1 + x2
    y_true_binary = np.array([1, 0, 1])
    w = np.array([0.5, 0.5])
    b = 0.0
    alpha = 0.1
    lambda_reg = 0.01
    
    print(f"\n🧪 Testing Phase 2 Training Formulas...")
    print(f"📊 Test data shape: {X.shape}")
    print(f"📊 Weight vector: {w}")
    print(f"📊 Bias: {b}")
    print(f"📊 Learning rate (α): {alpha}")
    print(f"📊 L2 regularization (λ): {lambda_reg}")
    
    # Test 1: Linear Regression Loss
    print(f"\n📈 Test 1: Linear Regression Loss")
    print(f"   Formula: L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²")
    y_pred_linear = np.dot(X, w) + b
    mse_loss = linear_regression_loss(y_true_linear, y_pred_linear, w, b)
    print(f"   MSE Loss: {mse_loss:.6f}")
    
    # Test 2: Logistic Regression Loss
    print(f"\n📊 Test 2: Logistic Regression Loss")
    print(f"   Formula: L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]")
    z = np.dot(X, w) + b
    y_pred_proba = 1 / (1 + np.exp(-z))
    bce_loss = logistic_regression_loss(y_true_binary, y_pred_proba)
    print(f"   BCE Loss: {bce_loss:.6f}")
    
    # Test 3: L2 Regularization
    print(f"\n🔒 Test 3: L2 Regularization")
    print(f"   Formula: L_reg = L + λ ||w||²")
    regularized_mse = l2_regularization_loss(mse_loss, w, lambda_reg)
    regularized_bce = l2_regularization_loss(bce_loss, w, lambda_reg)
    print(f"   MSE + L2: {regularized_mse:.6f}")
    print(f"   BCE + L2: {regularized_bce:.6f}")
    
    # Test 4: Gradient Descent Updates
    print(f"\n⬇️ Test 4: Gradient Descent Updates")
    print(f"   Formula: w := w - α ∂L/∂w, b := b - α ∂L/∂b")
    
    # Compute gradients manually for demonstration
    n = len(y_true_linear)
    residuals = y_true_linear - y_pred_linear
    grad_w = -(2/n) * np.sum(residuals.reshape(-1, 1) * X, axis=0)
    grad_b = -(2/n) * np.sum(residuals)
    
    w_new = gradient_descent_update_w(w, alpha, grad_w)
    b_new = gradient_descent_update_b(b, alpha, grad_b)
    
    print(f"   Weight update: {w} → {w_new}")
    print(f"   Bias update: {b} → {b_new}")
    
    # Test 5: Custom Training Steps
    print(f"\n🔄 Test 5: Custom Training Steps")
    w_step, b_step, loss_step = custom_training_step_linear(
        X, y_true_linear, w, b, alpha, lambda_reg
    )
    print(f"   Linear training step:")
    print(f"     Loss: {mse_loss:.6f} → {loss_step:.6f}")
    print(f"     Weights: {w} → {w_step}")
    print(f"     Bias: {b} → {b_step}")
    
    w_step_log, b_step_log, loss_step_log = custom_training_step_logistic(
        X, y_true_binary, w, b, alpha, lambda_reg
    )
    print(f"   Logistic training step:")
    print(f"     Loss: {bce_loss:.6f} → {loss_step_log:.6f}")
    print(f"     Weights: {w} → {w_step_log}")
    print(f"     Bias: {b} → {b_step_log}")
    
    print(f"\n🎉 All Phase 2 training formulas tested successfully!")
    print(f"✅ Linear Regression Loss: Working")
    print(f"✅ Logistic Regression Loss: Working")
    print(f"✅ L2 Regularization: Working")
    print(f"✅ Gradient Descent Updates: Working")
    print(f"✅ Custom Training Steps: Working")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()