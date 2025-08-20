"""
Phase 3: Model Training Package for CBB Betting ML System

This package contains all machine learning models for:
- Classification: Game outcome prediction (win/loss)
- Regression: Point differential prediction

Models implemented:
- Logistic Regression (classification only)
- Random Forest (classification + regression)
- XGBoost (classification + regression)
- Neural Network (classification + regression, PyTorch)
"""

from .logistic_regression import LogisticRegressionModel
from .random_forest import RandomForestModel
from .xgboost_model import XGBoostModel
from .neural_network import NeuralNetworkModel

__all__ = [
    'LogisticRegressionModel',
    'RandomForestModel', 
    'XGBoostModel',
    'NeuralNetworkModel'
]

__version__ = '1.0.0'
__author__ = 'CBB Betting ML System Team'