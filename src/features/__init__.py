"""
Feature Engineering Package for NCAA CBB Betting ML System (Phase 2).

This package provides modular feature engineering capabilities including:
- Team performance and efficiency features
- Player availability and impact features
- Market efficiency and odds-based features
- Dynamic situational features
- Orchestrated feature pipeline for ML model preparation
"""

__version__ = "2.0.0"
__author__ = "NCAA CBB Betting ML Team"

# Import main feature engineering classes
from .team_features import TeamFeatures
from .player_features import PlayerFeatures
from .market_features import MarketFeatures
from .dynamic_features import DynamicFeatures
from .feature_pipeline import FeaturePipeline

# Import utility functions
from .feature_utils import normalize_series

__all__ = [
    'TeamFeatures',
    'PlayerFeatures', 
    'MarketFeatures',
    'DynamicFeatures',
    'FeaturePipeline',
    'normalize_series'
]