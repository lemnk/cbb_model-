"""
Feature Engineering Package for CBB Betting ML System (Phase 2).

This package provides comprehensive feature engineering capabilities including:
- Team context and performance features
- Dynamic game flow and momentum features
- Player availability and injury features
- Market efficiency and odds-based features
- Orchestrated feature pipeline for ML model preparation
"""

__version__ = "2.0.0"
__author__ = "CBB Betting ML Team"

# Import main feature engineering classes
from .team_features import TeamFeatureEngineer, create_team_feature_engineer
from .dynamic_features import DynamicFeatureEngineer, create_dynamic_feature_engineer
from .player_features import PlayerFeatureEngineer, create_player_feature_engineer
from .market_features import MarketFeatureEngineer, create_market_feature_engineer
from .feature_pipeline import FeaturePipeline, create_feature_pipeline
from .feature_utils import FeatureUtils, create_feature_utils

# Import utility functions
from .feature_utils import (
    calculate_rolling_averages,
    compute_momentum_index,
    encode_run_lengths,
    calculate_line_drift,
    compute_implied_probability_edge,
    validate_feature_set
)

__all__ = [
    # Main feature engineers
    'TeamFeatureEngineer',
    'create_team_feature_engineer',
    'DynamicFeatureEngineer', 
    'create_dynamic_feature_engineer',
    'PlayerFeatureEngineer',
    'create_player_feature_engineer',
    'MarketFeatureEngineer',
    'create_market_feature_engineer',
    'FeaturePipeline',
    'create_feature_pipeline',
    'FeatureUtils',
    'create_feature_utils',
    
    # Utility functions
    'calculate_rolling_averages',
    'compute_momentum_index',
    'encode_run_lengths',
    'calculate_line_drift',
    'compute_implied_probability_edge',
    'validate_feature_set'
]