"""
Ensemble methods package for Phase 4.
Contains averaging, weighted, and stacked ensemble implementations.
"""

from .ensemble_methods import (
    averaging_ensemble,
    weighted_ensemble,
    stacked_ensemble
)

__all__ = [
    'averaging_ensemble',
    'weighted_ensemble', 
    'stacked_ensemble'
]