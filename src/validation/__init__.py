"""
Validation package for Phase 4.
Contains walk-forward validation and backtesting functionality.
"""

from .walk_forward import (
    walk_forward_split,
    WalkForwardValidator
)

__all__ = [
    'walk_forward_split',
    'WalkForwardValidator'
]