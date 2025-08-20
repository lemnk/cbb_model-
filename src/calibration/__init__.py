"""
Calibration package for Phase 4.
Contains Platt scaling, isotonic regression, and calibration metrics.
"""

from .calibration_methods import (
    platt_scaling,
    isotonic_calibration,
    brier_score
)

__all__ = [
    'platt_scaling',
    'isotonic_calibration',
    'brier_score'
]