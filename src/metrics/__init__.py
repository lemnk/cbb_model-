"""
Metrics package for Phase 4: Model Optimization & Deployment.
Contains all evaluation metrics, calibration functions, and ROI calculations.
"""

from .core_metrics import (
    roc_auc_score,
    log_loss,
    expected_calibration_error,
    roi,
    brier_score
)

__all__ = [
    'roc_auc_score',
    'log_loss', 
    'expected_calibration_error',
    'roi',
    'brier_score'
]