"""
Performance Monitoring module for Phase 5: Monitoring & CI/CD.
Continuously evaluates model metrics and profitability for the CBB Betting ML System.
"""

from typing import Dict, List, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """Data class for storing individual metric results."""
    value: float
    threshold: float
    status: str  # 'PASS', 'WARNING', 'ALERT'
    details: Dict[str, Any] = None


class PerformanceMonitor:
    """
    Performance monitoring system for ML model evaluation.
    
    This class computes key performance metrics and compares them against
    configurable thresholds to generate alerts for model performance issues.
    
    Metrics computed:
    - Accuracy, Precision, Recall, F1-Score
    - Log Loss, Brier Score, ROC-AUC
    - Expected Value (profitability metric)
    """
    
    def __init__(self, thresholds: Dict[str, float]):
        """
        Initialize PerformanceMonitor with metric thresholds.
        
        Args:
            thresholds: Dictionary of metric names and their threshold values
        """
        self.thresholds = thresholds.copy()
        self._validate_thresholds()
        
        logger.info(f"PerformanceMonitor initialized with {len(self.thresholds)} metric thresholds")
        logger.info(f"Thresholds: {self.thresholds}")
    
    def _validate_thresholds(self):
        """Validate that all required thresholds are provided."""
        required_metrics = [
            'accuracy', 'log_loss', 'brier_score', 'precision', 
            'recall', 'f1', 'roc_auc', 'expected_value'
        ]
        
        missing_metrics = [metric for metric in required_metrics if metric not in self.thresholds]
        if missing_metrics:
            raise ValueError(f"Missing required thresholds: {missing_metrics}")
        
        # Validate threshold values
        for metric, threshold in self.thresholds.items():
            if not isinstance(threshold, (int, float)):
                raise ValueError(f"Threshold for {metric} must be numeric, got {type(threshold)}")
    
    def _compute_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
        """
        Compute confusion matrix components.
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels (binary)
            
        Returns:
            Dictionary with TP, TN, FP, FN counts
        """
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy metric.
        
        Formula: Accuracy = (TP + TN) / (TP + TN + FP + FN)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: Accuracy score between 0 and 1
        """
        cm = self._compute_confusion_matrix(y_true, y_pred)
        total = cm['tp'] + cm['tn'] + cm['fp'] + cm['fn']
        
        if total == 0:
            return 0.0
        
        accuracy = (cm['tp'] + cm['tn']) / total
        return float(accuracy)
    
    def _compute_log_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute log loss metric.
        
        Formula: Log Loss = -(1/n) Σ [y log(p) + (1-y) log(1-p)]
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            float: Log loss score (lower is better)
        """
        # Handle edge cases
        epsilon = 1e-15
        y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)
        
        # Compute log loss: -(1/n) Σ [y log(p) + (1-y) log(1-p)]
        n = len(y_true)
        if n == 0:
            return 0.0
        
        log_loss = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        return float(log_loss)
    
    def _compute_brier_score(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute Brier score metric.
        
        Formula: Brier Score = (1/n) Σ (p - y)²
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            float: Brier score (lower is better)
        """
        n = len(y_true)
        if n == 0:
            return 0.0
        
        # Compute Brier score: (1/n) Σ (p - y)²
        brier_score = np.mean((y_pred_proba - y_true) ** 2)
        return float(brier_score)
    
    def _compute_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute precision metric.
        
        Formula: Precision = TP / (TP + FP)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: Precision score between 0 and 1
        """
        cm = self._compute_confusion_matrix(y_true, y_pred)
        denominator = cm['tp'] + cm['fp']
        
        if denominator == 0:
            return 0.0
        
        precision = cm['tp'] / denominator
        return float(precision)
    
    def _compute_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute recall metric.
        
        Formula: Recall = TP / (TP + FN)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: Recall score between 0 and 1
        """
        cm = self._compute_confusion_matrix(y_true, y_pred)
        denominator = cm['tp'] + cm['fn']
        
        if denominator == 0:
            return 0.0
        
        recall = cm['tp'] / denominator
        return float(recall)
    
    def _compute_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute F1 score metric.
        
        Formula: F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            
        Returns:
            float: F1 score between 0 and 1
        """
        precision = self._compute_precision(y_true, y_pred)
        recall = self._compute_recall(y_true, y_pred)
        
        denominator = precision + recall
        if denominator == 0:
            return 0.0
        
        f1 = 2 * (precision * recall) / denominator
        return float(f1)
    
    def _compute_roc_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """
        Compute ROC-AUC score using sklearn.
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities for positive class
            
        Returns:
            float: ROC-AUC score between 0 and 1
        """
        try:
            # Handle edge cases
            if len(np.unique(y_true)) < 2:
                logger.warning("ROC-AUC requires at least 2 classes")
                return 0.5
            
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            return float(roc_auc)
        except Exception as e:
            logger.error(f"Error computing ROC-AUC: {e}")
            return 0.5
    
    def _compute_expected_value(self, y_true: np.ndarray, y_pred_proba: np.ndarray, odds: np.ndarray) -> float:
        """
        Compute expected value (profitability metric).
        
        Formula: Expected Value = (1/n) Σ [p × odds - (1-p)]
        
        Args:
            y_true: Ground truth labels
            y_pred_proba: Predicted probabilities for positive class
            odds: Decimal odds for the positive outcome
            
        Returns:
            float: Expected value (positive = profitable, negative = unprofitable)
        """
        n = len(y_true)
        if n == 0:
            return 0.0
        
        # Compute expected value: (1/n) Σ [p × odds - (1-p)]
        # This represents the expected profit per bet
        expected_value = np.mean(y_pred_proba * odds - (1 - y_pred_proba))
        return float(expected_value)
    
    def _determine_status(self, value: float, threshold: float, metric_name: str) -> str:
        """
        Determine the status of a metric based on its value and threshold.
        
        Args:
            value: Computed metric value
            threshold: Threshold value for the metric
            metric_name: Name of the metric for logging
            
        Returns:
            str: Status ('PASS', 'WARNING', or 'ALERT')
        """
        # For metrics where lower is better (log_loss, brier_score)
        if metric_name in ['log_loss', 'brier_score']:
            if value <= threshold:
                return 'PASS'
            elif value <= threshold * 1.2:  # 20% tolerance
                return 'WARNING'
            else:
                return 'ALERT'
        
        # For metrics where higher is better (accuracy, precision, recall, f1, roc_auc)
        elif metric_name in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            if value >= threshold:
                return 'PASS'
            elif value >= threshold * 0.8:  # 20% tolerance
                return 'WARNING'
            else:
                return 'ALERT'
        
        # For expected_value (can be positive or negative)
        elif metric_name == 'expected_value':
            if value >= threshold:
                return 'PASS'
            elif value >= threshold - 0.02:  # Small tolerance for negative EV
                return 'WARNING'
            else:
                return 'ALERT'
        
        # Default case
        else:
            return 'PASS'
    
    def evaluate(self, y_true: Union[List, np.ndarray], 
                y_pred_proba: Union[List, np.ndarray], 
                odds: Union[List, np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate model performance against all metrics and thresholds.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class (0 to 1)
            odds: Decimal odds for the positive outcome
            
        Returns:
            Dictionary containing metric results with values, thresholds, and status
        """
        logger.info("Starting performance evaluation")
        
        # Convert inputs to numpy arrays
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        odds = np.array(odds)
        
        # Validate inputs
        if len(y_true) != len(y_pred_proba) or len(y_true) != len(odds):
            raise ValueError("All input arrays must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        # Validate input values
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError("y_true must contain only 0 and 1 values")
        
        if not np.all((y_pred_proba >= 0) & (y_pred_proba <= 1)):
            raise ValueError("y_pred_proba must contain values between 0 and 1")
        
        if not np.all(odds > 1):
            raise ValueError("odds must contain values greater than 1")
        
        logger.info(f"Evaluating performance for {len(y_true)} predictions")
        
        # Convert probabilities to binary predictions for classification metrics
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Compute all metrics
        metrics = {}
        
        # Accuracy
        accuracy_value = self._compute_accuracy(y_true, y_pred)
        accuracy_status = self._determine_status(accuracy_value, self.thresholds['accuracy'], 'accuracy')
        metrics['accuracy'] = {
            'value': accuracy_value,
            'threshold': self.thresholds['accuracy'],
            'status': accuracy_status
        }
        
        # Log Loss
        log_loss_value = self._compute_log_loss(y_true, y_pred_proba)
        log_loss_status = self._determine_status(log_loss_value, self.thresholds['log_loss'], 'log_loss')
        metrics['log_loss'] = {
            'value': log_loss_value,
            'threshold': self.thresholds['log_loss'],
            'status': log_loss_status
        }
        
        # Brier Score
        brier_score_value = self._compute_brier_score(y_true, y_pred_proba)
        brier_score_status = self._determine_status(brier_score_value, self.thresholds['brier_score'], 'brier_score')
        metrics['brier_score'] = {
            'value': brier_score_value,
            'threshold': self.thresholds['brier_score'],
            'status': brier_score_status
        }
        
        # Precision
        precision_value = self._compute_precision(y_true, y_pred)
        precision_status = self._determine_status(precision_value, self.thresholds['precision'], 'precision')
        metrics['precision'] = {
            'value': precision_value,
            'threshold': self.thresholds['precision'],
            'status': precision_status
        }
        
        # Recall
        recall_value = self._compute_recall(y_true, y_pred)
        recall_status = self._determine_status(recall_value, self.thresholds['recall'], 'recall')
        metrics['recall'] = {
            'value': recall_value,
            'threshold': self.thresholds['recall'],
            'status': recall_status
        }
        
        # F1 Score
        f1_value = self._compute_f1_score(y_true, y_pred)
        f1_status = self._determine_status(f1_value, self.thresholds['f1'], 'f1')
        metrics['f1'] = {
            'value': f1_value,
            'threshold': self.thresholds['f1'],
            'status': f1_status
        }
        
        # ROC-AUC
        roc_auc_value = self._compute_roc_auc(y_true, y_pred_proba)
        roc_auc_status = self._determine_status(roc_auc_value, self.thresholds['roc_auc'], 'roc_auc')
        metrics['roc_auc'] = {
            'value': roc_auc_value,
            'threshold': self.thresholds['roc_auc'],
            'status': roc_auc_status
        }
        
        # Expected Value
        expected_value_value = self._compute_expected_value(y_true, y_pred_proba, odds)
        expected_value_status = self._determine_status(expected_value_value, self.thresholds['expected_value'], 'expected_value')
        metrics['expected_value'] = {
            'value': expected_value_value,
            'threshold': self.thresholds['expected_value'],
            'status': expected_value_status
        }
        
        # Log results
        alert_count = sum(1 for metric in metrics.values() if metric['status'] == 'ALERT')
        warning_count = sum(1 for metric in metrics.values() if metric['status'] == 'WARNING')
        
        logger.info(f"Performance evaluation complete: {alert_count} alerts, {warning_count} warnings")
        
        return metrics
    
    def get_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """
        Generate a human-readable performance summary.
        
        Args:
            results: Results from evaluate method
            
        Returns:
            String containing formatted performance summary
        """
        summary = []
        summary.append("=" * 70)
        summary.append("PERFORMANCE MONITORING SUMMARY")
        summary.append("=" * 70)
        
        # Count statuses
        status_counts = {}
        for metric_name, result in results.items():
            status = result['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Overall status
        if status_counts.get('ALERT', 0) > 0:
            overall_status = "🚨 PERFORMANCE ALERTS DETECTED"
        elif status_counts.get('WARNING', 0) > 0:
            overall_status = "⚠️ PERFORMANCE WARNINGS DETECTED"
        else:
            overall_status = "✅ ALL METRICS PASSING"
        
        summary.append(f"Overall Status: {overall_status}")
        summary.append("")
        
        # Status summary
        summary.append("Status Summary:")
        for status in ['PASS', 'WARNING', 'ALERT']:
            count = status_counts.get(status, 0)
            if count > 0:
                icon = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "🚨"
                summary.append(f"  {icon} {status}: {count} metrics")
        summary.append("")
        
        # Metric details
        summary.append("Metric Details:")
        summary.append("-" * 50)
        
        for metric_name, result in results.items():
            status_icon = "✅" if result['status'] == "PASS" else "⚠️" if result['status'] == "WARNING" else "🚨"
            summary.append(f"{status_icon} {metric_name.upper()}:")
            summary.append(f"    Value: {result['value']:.6f}")
            summary.append(f"    Threshold: {result['threshold']:.6f}")
            summary.append(f"    Status: {result['status']}")
            summary.append("")
        
        summary.append("=" * 70)
        
        return "\n".join(summary)


# Example usage and testing
if __name__ == "__main__":
    # Sample thresholds
    sample_thresholds = {
        'accuracy': 0.55,
        'log_loss': 0.7,
        'brier_score': 0.25,
        'precision': 0.5,
        'recall': 0.5,
        'f1': 0.5,
        'roc_auc': 0.6,
        'expected_value': 0.0
    }
    
    print("Testing Performance Monitor...")
    print()
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate realistic predictions and outcomes
    y_true = np.random.binomial(1, 0.6, n_samples)  # 60% positive class
    y_pred_proba = np.random.beta(2, 1, n_samples)  # Beta distribution for probabilities
    odds = np.random.uniform(1.5, 3.0, n_samples)   # Realistic odds range
    
    # Test performance monitor
    monitor = PerformanceMonitor(sample_thresholds)
    results = monitor.evaluate(y_true, y_pred_proba, odds)
    
    # Print summary
    summary = monitor.get_summary(results)
    print(summary)