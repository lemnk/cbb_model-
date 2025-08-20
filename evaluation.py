#!/usr/bin/env python3
"""
Phase 3: Model Evaluation Module for CBB Betting ML System

This module implements all evaluation metrics using exact formulas from requirements:
- Classification: Accuracy, Precision, Recall, F1-Score, AUC
- Regression: RMSE, MAE, R²
- Visualization: Confusion Matrix, ROC Curve, Feature Importance
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from models.logistic_regression import LogisticRegressionModel
    from models.random_forest import RandomForestModel
    from models.xgboost_model import XGBoostModel
    from models.neural_network import NeuralNetworkModel
    print("✅ Successfully imported all model modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class ModelEvaluator:
    """
    Comprehensive model evaluation using exact formulas from requirements.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize Model Evaluator.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Results storage
        self.evaluation_results = {}
        
    def accuracy(self, tp: int, tn: int, fp: int, fn: int) -> float:
        """
        Calculate accuracy using exact formula: (tp + tn) / (tp + tn + fp + fn)
        
        Args:
            tp: True positives
            tn: True negatives
            fp: False positives
            fn: False negatives
            
        Returns:
            float: Accuracy value
        """
        return (tp + tn) / (tp + tn + fp + fn)
    
    def precision(self, tp: int, fp: int) -> float:
        """
        Calculate precision using exact formula: tp / (tp + fp)
        
        Args:
            tp: True positives
            fp: False positives
            
        Returns:
            float: Precision value
        """
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    def recall(self, tp: int, fn: int) -> float:
        """
        Calculate recall using exact formula: tp / (tp + fn)
        
        Args:
            tp: True positives
            fn: False negatives
            
        Returns:
            float: Recall value
        """
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    def f1_score(self, tp: int, fp: int, fn: int) -> float:
        """
        Calculate F1-score using exact formula: 2 * (precision * recall) / (precision + recall)
        
        Args:
            tp: True positives
            fp: False positives
            fn: False negatives
            
        Returns:
            float: F1-score value
        """
        p = self.precision(tp, fp)
        r = self.recall(tp, fn)
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate RMSE using exact formula: sqrt(mean((y_true - y_pred)²))
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate MAE using exact formula: mean(|y_true - y_pred|)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate R² using exact formula: 1 - (ss_res / ss_tot)
        
        Where:
        - ss_res = sum((y_true - y_pred)²)
        - ss_tot = sum((y_true - mean(y_true))²)
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            float: R² value
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def evaluate_classification_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Unknown") -> dict:
        """
        Evaluate classification model using exact formulas.
        
        Args:
            model: Trained classification model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            dict: Evaluation results
        """
        print(f"🔄 Evaluating {model_name} (Classification)...")
        
        # Get predictions
        predictions, probabilities = model.predict(X_test)
        
        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (y_test == 1))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))
        
        # Calculate metrics using exact formulas
        accuracy = self.accuracy(tp, tn, fp, fn)
        precision = self.precision(tp, fp)
        recall = self.recall(tp, fn)
        f1 = self.f1_score(tp, fp, fn)
        
        # AUC using sklearn as specified in requirements
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, probabilities)
        
        # Store results
        results = {
            'model_name': model_name,
            'task': 'classification',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            },
            'predictions': predictions,
            'probabilities': probabilities,
            'y_true': y_test
        }
        
        self.evaluation_results[f"{model_name}_classification"] = results
        
        print(f"✅ {model_name} evaluation complete:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   AUC: {auc:.4f}")
        
        return results
    
    def evaluate_regression_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str = "Unknown") -> dict:
        """
        Evaluate regression model using exact formulas.
        
        Args:
            model: Trained regression model
            X_test: Test features
            y_test: Test values
            model_name: Name of the model
            
        Returns:
            dict: Evaluation results
        """
        print(f"🔄 Evaluating {model_name} (Regression)...")
        
        # Get predictions
        predictions, _ = model.predict(X_test)
        
        # Calculate metrics using exact formulas
        rmse = self.rmse(y_test, predictions)
        mae = self.mae(y_test, predictions)
        r2 = self.r2(y_test, predictions)
        
        # Store results
        results = {
            'model_name': model_name,
            'task': 'regression',
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': predictions,
            'y_true': y_test
        }
        
        self.evaluation_results[f"{model_name}_regression"] = results
        
        print(f"✅ {model_name} evaluation complete:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, results: dict, save_path: str = None):
        """
        Plot confusion matrix for classification results.
        
        Args:
            results: Classification evaluation results
            save_path: Path to save the plot
        """
        if results['task'] != 'classification':
            print("❌ Can only plot confusion matrix for classification results")
            return
        
        cm = results['confusion_matrix']
        tp, tn, fp, fn = cm['tp'], cm['tn'], cm['fp'], cm['fn']
        
        # Create confusion matrix array
        cm_array = np.array([[tn, fp], [fn, tp]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'])
        plt.title(f'Confusion Matrix - {results["model_name"]}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curve(self, results: dict, save_path: str = None):
        """
        Plot ROC curve for classification results.
        
        Args:
            results: Classification evaluation results
            save_path: Path to save the plot
        """
        if results['task'] != 'classification':
            print("❌ Can only plot ROC curve for classification results")
            return
        
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(results['y_true'], results['probabilities'])
        auc = results['auc']
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {results["model_name"]}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, model, model_name: str, save_path: str = None):
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Trained model with feature importance
            model_name: Name of the model
            save_path: Path to save the plot
        """
        try:
            feature_importance = model.get_feature_importance()
            
            if feature_importance is not None and len(feature_importance) > 0:
                # Take top 20 features
                top_features = feature_importance.head(20)
                
                plt.figure(figsize=(12, 8))
                if 'importance' in top_features.columns:
                    plt.barh(range(len(top_features)), top_features['importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('Feature Importance')
                elif 'coefficient' in top_features.columns:
                    plt.barh(range(len(top_features)), np.abs(top_features['coefficient']))
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('|Coefficient|')
                
                plt.title(f'Feature Importance - {model_name}')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
                plt.show()
            else:
                print(f"⚠️ No feature importance available for {model_name}")
                
        except Exception as e:
            print(f"⚠️ Could not plot feature importance for {model_name}: {str(e)}")
    
    def plot_regression_predictions(self, results: dict, save_path: str = None):
        """
        Plot regression predictions vs actual values.
        
        Args:
            results: Regression evaluation results
            save_path: Path to save the plot
        """
        if results['task'] != 'regression':
            print("❌ Can only plot regression predictions for regression results")
            return
        
        y_true = results['y_true']
        predictions = results['predictions']
        
        plt.figure(figsize=(10, 8))
        
        # Scatter plot
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, predictions, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Predictions vs Actual - {results["model_name"]}')
        plt.grid(True, alpha=0.3)
        
        # Residuals plot
        plt.subplot(2, 2, 2)
        residuals = y_true - predictions
        plt.scatter(predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # Residuals distribution
        plt.subplot(2, 2, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        # Metrics text
        plt.subplot(2, 2, 4)
        plt.axis('off')
        metrics_text = f"""
        Model: {results['model_name']}
        
        RMSE: {results['rmse']:.4f}
        MAE: {results['mae']:.4f}
        R²: {results['r2']:.4f}
        
        Sample Size: {len(y_true)}
        """
        plt.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_evaluation_report(self) -> str:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            str: Evaluation report
        """
        if not self.evaluation_results:
            return "No evaluation results available."
        
        report = "=" * 80 + "\n"
        report += "CBB BETTING ML SYSTEM - PHASE 3 EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Models evaluated: {len(self.evaluation_results)}\n\n"
        
        # Classification results
        classification_results = {k: v for k, v in self.evaluation_results.items() if v['task'] == 'classification'}
        if classification_results:
            report += "CLASSIFICATION RESULTS:\n"
            report += "-" * 50 + "\n"
            
            for model_name, results in classification_results.items():
                report += f"\n{results['model_name']}:\n"
                report += f"  Accuracy: {results['accuracy']:.4f}\n"
                report += f"  Precision: {results['precision']:.4f}\n"
                report += f"  Recall: {results['recall']:.4f}\n"
                report += f"  F1-Score: {results['f1_score']:.4f}\n"
                report += f"  AUC: {results['auc']:.4f}\n"
                
                cm = results['confusion_matrix']
                report += f"  Confusion Matrix:\n"
                report += f"    TP: {cm['tp']}, TN: {cm['tn']}, FP: {cm['fp']}, FN: {cm['fn']}\n"
        
        # Regression results
        regression_results = {k: v for k, v in self.evaluation_results.items() if v['task'] == 'regression'}
        if regression_results:
            report += "\n" + "=" * 80 + "\n"
            report += "REGRESSION RESULTS:\n"
            report += "=" * 80 + "\n"
            
            for model_name, results in regression_results.items():
                report += f"\n{results['model_name']}:\n"
                report += f"  RMSE: {results['rmse']:.4f}\n"
                report += f"  MAE: {results['mae']:.4f}\n"
                report += f"  R²: {results['r2']:.4f}\n"
        
        # Formula verification
        report += "\n" + "=" * 80 + "\n"
        report += "FORMULA VERIFICATION:\n"
        report += "=" * 80 + "\n"
        report += "• Accuracy = (tp + tn) / (tp + tn + fp + fn) ✓\n"
        report += "• Precision = tp / (tp + fp) ✓\n"
        report += "• Recall = tp / (tp + fn) ✓\n"
        report += "• F1-Score = 2 * (precision * recall) / (precision + recall) ✓\n"
        report += "• RMSE = sqrt(mean((y_true - y_pred)²)) ✓\n"
        report += "• MAE = mean(|y_true - y_pred|) ✓\n"
        report += "• R² = 1 - (ss_res / ss_tot) ✓\n"
        report += "• AUC using sklearn.metrics.roc_auc_score ✓\n"
        
        return report
    
    def save_evaluation_results(self, filepath: str):
        """
        Save evaluation results to file.
        
        Args:
            filepath: Path to save results
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert results to DataFrame for easy saving
        results_data = []
        
        for model_name, results in self.evaluation_results.items():
            if results['task'] == 'classification':
                results_data.append({
                    'model_name': results['model_name'],
                    'task': results['task'],
                    'accuracy': results['accuracy'],
                    'precision': results['precision'],
                    'recall': results['recall'],
                    'f1_score': results['f1_score'],
                    'auc': results['auc'],
                    'tp': results['confusion_matrix']['tp'],
                    'tn': results['confusion_matrix']['tn'],
                    'fp': results['confusion_matrix']['fp'],
                    'fn': results['confusion_matrix']['fn']
                })
            else:  # regression
                results_data.append({
                    'model_name': results['model_name'],
                    'task': results['task'],
                    'rmse': results['rmse'],
                    'mae': results['mae'],
                    'r2': results['r2']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(filepath, index=False)
        print(f"✅ Evaluation results saved to: {filepath}")
        
        # Save summary report
        report_path = filepath.replace('.csv', '_report.txt')
        with open(report_path, 'w') as f:
            f.write(self.generate_evaluation_report())
        print(f"✅ Evaluation report saved to: {report_path}")

def main():
    """
    Main function to demonstrate evaluation functionality.
    """
    print("🧪 Model Evaluation Module Demo")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Test metric calculations
    print("\n📊 Testing Metric Calculations:")
    print(f"Accuracy (TP=80, TN=70, FP=20, FN=30): {evaluator.accuracy(80, 70, 20, 30):.4f}")
    print(f"Precision (TP=80, FP=20): {evaluator.precision(80, 20):.4f}")
    print(f"Recall (TP=80, FN=30): {evaluator.recall(80, 30):.4f}")
    print(f"F1-Score (TP=80, FP=20, FN=30): {evaluator.f1_score(80, 20, 30):.4f}")
    
    # Test regression metrics
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    
    print(f"\n📈 Testing Regression Metrics:")
    print(f"RMSE: {evaluator.rmse(y_true, y_pred):.4f}")
    print(f"MAE: {evaluator.mae(y_true, y_pred):.4f}")
    print(f"R²: {evaluator.r2(y_true, y_pred):.4f}")
    
    print("\n✅ Evaluation module ready for use!")

if __name__ == "__main__":
    main()