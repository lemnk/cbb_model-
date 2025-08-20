import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

class LogisticRegressionModel:
    """
    Logistic Regression model for game outcome prediction (classification).
    
    Implements the exact formula: p̂ᵢ = σ(wᵀxᵢ + b), σ(z) = 1/(1 + e⁻ᶻ)
    """
    
    def __init__(self, random_state=42):
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def logistic_prediction(self, x, w, b):
        """
        Implement the exact logistic prediction formula.
        
        Args:
            x: Feature vector
            w: Weight vector
            b: Bias term
            
        Returns:
            Probability prediction: σ(wᵀx + b) = 1/(1 + e⁻ᶻ)
        """
        z = np.dot(w, x) + b
        return 1 / (1 + np.exp(-z))
    
    def prepare_features(self, df, target_col='won', exclude_cols=None):
        """
        Prepare features for training, excluding non-feature columns.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            exclude_cols: Columns to exclude from features
            
        Returns:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
        """
        if exclude_cols is None:
            exclude_cols = ['game_id', 'date', 'team', 'opponent', 'points', 'opponent_points']
        
        # Add target_col to exclude list if not already there
        if target_col not in exclude_cols:
            exclude_cols.append(target_col)
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols].values
        y = df[target_col].values
        
        return X, y, feature_cols
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the logistic regression model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Fit model
        self.model.fit(X_train_scaled, y_train)
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_score = self.model.score(X_val_scaled, y_val)
            print(f"✅ Logistic Regression fitted. Validation accuracy: {val_score:.4f}")
        else:
            print("✅ Logistic Regression fitted.")
        
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Binary predictions
            probabilities: Probability predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance using exact formulas from requirements.
        
        Args:
            X: Feature matrix
            y_true: True labels
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions, probabilities = self.predict(X)
        
        # Calculate confusion matrix components
        tp = np.sum((predictions == 1) & (y_true == 1))
        tn = np.sum((predictions == 0) & (y_true == 0))
        fp = np.sum((predictions == 1) & (y_true == 0))
        fn = np.sum((predictions == 0) & (y_true == 1))
        
        # Calculate metrics using exact formulas from requirements
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # AUC using sklearn as specified
        auc = roc_auc_score(y_true, probabilities)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            },
            'feature_importance': feature_importance
        }
    
    def save_model(self, filepath):
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✅ Model loaded from: {filepath}")
    
    def get_feature_importance(self):
        """
        Get feature importance based on coefficients.
        
        Returns:
            DataFrame: Feature importance sorted by absolute coefficient value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0],
            'abs_coefficient': np.abs(self.model.coef_[0])
        }).sort_values('abs_coefficient', ascending=False)