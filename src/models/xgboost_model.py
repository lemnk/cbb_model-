import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class XGBoostModel:
    """
    XGBoost model for both classification and regression tasks.
    
    Classification: Game outcome prediction (win/loss)
    Regression: Point differential prediction
    """
    
    def __init__(self, task='classification', random_state=42, **kwargs):
        """
        Initialize XGBoost model.
        
        Args:
            task: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters for XGBoost
        """
        self.task = task
        self.random_state = random_state
        
        if task == 'classification':
            self.model = xgb.XGBClassifier(
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                **kwargs
            )
        elif task == 'regression':
            self.model = xgb.XGBRegressor(
                random_state=random_state,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='rmse',
                **kwargs
            )
        else:
            raise ValueError("Task must be 'classification' or 'regression'")
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        
    def prepare_features(self, df, target_col, exclude_cols=None):
        """
        Prepare features for training.
        
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
        Fit the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Fit model
        if self.task == 'classification':
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10 if eval_set else None,
                verbose=False
            )
        else:
            self.model.fit(
                X_train_scaled, y_train,
                eval_set=eval_set,
                early_stopping_rounds=10 if eval_set else None,
                verbose=False
            )
        
        # Validation if provided
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            if self.task == 'classification':
                val_score = self.model.score(X_val_scaled, y_val)
                print(f"✅ XGBoost ({self.task}) fitted. Validation accuracy: {val_score:.4f}")
            else:
                val_score = self.model.score(X_val_scaled, y_val)
                print(f"✅ XGBoost ({self.task}) fitted. Validation R²: {val_score:.4f}")
        else:
            print(f"✅ XGBoost ({self.task}) fitted.")
        
        self.is_fitted = True
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            predictions: Predictions
            probabilities: Probability predictions (classification only)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if self.task == 'classification':
            probabilities = self.model.predict_proba(X_scaled)[:, 1]
            return predictions, probabilities
        else:
            return predictions, None
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance using exact formulas from requirements.
        
        Args:
            X: Feature matrix
            y_true: True labels/values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        predictions, probabilities = self.predict(X)
        
        if self.task == 'classification':
            return self._evaluate_classification(y_true, predictions, probabilities)
        else:
            return self._evaluate_regression(y_true, predictions)
    
    def _evaluate_classification(self, y_true, predictions, probabilities):
        """
        Evaluate classification performance using exact formulas.
        """
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
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'confusion_matrix': {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
        }
    
    def _evaluate_regression(self, y_true, predictions):
        """
        Evaluate regression performance using exact formulas.
        """
        # Calculate metrics using exact formulas from requirements
        rmse = np.sqrt(np.mean((y_true - predictions) ** 2))
        mae = np.mean(np.abs(y_true - predictions))
        
        # R² calculation
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def get_feature_importance(self):
        """
        Get feature importance based on XGBoost feature importances.
        
        Returns:
            DataFrame: Feature importance sorted by importance value
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        return pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
    
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
            'task': self.task,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ XGBoost model saved to: {filepath}")
    
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
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        
        print(f"✅ XGBoost model loaded from: {filepath}")
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, param_grid=None):
        """
        Perform hyperparameter tuning using validation set.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            param_grid: Parameter grid for tuning
            
        Returns:
            dict: Best parameters found
        """
        from sklearn.model_selection import GridSearchCV
        
        if param_grid is None:
            if self.task == 'classification':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            else:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=3, scoring='accuracy' if self.task == 'classification' else 'r2',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        print(f"✅ Hyperparameter tuning complete. Best score: {grid_search.best_score_:.4f}")
        print(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_params_