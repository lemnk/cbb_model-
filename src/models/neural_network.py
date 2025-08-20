import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

class NeuralNetwork(nn.Module):
    """
    PyTorch Neural Network for both classification and regression tasks.
    
    Classification: Game outcome prediction (win/loss)
    Regression: Point differential prediction
    """
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], task='classification', dropout=0.2):
        super(NeuralNetwork, self).__init__()
        
        self.task = task
        self.input_size = input_size
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        # Output layer
        if task == 'classification':
            layers.append(nn.Linear(prev_size, 2))  # 2 classes: win/loss
        else:
            layers.append(nn.Linear(prev_size, 1))  # 1 output: point differential
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class NeuralNetworkModel:
    """
    Neural Network model wrapper for training and evaluation.
    """
    
    def __init__(self, task='classification', random_state=42, **kwargs):
        """
        Initialize Neural Network model.
        
        Args:
            task: 'classification' or 'regression'
            random_state: Random seed for reproducibility
            **kwargs: Additional parameters
        """
        self.task = task
        self.random_state = random_state
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
        self.history = {'train_loss': [], 'val_loss': []}
        
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
    
    def _create_data_loaders(self, X_train, y_train, X_val=None, y_val=None, batch_size=32):
        """
        Create PyTorch DataLoaders for training and validation.
        """
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train) if self.task == 'classification' else torch.FloatTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val) if self.task == 'classification' else torch.FloatTensor(y_val)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Fit the Neural Network model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize model
        input_size = X_train.shape[1]
        self.model = NeuralNetwork(input_size, task=self.task).to(self.device)
        
        # Loss function and optimizer
        if self.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(
            X_train_scaled, y_train, X_val_scaled, y_val, batch_size
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.task == 'classification':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            self.history['train_loss'].append(avg_train_loss)
            
            # Validation
            val_loss = None
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        
                        if self.task == 'classification':
                            loss = criterion(outputs, batch_y)
                        else:
                            loss = criterion(outputs.squeeze(), batch_y.float())
                        
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                self.history['val_loss'].append(avg_val_loss)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Progress reporting
            if epoch % 10 == 0:
                val_str = f", Val Loss: {avg_val_loss:.4f}" if val_loss is not None else ""
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}{val_str}")
        
        self.is_fitted = True
        
        if val_loader is not None:
            print(f"✅ Neural Network ({self.task}) fitted. Best validation loss: {best_val_loss:.4f}")
        else:
            print(f"✅ Neural Network ({self.task}) fitted.")
    
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
        
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            if self.task == 'classification':
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                return predictions.cpu().numpy(), probabilities[:, 1].cpu().numpy()
            else:
                predictions = outputs.squeeze()
                return predictions.cpu().numpy(), None
    
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
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'task': self.task,
            'input_size': self.model.input_size,
            'is_fitted': self.is_fitted,
            'history': self.history
        }
        
        joblib.dump(model_data, filepath)
        print(f"✅ Neural Network model saved to: {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        # Reconstruct model
        self.model = NeuralNetwork(model_data['input_size'], task=model_data['task']).to(self.device)
        self.model.load_state_dict(model_data['model_state_dict'])
        
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.task = model_data['task']
        self.is_fitted = model_data['is_fitted']
        self.history = model_data['history']
        
        print(f"✅ Neural Network model loaded from: {filepath}")
    
    def get_training_history(self):
        """
        Get training history for plotting.
        
        Returns:
            dict: Training and validation loss history
        """
        return self.history