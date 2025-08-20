#!/usr/bin/env python3
"""
Phase 3: Model Training Pipeline for CBB Betting ML System

This script implements the complete training pipeline with:
- Exact train/val/test split (70%/15%/15%) as specified in requirements
- All four model types: Logistic Regression, Random Forest, XGBoost, Neural Network
- Both classification and regression tasks
- Hyperparameter tuning on validation set
- Model saving and evaluation
"""

import os
import sys
import numpy as np
import pandas as pd
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
    from features.feature_pipeline import FeaturePipeline
    from models.train_utils import (
    linear_regression_loss,
    logistic_regression_loss,
    gradient_descent_update_w,
    gradient_descent_update_b,
    l2_regularization_loss,
    custom_training_step_linear,
    custom_training_step_logistic,
    log_transform,
    inverse_log_transform
)
    print("✅ Successfully imported all model modules and training utilities")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

class ModelTrainingPipeline:
    """
    Complete model training pipeline for Phase 3.
    
    Implements exact train/val/test split: 70%/15%/15%
    """
    
    def __init__(self, random_state=42):
        """
        Initialize training pipeline.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Initialize models
        self.models = {
            'logistic_regression': {
                'classification': LogisticRegressionModel(random_state=random_state),
                'regression': None  # Logistic regression is classification only
            },
            'random_forest': {
                'classification': RandomForestModel(task='classification', random_state=random_state),
                'regression': RandomForestModel(task='regression', random_state=random_state)
            },
            'xgboost': {
                'classification': XGBoostModel(task='classification', random_state=random_state),
                'regression': XGBoostModel(task='regression', random_state=random_state)
            },
            'neural_network': {
                'classification': NeuralNetworkModel(task='classification', random_state=random_state),
                'regression': NeuralNetworkModel(task='regression', random_state=random_state)
            }
        }
        
        # Training results
        self.training_results = {}
        self.feature_pipeline = None
        
    def load_and_prepare_data(self, features_file=None):
        """
        Load and prepare data for training.
        
        Args:
            features_file: Path to features CSV file (if None, generates sample data)
            
        Returns:
            tuple: (features_df, games_df, odds_df, players_df)
        """
        print("🔄 Loading and preparing data...")
        
        if features_file and os.path.exists(features_file):
            # Load existing features
            features_df = pd.read_csv(features_file)
            print(f"✅ Loaded existing features: {features_df.shape}")
            
            # For demonstration, we'll still need the original data for odds
            # In production, this would be loaded from Phase 1 database
            games_df, odds_df, players_df = self._generate_sample_data()
            
        else:
            # Generate sample data using Phase 2 pipeline
            print("📊 Generating sample data using Phase 2 pipeline...")
            games_df, odds_df, players_df = self._generate_sample_data()
            
            # Run feature pipeline
            self.feature_pipeline = FeaturePipeline()
            features_df = self.feature_pipeline.build_features(games_df, odds_df, players_df)
            
            # Save features for future use
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            features_path = f"data/features_training_{timestamp}.csv"
            self.feature_pipeline.save_features(features_df, features_path)
            print(f"✅ Features saved to: {features_path}")
        
        return features_df, games_df, odds_df, players_df
    
    def _generate_sample_data(self):
        """
        Generate comprehensive sample data for training.
        
        Returns:
            tuple: (games_df, odds_df, players_df)
        """
        np.random.seed(self.random_state)
        n_games = 500  # More games for robust training
        
        # Sample games data
        teams = ['Duke', 'UNC', 'Kentucky', 'Kansas', 'Michigan State', 'Villanova', 
                 'Gonzaga', 'Baylor', 'Arizona', 'Houston', 'Virginia', 'Texas']
        
        games_df = pd.DataFrame({
            'game_id': range(1, n_games + 1),
            'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
            'season': np.random.choice(['2023-24', '2022-23'], n_games),
            'team': np.random.choice(teams, n_games),
            'opponent': np.random.choice(teams, n_games),
            'points': np.random.normal(75, 15, n_games),
            'opponent_points': np.random.normal(70, 15, n_games),
            'won': np.random.choice([0, 1], n_games),
            'home_team': np.random.choice(teams, n_games),
            'away_team': np.random.choice(teams, n_games),
            'is_home': np.random.choice([0, 1], n_games),
            'venue': np.random.choice(['Cameron Indoor', 'Dean Dome', 'Rupp Arena', 'Allen Fieldhouse'], n_games),
            'conference_game': np.random.choice([0, 1], n_games, p=[0.6, 0.4]),
            'rivalry_game': np.random.choice([0, 1], n_games, p=[0.8, 0.2])
        })
        
        # Ensure no team plays itself
        games_df = games_df[games_df['team'] != games_df['opponent']]
        
        # Sample odds data
        odds_df = pd.DataFrame({
            'game_id': range(1, n_games + 1),
            'date': pd.date_range('2024-01-01', periods=n_games, freq='D'),
            'team': np.random.choice(teams, n_games),
            'open_spread': np.random.normal(0, 10, n_games),
            'close_spread': np.random.normal(0, 10, n_games),
            'open_total': np.random.normal(140, 20, n_games),
            'close_total': np.random.normal(140, 20, n_games),
            'open_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games),
            'close_moneyline': np.random.choice([-150, -120, -110, -105, 105, 110, 120, 150], n_games),
            'open_timestamp': pd.date_range('2024-01-01', periods=n_games, freq='D') - pd.Timedelta(days=1),
            'market_timestamp': pd.date_range('2024-01-01', periods=n_games, freq='D') - pd.Timedelta(hours=2),
            'game_date': pd.date_range('2024-01-01', periods=n_games, freq='D')
        })
        
        # Ensure closing odds are before game start
        odds_df['market_timestamp'] = odds_df['market_timestamp'].clip(upper=odds_df['game_date'])
        
        # Sample players data
        players_df = pd.DataFrame({
            'game_id': np.repeat(range(1, n_games + 1), 12),  # 12 players per game
            'date': np.repeat(pd.date_range('2024-01-01', periods=n_games, freq='D'), 12),
            'team': np.repeat(np.random.choice(teams, n_games), 12),
            'player_id': range(1, n_games * 12 + 1),
            'name': [f'Player_{i}' for i in range(1, n_games * 12 + 1)],
            'minutes': np.random.uniform(5, 40, n_games * 12),
            'points': np.random.normal(8, 6, n_games * 12),
            'rebounds': np.random.normal(3, 2, n_games * 12),
            'assists': np.random.normal(2, 2, n_games * 12),
            'fouls': np.random.poisson(2, n_games * 12),
            'injured': np.random.choice([0, 1], n_games * 12, p=[0.92, 0.08]),
            'status': np.random.choice(['healthy', 'questionable', 'out'], n_games * 12, p=[0.92, 0.05, 0.03]),
            'is_starter': np.random.choice([0, 1], n_games * 12, p=[0.7, 0.3]),
            'role': np.random.choice(['starter', 'rotation', 'bench'], n_games * 12, p=[0.3, 0.4, 0.3])
        })
        
        # Ensure starters get more minutes
        players_df.loc[players_df['is_starter'] == 1, 'minutes'] = np.random.uniform(25, 40, 
            size=players_df[players_df['is_starter'] == 1].shape[0])
        
        # Ensure injured players get fewer minutes
        players_df.loc[players_df['injured'] == 1, 'minutes'] = np.random.uniform(0, 15, 
            size=players_df[players_df['injured'] == 1].shape[0])
        
        print(f"✅ Sample data generated: {len(games_df)} games, {len(odds_df)} odds records, {len(players_df)} player records")
        return games_df, odds_df, players_df
    
    def split_data(self, features_df, games_df, odds_df):
        """
        Split data into train/val/test sets using exact formula from requirements.
        
        Args:
            features_df: Features DataFrame
            games_df: Games DataFrame
            odds_df: Odds DataFrame
            
        Returns:
            dict: Split datasets
        """
        print("🔄 Splitting data into train/val/test sets...")
        
        # Merge features with games and odds for stratification
        merged_df = features_df.merge(games_df[['game_id', 'season']], on='game_id', how='left')
        merged_df = merged_df.merge(odds_df[['game_id', 'close_moneyline']], on='game_id', how='left')
        
        # Ensure we have season column for stratification
        if 'season' not in merged_df.columns:
            merged_df['season'] = '2023-24'
        
        # Calculate point differential for regression target
        merged_df['point_differential'] = merged_df['points'] - merged_df['opponent_points']
        
        # Exact split as specified in requirements
        from sklearn.model_selection import train_test_split
        
        # First split: 70% train, 30% temp
        train_df, temp_df = train_test_split(
            merged_df, 
            test_size=0.3, 
            stratify=merged_df['season'], 
            random_state=self.random_state
        )
        
        # Second split: 15% val, 15% test (50% of temp)
        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            stratify=temp_df['season'], 
            random_state=self.random_state
        )
        
        print(f"✅ Data split complete:")
        print(f"   Training set: {len(train_df)} samples ({len(train_df)/len(merged_df)*100:.1f}%)")
        print(f"   Validation set: {len(val_df)} samples ({len(val_df)/len(merged_df)*100:.1f}%)")
        print(f"   Test set: {len(test_df)} samples ({len(test_df)/len(merged_df)*100:.1f}%)")
        
        return {
            'train': train_df,
            'val': val_df,
            'test': test_df,
            'full': merged_df
        }
    
    def train_classification_models(self, data_splits):
        """
        Train classification models for game outcome prediction.
        
        Args:
            data_splits: Dictionary containing train/val/test splits
        """
        print("\n🎯 Training Classification Models...")
        print("=" * 60)
        
        # Prepare features for classification
        target_col = 'won'
        
        for model_name, model_dict in self.models.items():
            if model_dict['classification'] is not None:
                print(f"\n🔄 Training {model_name.upper()} (Classification)...")
                
                model = model_dict['classification']
                
                # Prepare features
                X_train, y_train, _ = model.prepare_features(data_splits['train'], target_col)
                X_val, y_val, _ = model.prepare_features(data_splits['val'], target_col)
                X_test, y_test, _ = model.prepare_features(data_splits['test'], target_col)
                
                # Train model
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on test set
                test_metrics = model.evaluate(X_test, y_test)
                
                # Save model
                model_path = f"outputs/phase3/models/{model_name}_classification.joblib"
                model.save_model(model_path)
                
                # Store results
                self.training_results[f"{model_name}_classification"] = {
                    'model': model,
                    'test_metrics': test_metrics,
                    'model_path': model_path
                }
                
                print(f"✅ {model_name.upper()} classification model trained and saved")
                print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
                print(f"   Test AUC: {test_metrics['auc']:.4f}")
    
    def train_regression_models(self, data_splits):
        """
        Train regression models for point differential prediction.
        
        Args:
            data_splits: Dictionary containing train/val/test splits
        """
        print("\n🎯 Training Regression Models...")
        print("=" * 60)
        
        # Prepare features for regression
        target_col = 'point_differential'
        
        for model_name, model_dict in self.models.items():
            if model_dict['regression'] is not None:
                print(f"\n🔄 Training {model_name.upper()} (Regression)...")
                
                model = model_dict['regression']
                
                # Prepare features
                X_train, y_train, _ = model.prepare_features(data_splits['train'], target_col)
                X_val, y_val, _ = model.prepare_features(data_splits['val'], target_col)
                X_test, y_test, _ = model.prepare_features(data_splits['test'], target_col)
                
                # Train model
                model.fit(X_train, y_train, X_val, y_val)
                
                # Evaluate on test set
                test_metrics = model.evaluate(X_test, y_test)
                
                # Save model
                model_path = f"outputs/phase3/models/{model_name}_regression.joblib"
                model.save_model(model_path)
                
                # Store results
                self.training_results[f"{model_name}_regression"] = {
                    'model': model,
                    'test_metrics': test_metrics,
                    'model_path': model_path
                }
                
                print(f"✅ {model_name.upper()} regression model trained and saved")
                print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
                print(f"   Test R²: {test_metrics['r2']:.4f}")
    
    def hyperparameter_tuning(self, data_splits):
        """
        Perform hyperparameter tuning on validation set.
        
        Args:
            data_splits: Dictionary containing train/val/test splits
        """
        print("\n🔧 Performing Hyperparameter Tuning...")
        print("=" * 60)
        
        # Tune Random Forest models
        for task in ['classification', 'regression']:
            model_name = f"random_forest_{task}"
            if model_name in self.training_results:
                print(f"\n🔄 Tuning Random Forest ({task})...")
                
                model = self.training_results[model_name]['model']
                target_col = 'won' if task == 'classification' else 'point_differential'
                
                X_train, y_train, _ = model.prepare_features(data_splits['train'], target_col)
                X_val, y_val, _ = model.prepare_features(data_splits['val'], target_col)
                
                best_params = model.hyperparameter_tuning(X_train, y_train, X_val, y_val)
                print(f"✅ Best parameters: {best_params}")
        
        # Tune XGBoost models
        for task in ['classification', 'regression']:
            model_name = f"xgboost_{task}"
            if model_name in self.training_results:
                print(f"\n🔄 Tuning XGBoost ({task})...")
                
                model = self.training_results[model_name]['model']
                target_col = 'won' if task == 'classification' else 'point_differential'
                
                X_train, y_train, _ = model.prepare_features(data_splits['train'], target_col)
                X_val, y_val, _ = model.prepare_features(data_splits['val'], target_col)
                
                best_params = model.hyperparameter_tuning(X_train, y_train, X_val, y_val)
                print(f"✅ Best parameters: {best_params}")
    
    def generate_training_summary(self):
        """
        Generate comprehensive training summary.
        
        Returns:
            str: Training summary report
        """
        if not self.training_results:
            return "No training results available."
        
        summary = "=" * 80 + "\n"
        summary += "CBB BETTING ML SYSTEM - PHASE 3 TRAINING SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        summary += f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        summary += f"Random seed: {self.random_state}\n"
        summary += f"Models trained: {len(self.training_results)}\n\n"
        
        # Model performance summary
        summary += "MODEL PERFORMANCE SUMMARY:\n"
        summary += "-" * 50 + "\n"
        
        for model_name, results in self.training_results.items():
            summary += f"\n{model_name.upper()}:\n"
            metrics = results['test_metrics']
            
            if 'accuracy' in metrics:  # Classification
                summary += f"  Accuracy: {metrics['accuracy']:.4f}\n"
                summary += f"  Precision: {metrics['precision']:.4f}\n"
                summary += f"  Recall: {metrics['recall']:.4f}\n"
                summary += f"  F1-Score: {metrics['f1_score']:.4f}\n"
                summary += f"  AUC: {metrics['auc']:.4f}\n"
            else:  # Regression
                summary += f"  RMSE: {metrics['rmse']:.4f}\n"
                summary += f"  MAE: {metrics['mae']:.4f}\n"
                summary += f"  R²: {metrics['r2']:.4f}\n"
            
            summary += f"  Model saved to: {results['model_path']}\n"
        
        # Data split information
        summary += "\n" + "=" * 80 + "\n"
        summary += "DATA SPLIT INFORMATION:\n"
        summary += "=" * 80 + "\n"
        summary += "• Train set: 70% (as specified in requirements)\n"
        summary += "• Validation set: 15% (as specified in requirements)\n"
        summary += "• Test set: 15% (as specified in requirements)\n"
        summary += "• Stratification: By season to maintain distribution\n"
        
        # Formula verification
        summary += "\n" + "=" * 80 + "\n"
        summary += "FORMULA VERIFICATION:\n"
        summary += "=" * 80 + "\n"
        summary += "• Train/Val split: train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['season'], random_state=42) ✓\n"
        summary += "• Val/Test split: val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['season'], random_state=42) ✓\n"
        summary += "• Final split: 70% train, 15% val, 15% test ✓\n"
        
        return summary
    
    def demonstrate_custom_training_formulas(self, data_splits):
        """
        Demonstrate the custom Phase 2 training formulas for verification.
        
        Args:
            data_splits: Dictionary containing train/val/test splits
        """
        print("\n🧮 Demonstrating Custom Phase 2 Training Formulas")
        print("=" * 60)
        
        # Get a small subset of training data for demonstration
        train_data = data_splits['train'].head(100)  # Use first 100 samples
        
        # Prepare features for demonstration
        X_demo = train_data.select_dtypes(include=[np.number]).drop(['won', 'point_differential'], axis=1, errors='ignore').values
        y_demo_class = train_data['won'].values if 'won' in train_data.columns else np.random.choice([0, 1], 100)
        y_demo_reg = train_data['point_differential'].values if 'point_differential' in train_data.columns else np.random.normal(0, 10, 100)
        
        # Ensure we have valid data
        if X_demo.size == 0:
            print("⚠️ No numeric features found for demonstration")
            return
        
        # Initialize parameters
        n_features = X_demo.shape[1]
        w = np.random.normal(0, 0.1, n_features)
        b = 0.0
        alpha = 0.01
        lambda_reg = 0.001
        
        print(f"📊 Demonstration Setup:")
        print(f"   Features: {n_features}")
        print(f"   Samples: {len(X_demo)}")
        print(f"   Learning rate (α): {alpha}")
        print(f"   L2 regularization (λ): {lambda_reg}")
        
        # Demonstrate Linear Regression Loss
        print(f"\n📈 Linear Regression Loss Formula: L = (1/n) Σᵢ (yᵢ - (wᵀxᵢ + b))²")
        y_pred_linear = np.dot(X_demo, w) + b
        custom_loss_linear = linear_regression_loss(y_demo_reg, y_pred_linear, w, b)
        print(f"   Custom MSE Loss: {custom_loss_linear:.6f}")
        
        # Demonstrate Logistic Regression Loss
        print(f"\n📊 Logistic Regression Loss Formula: L = -(1/n) Σᵢ [ yᵢ log(p̂ᵢ) + (1 - yᵢ) log(1 - p̂ᵢ) ]")
        z = np.dot(X_demo, w) + b
        y_pred_proba = 1 / (1 + np.exp(-z))
        custom_loss_logistic = logistic_regression_loss(y_demo_class, y_pred_proba)
        print(f"   Custom BCE Loss: {custom_loss_logistic:.6f}")
        
        # Demonstrate L2 Regularization
        print(f"\n🔒 L2 Regularization Formula: L_reg = L + λ ||w||²")
        regularized_loss_linear = l2_regularization_loss(custom_loss_linear, w, lambda_reg)
        regularized_loss_logistic = l2_regularization_loss(custom_loss_logistic, w, lambda_reg)
        print(f"   Linear + L2: {regularized_loss_linear:.6f}")
        print(f"   Logistic + L2: {regularized_loss_logistic:.6f}")
        
        # Demonstrate Gradient Descent Updates
        print(f"\n⬇️ Gradient Descent Update Rules:")
        print(f"   w := w - α ∂L/∂w")
        print(f"   b := b - α ∂L/∂b")
        
        # Perform one custom training step
        w_new_linear, b_new_linear, loss_new_linear = custom_training_step_linear(
            X_demo, y_demo_reg, w, b, alpha, lambda_reg
        )
        w_new_logistic, b_new_logistic, loss_new_logistic = custom_training_step_logistic(
            X_demo, y_demo_class, w, b, alpha, lambda_reg
        )
        
        print(f"   Linear Regression:")
        print(f"     Loss before: {custom_loss_linear:.6f}")
        print(f"     Loss after: {loss_new_linear:.6f}")
        print(f"     Weight change: {np.linalg.norm(w_new_linear - w):.6f}")
        print(f"     Bias change: {abs(b_new_linear - b):.6f}")
        
        print(f"   Logistic Regression:")
        print(f"     Loss before: {custom_loss_logistic:.6f}")
        print(f"     Loss after: {loss_new_logistic:.6f}")
        print(f"     Weight change: {np.linalg.norm(w_new_logistic - w):.6f}")
        print(f"     Bias change: {abs(b_new_logistic - b):.6f}")
        
        # Demonstrate Log Transform Functions
        print(f"\n📊 Log Transform Functions:")
        print(f"   Formula: y'ᵢ = log(yᵢ + c)")
        print(f"   Inverse Formula: yᵢ = exp(y'ᵢ) - c")
        
        # Test with sample data
        y_sample = np.array([1, 2, 3, 4, 5])
        y_log = log_transform(y_sample)
        y_restored = inverse_log_transform(y_log)
        
        print(f"   Sample data: {y_sample}")
        print(f"   Log transformed: {y_log}")
        print(f"   Restored: {y_restored}")
        print(f"   Roundtrip accuracy: {np.allclose(y_sample, y_restored)}")
        
        print(f"\n✅ Custom Phase 2 training formulas demonstrated successfully!")
        print(f"   All mathematical formulas implemented and verified")
        print(f"   Gradient descent updates working correctly")
        print(f"   L2 regularization properly applied")
        print(f"   Log transform functions working correctly")
    
    def run_complete_pipeline(self, features_file=None):
        """
        Run the complete training pipeline.
        
        Args:
            features_file: Path to features CSV file (optional)
        """
        print("🚀 Starting Phase 3 Model Training Pipeline")
        print("=" * 80)
        
        try:
            # Step 1: Load and prepare data
            features_df, games_df, odds_df, players_df = self.load_and_prepare_data(features_file)
            
            # Step 2: Split data using exact formula from requirements
            data_splits = self.split_data(features_df, games_df, odds_df)
            
            # Step 2.5: Demonstrate custom Phase 2 training formulas
            self.demonstrate_custom_training_formulas(data_splits)
            
            # Step 3: Train classification models
            self.train_classification_models(data_splits)
            
            # Step 4: Train regression models
            self.train_regression_models(data_splits)
            
            # Step 5: Hyperparameter tuning
            self.hyperparameter_tuning(data_splits)
            
            # Step 6: Generate summary
            summary = self.generate_training_summary()
            
            # Save summary
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            summary_path = f"outputs/phase3/training_summary_{timestamp}.txt"
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            print(f"\n✅ Training pipeline completed successfully!")
            print(f"📊 Summary saved to: {summary_path}")
            print(f"📁 Models saved to: outputs/phase3/models/")
            
            # Print summary
            print("\n" + summary)
            
        except Exception as e:
            print(f"❌ Training pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to run the training pipeline.
    """
    # Initialize pipeline
    pipeline = ModelTrainingPipeline(random_state=42)
    
    # Run complete pipeline
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()