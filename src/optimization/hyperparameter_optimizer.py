"""
Hyperparameter optimization implementation for Phase 4.
Implements grid search, random search, and Bayesian optimization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Callable, Tuple
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer
import optuna
from src.metrics import roc_auc_score, log_loss


class HyperparameterOptimizer:
    """
    Base class for hyperparameter optimization.
    """
    
    def __init__(self, model, param_grid, cv=5, scoring='roc_auc', random_state=42):
        """
        Initialize optimizer.
        
        Parameters:
        -----------
        model : estimator
            Base model to optimize
        param_grid : dict
            Parameter grid to search
        cv : int
            Cross-validation folds
        scoring : str
            Scoring metric
        random_state : int
            Random seed
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.best_params_ = None
        self.best_score_ = None
        self.best_estimator_ = None
        self.results_ = None
        
    def optimize(self, X, y):
        """
        Perform hyperparameter optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        self : Fitted optimizer
        """
        raise NotImplementedError("Subclasses must implement optimize()")
    
    def get_best_params(self):
        """Get best hyperparameters found."""
        return self.best_params_
    
    def get_best_score(self):
        """Get best score achieved."""
        return self.best_score_
    
    def get_best_estimator(self):
        """Get best fitted estimator."""
        return self.best_estimator_


class GridSearchOptimizer(HyperparameterOptimizer):
    """
    Grid search hyperparameter optimization.
    """
    
    def optimize(self, X, y):
        """
        Perform grid search optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        self : Fitted optimizer
        """
        # Create custom scorers
        scorers = {
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'log_loss': make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        }
        
        # Perform grid search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=scorers,
            refit='roc_auc',  # Refit on best ROC-AUC
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        # Store results
        self.best_params_ = grid_search.best_params_
        self.best_score_ = grid_search.best_score_
        self.best_estimator_ = grid_search.best_estimator_
        self.results_ = pd.DataFrame(grid_search.cv_results_)
        
        return self


class RandomSearchOptimizer(HyperparameterOptimizer):
    """
    Random search hyperparameter optimization.
    """
    
    def __init__(self, model, param_distributions, n_iter=100, **kwargs):
        """
        Initialize random search optimizer.
        
        Parameters:
        -----------
        model : estimator
            Base model to optimize
        param_distributions : dict
            Parameter distributions to sample from
        n_iter : int
            Number of iterations
        **kwargs : Additional arguments for base class
        """
        super().__init__(model, param_distributions, **kwargs)
        self.param_distributions = param_distributions
        self.n_iter = n_iter
    
    def optimize(self, X, y):
        """
        Perform random search optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        self : Fitted optimizer
        """
        # Create custom scorers
        scorers = {
            'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
            'log_loss': make_scorer(log_loss, needs_proba=True, greater_is_better=False)
        }
        
        # Perform random search
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=scorers,
            refit='roc_auc',  # Refit on best ROC-AUC
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        # Store results
        self.best_params_ = random_search.best_params_
        self.best_score_ = random_search.best_score_
        self.best_estimator_ = random_search.best_estimator_
        self.results_ = pd.DataFrame(random_search.cv_results_)
        
        return self


class BayesianOptimizer(HyperparameterOptimizer):
    """
    Bayesian optimization using Optuna.
    """
    
    def __init__(self, model, param_space, n_trials=100, **kwargs):
        """
        Initialize Bayesian optimizer.
        
        Parameters:
        -----------
        model : estimator
            Base model to optimize
        param_space : dict
            Parameter space definition
        n_trials : int
            Number of trials
        **kwargs : Additional arguments for base class
        """
        super().__init__(model, param_space, **kwargs)
        self.param_space = param_space
        self.n_trials = n_trials
        self.study = None
    
    def optimize(self, X, y):
        """
        Perform Bayesian optimization.
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Training labels
            
        Returns:
        --------
        self : Fitted optimizer
        """
        from sklearn.model_selection import cross_val_score
        
        def objective(trial):
            # Sample hyperparameters
            params = {}
            for param_name, param_config in self.param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'log':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=True)
            
            # Create model with sampled parameters
            model_instance = self.model.__class__(**params)
            
            # Perform cross-validation
            try:
                scores = cross_val_score(
                    model_instance, X, y, 
                    cv=self.cv, 
                    scoring='roc_auc',
                    n_jobs=-1
                )
                return scores.mean()
            except:
                return -1.0  # Return low score for invalid parameters
        
        # Create study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state)
        )
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
        
        # Store results
        self.best_params_ = self.study.best_params
        self.best_score_ = self.study.best_value
        
        # Fit best model
        best_model = self.model.__class__(**self.best_params_)
        self.best_estimator_ = best_model.fit(X, y)
        
        # Create results DataFrame
        self.results_ = pd.DataFrame([
            {
                'trial': trial.number,
                'value': trial.value,
                'params': trial.params
            }
            for trial in self.study.trials
        ])
        
        return self
    
    def plot_optimization_history(self, save_path=None):
        """
        Plot optimization history.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        if self.study is None:
            print("No study available. Run optimize() first.")
            return
        
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Optimization history
        optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
        ax1.set_title('Optimization History')
        
        # Parameter importance
        try:
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title('Parameter Importance')
        except:
            ax2.text(0.5, 0.5, 'Parameter importance not available', 
                    ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_optimizer(optimizer_type, model, param_space, **kwargs):
    """
    Factory function to create optimizer instances.
    
    Parameters:
    -----------
    optimizer_type : str
        'grid', 'random', or 'bayesian'
    model : estimator
        Base model to optimize
    param_space : dict
        Parameter space definition
    **kwargs : Additional arguments
        
    Returns:
    --------
    HyperparameterOptimizer : Optimizer instance
    """
    if optimizer_type == 'grid':
        return GridSearchOptimizer(model, param_space, **kwargs)
    elif optimizer_type == 'random':
        return RandomSearchOptimizer(model, param_space, **kwargs)
    elif optimizer_type == 'bayesian':
        return BayesianOptimizer(model, param_space, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")