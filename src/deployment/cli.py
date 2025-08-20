"""
CLI tools for Phase 4 deployment.
Provides command-line interface for batch predictions.
"""

import pandas as pd
import numpy as np
import argparse
import joblib
import os
import sys
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cli_predict(input_file: str, output_file: str, model_name: str = None):
    """
    CLI function for batch predictions.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file with features
    output_file : str
        Path to output CSV file for predictions
    model_name : str, optional
        Specific model to use (default: ensemble)
    """
    try:
        # Load input data
        logger.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        if df.empty:
            logger.error("Input file is empty")
            return
        
        # Load models
        model_path = "outputs/phase3/models"
        if not os.path.exists(model_path):
            logger.error(f"Model directory not found: {model_path}")
            return
        
        models = {}
        model_files = {
            'logistic_regression': 'logistic_regression_model.joblib',
            'random_forest': 'random_forest_model.joblib',
            'xgboost': 'xgboost_model.joblib',
            'neural_network': 'neural_network_model.joblib'
        }
        
        for name, filename in model_files.items():
            filepath = os.path.join(model_path, filename)
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
                logger.info(f"Loaded {name} model")
        
        if not models:
            logger.error("No models found")
            return
        
        # Get feature columns (exclude non-feature columns)
        exclude_cols = ['game_id', 'date', 'season', 'target', 'home_team', 'away_team']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if not feature_cols:
            logger.error("No feature columns found")
            return
        
        logger.info(f"Using {len(feature_cols)} feature columns: {feature_cols[:5]}...")
        
        # Make predictions
        X = df[feature_cols].values
        
        if model_name and model_name in models:
            # Use specific model
            model = models[model_name]
            logger.info(f"Using {model_name} model")
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = model.predict(X)
            
            df["probability"] = probabilities
            df["model_used"] = model_name
            
        else:
            # Use ensemble
            logger.info("Using ensemble prediction")
            all_predictions = []
            
            for name, model in models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X)[:, 1]
                    else:
                        pred = model.predict(X)
                    all_predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Error with {name} model: {e}")
                    continue
            
            if all_predictions:
                # Average predictions
                ensemble_prob = np.mean(all_predictions, axis=0)
                df["probability"] = ensemble_prob
                df["model_used"] = "ensemble"
            else:
                logger.error("No models could make predictions")
                return
        
        # Calculate additional metrics
        df["odds"] = 1 / df["probability"].replace(0, np.inf)
        df["confidence"] = df["probability"].apply(
            lambda x: "high" if x > 0.7 else "medium" if x > 0.6 else "low"
        )
        
        # Save results
        logger.info(f"Saving predictions to {output_file}")
        df.to_csv(output_file, index=False)
        
        # Print summary
        logger.info(f"Predictions completed successfully!")
        logger.info(f"Input samples: {len(df)}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Probability range: {df['probability'].min():.3f} - {df['probability'].max():.3f}")
        logger.info(f"Average probability: {df['probability'].mean():.3f}")
        
    except Exception as e:
        logger.error(f"Error in cli_predict: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CBB Betting ML System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.deployment.cli input.csv output.csv
  python -m src.deployment.cli input.csv output.csv --model random_forest
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input CSV file with features"
    )
    
    parser.add_argument(
        "output_file", 
        help="Output CSV file for predictions"
    )
    
    parser.add_argument(
        "--model", "-m",
        choices=['logistic_regression', 'random_forest', 'xgboost', 'neural_network'],
        help="Specific model to use (default: ensemble)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run prediction
    cli_predict(args.input_file, args.output_file, args.model)


if __name__ == "__main__":
    main()