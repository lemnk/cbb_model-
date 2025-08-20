"""
FastAPI deployment app for Phase 4.
Provides REST API for model predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CBB Betting ML System API",
    description="API for College Basketball Betting Predictions",
    version="1.0.0"
)

# Load trained models
MODEL_PATH = "outputs/phase3/models"
models = {}

class PredictionRequest(BaseModel):
    """Request model for predictions."""
    features: Dict[str, float]
    
class PredictionResponse(BaseModel):
    """Response model for predictions."""
    probability: float
    odds: float
    confidence: str
    model_used: str

@app.on_event("startup")
async def load_models():
    """Load trained models on startup."""
    try:
        # Load models from disk
        model_files = {
            'logistic_regression': 'logistic_regression_model.joblib',
            'random_forest': 'random_forest_model.joblib',
            'xgboost': 'xgboost_model.joblib',
            'neural_network': 'neural_network_model.joblib'
        }
        
        for model_name, filename in model_files.items():
            filepath = os.path.join(MODEL_PATH, filename)
            if os.path.exists(filepath):
                models[model_name] = joblib.load(filepath)
                logger.info(f"Loaded {model_name} model from {filepath}")
            else:
                logger.warning(f"Model file not found: {filepath}")
        
        if not models:
            logger.warning("No models loaded. API will return errors.")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CBB Betting ML System API",
        "version": "1.0.0",
        "models_loaded": list(models.keys())
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "available_models": list(models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make prediction using loaded models.
    
    Parameters:
    -----------
    request : PredictionRequest
        Features for prediction
        
    Returns:
    --------
    PredictionResponse : Prediction results
    """
    if not models:
        raise HTTPException(
            status_code=500, 
            detail="No models loaded. Please ensure models are available."
        )
    
    try:
        # Convert features to numpy array
        feature_values = list(request.features.values())
        X = np.array(feature_values).reshape(1, -1)
        
        # Use ensemble prediction (average of all models)
        predictions = []
        model_names = []
        
        for model_name, model in models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(X)[0, 1]
                else:
                    prob = model.predict(X)[0]
                predictions.append(prob)
                model_names.append(model_name)
            except Exception as e:
                logger.warning(f"Error with {model_name}: {e}")
                continue
        
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="All models failed to make predictions"
            )
        
        # Ensemble prediction (simple average)
        ensemble_prob = np.mean(predictions)
        
        # Calculate odds
        odds = 1 / ensemble_prob if ensemble_prob > 0 else float("inf")
        
        # Determine confidence level
        if ensemble_prob > 0.7:
            confidence = "high"
        elif ensemble_prob > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            probability=float(ensemble_prob),
            odds=float(odds),
            confidence=confidence,
            model_used="ensemble"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/{model_name}")
async def predict_with_model(model_name: str, request: PredictionRequest):
    """
    Make prediction using a specific model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to use
    request : PredictionRequest
        Features for prediction
        
    Returns:
    --------
    PredictionResponse : Prediction results
    """
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
        )
    
    try:
        # Convert features to numpy array
        feature_values = list(request.features.values())
        X = np.array(feature_values).reshape(1, -1)
        
        # Get prediction from specific model
        model = models[model_name]
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X)[0, 1]
        else:
            prob = model.predict(X)[0]
        
        # Calculate odds
        odds = 1 / prob if prob > 0 else float("inf")
        
        # Determine confidence level
        if prob > 0.7:
            confidence = "high"
        elif prob > 0.6:
            confidence = "medium"
        else:
            confidence = "low"
        
        return PredictionResponse(
            probability=float(prob),
            odds=float(odds),
            confidence=confidence,
            model_used=model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": list(models.keys()),
        "total_models": len(models)
    }

@app.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """Get information about a specific model."""
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )
    
    model = models[model_name]
    info = {
        "name": model_name,
        "type": type(model).__name__,
        "has_predict_proba": hasattr(model, 'predict_proba'),
        "has_feature_importances": hasattr(model, 'feature_importances_'),
        "has_coef": hasattr(model, 'coef_')
    }
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)