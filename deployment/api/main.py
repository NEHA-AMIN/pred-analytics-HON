"""
FastAPI Model Serving Application

Production-ready API for serving purchase prediction models.

Features:
- Real-time predictions with sub-100ms latency
- Batch prediction support
- Model versioning and hot-swapping
- Health checks and monitoring
- Request validation with Pydantic
- Comprehensive error handling
- OpenAPI documentation
"""

import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="E-Commerce Purchase Prediction API",
    description="Production ML API for predicting user purchase intent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class UserItemFeatures(BaseModel):
    """Features for a single (user, item) prediction."""
    
    # User ID and Item ID
    user_id: str = Field(..., description="Unique user identifier")
    item_id: str = Field(..., description="Unique item identifier")
    
    # Recency features
    days_since_last_view: Optional[float] = Field(None, ge=0, description="Days since user last viewed any item")
    days_since_last_item_view: Optional[float] = Field(None, ge=0, description="Days since user last viewed this item")
    days_since_last_cart: Optional[float] = Field(None, ge=0, description="Days since user last added to cart")
    
    # Frequency features
    view_count_7d: int = Field(0, ge=0, description="Number of views in last 7 days")
    view_count_30d: int = Field(0, ge=0, description="Number of views in last 30 days")
    cart_count_7d: int = Field(0, ge=0, description="Number of cart additions in last 7 days")
    cart_count_30d: int = Field(0, ge=0, description="Number of cart additions in last 30 days")
    purchase_count_7d: int = Field(0, ge=0, description="Number of purchases in last 7 days")
    purchase_count_30d: int = Field(0, ge=0, description="Number of purchases in last 30 days")
    
    # Intent features
    cart_to_purchase_ratio: float = Field(0.0, ge=0, le=1, description="Historical cart to purchase conversion rate")
    avg_session_duration: float = Field(0.0, ge=0, description="Average session duration in minutes")
    
    # Session context features
    session_item_count: int = Field(1, ge=1, description="Number of items in current session")
    session_duration_minutes: float = Field(0.0, ge=0, description="Current session duration in minutes")
    
    # User-item affinity features
    item_view_count: int = Field(0, ge=0, description="Number of times user viewed this item")
    item_cart_count: int = Field(0, ge=0, description="Number of times user added this item to cart")
    
    # Item features
    item_price: float = Field(..., gt=0, description="Current item price")
    item_base_price: float = Field(..., gt=0, description="Original item price")
    item_discount_pct: float = Field(0.0, ge=0, le=100, description="Discount percentage")
    item_category: str = Field(..., description="Item category")
    
    @validator('item_price', 'item_base_price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "user_000123",
                "item_id": "item_00456",
                "days_since_last_view": 2.5,
                "days_since_last_item_view": 5.0,
                "days_since_last_cart": 10.0,
                "view_count_7d": 15,
                "view_count_30d": 45,
                "cart_count_7d": 2,
                "cart_count_30d": 5,
                "purchase_count_7d": 0,
                "purchase_count_30d": 1,
                "cart_to_purchase_ratio": 0.2,
                "avg_session_duration": 12.5,
                "session_item_count": 8,
                "session_duration_minutes": 15.0,
                "item_view_count": 3,
                "item_cart_count": 1,
                "item_price": 89.99,
                "item_base_price": 129.99,
                "item_discount_pct": 30.0,
                "item_category": "Electronics"
            }
        }


class PredictionRequest(BaseModel):
    """Request for single or batch predictions."""
    
    features: List[UserItemFeatures] = Field(..., min_items=1, max_items=1000, description="List of feature sets")
    return_probabilities: bool = Field(True, description="Whether to return probability scores")
    threshold: float = Field(0.5, ge=0, le=1, description="Classification threshold")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    {
                        "user_id": "user_000123",
                        "item_id": "item_00456",
                        "days_since_last_view": 2.5,
                        "view_count_7d": 15,
                        "item_price": 89.99,
                        "item_base_price": 129.99,
                        "item_discount_pct": 30.0,
                        "item_category": "Electronics"
                    }
                ],
                "return_probabilities": True,
                "threshold": 0.5
            }
        }


class PredictionResult(BaseModel):
    """Result for a single prediction."""
    
    user_id: str
    item_id: str
    prediction: int = Field(..., ge=0, le=1, description="Binary prediction (0=no purchase, 1=purchase)")
    probability: Optional[float] = Field(None, ge=0, le=1, description="Purchase probability")
    confidence: str = Field(..., description="Confidence level (low/medium/high)")


class PredictionResponse(BaseModel):
    """Response containing predictions and metadata."""
    
    predictions: List[PredictionResult]
    model_version: str
    timestamp: str
    latency_ms: float
    
    class Config:
        schema_extra = {
            "example": {
                "predictions": [
                    {
                        "user_id": "user_000123",
                        "item_id": "item_00456",
                        "prediction": 1,
                        "probability": 0.78,
                        "confidence": "high"
                    }
                ],
                "model_version": "xgboost_20240101_120000",
                "timestamp": "2024-01-01T12:00:00",
                "latency_ms": 45.2
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool
    model_version: Optional[str]
    uptime_seconds: float
    timestamp: str


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """
    Manages model loading, caching, and inference.
    
    Features:
    - Lazy loading
    - Model versioning
    - Hot-swapping support
    - Feature preprocessing
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize model manager.
        
        Args:
            model_path: Path to pickled model file
        """
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_version = None
        self.load_time = None
        
        if model_path and model_path.exists():
            self.load_model(model_path)
    
    def load_model(self, model_path: Path):
        """
        Load model from disk.
        
        Args:
            model_path: Path to pickled model file
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Handle different pickle formats
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.preprocessor = model_data.get('preprocessor')
                self.feature_names = model_data.get('feature_names')
            else:
                self.model = model_data
            
            self.model_path = model_path
            self.model_version = model_path.stem
            self.load_time = datetime.now()
            
            logger.info(f"✅ Model loaded successfully: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def prepare_features(self, features: List[UserItemFeatures]) -> pd.DataFrame:
        """
        Convert Pydantic models to DataFrame with correct feature order.
        
        Args:
            features: List of feature objects
            
        Returns:
            DataFrame ready for prediction
        """
        # Convert to list of dicts
        feature_dicts = [f.dict() for f in features]
        df = pd.DataFrame(feature_dicts)
        
        # Remove ID columns (not used for prediction)
        id_columns = ['user_id', 'item_id']
        ids_df = df[id_columns].copy()
        
        # One-hot encode categorical features
        if 'item_category' in df.columns:
            df = pd.get_dummies(df, columns=['item_category'], prefix='category')
        
        # Remove ID columns from features
        feature_df = df.drop(columns=id_columns, errors='ignore')
        
        # If we have stored feature names, ensure correct order
        if self.feature_names is not None:
            # Add missing columns with 0
            for col in self.feature_names:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Select and order columns
            feature_df = feature_df[self.feature_names]
        
        # Apply preprocessor if available
        if self.preprocessor is not None:
            feature_array = self.preprocessor.transform(feature_df)
            feature_df = pd.DataFrame(
                feature_array,
                columns=feature_df.columns,
                index=feature_df.index
            )
        
        return feature_df, ids_df
    
    def predict(
        self,
        features: List[UserItemFeatures],
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Make predictions for given features.
        
        Args:
            features: List of feature objects
            return_probabilities: Whether to return probability scores
            
        Returns:
            List of prediction dictionaries
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Prepare features
        feature_df, ids_df = self.prepare_features(features)
        
        # Make predictions
        if return_probabilities:
            # Get probability of positive class
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(feature_df)[:, 1]
            else:
                # For XGBoost DMatrix
                import xgboost as xgb
                dmatrix = xgb.DMatrix(feature_df)
                probabilities = self.model.predict(dmatrix)
        else:
            probabilities = None
        
        # Get binary predictions (using default threshold of 0.5)
        if probabilities is not None:
            predictions = (probabilities >= 0.5).astype(int)
        else:
            predictions = self.model.predict(feature_df)
        
        # Combine results
        results = []
        for idx in range(len(features)):
            result = {
                'user_id': ids_df.iloc[idx]['user_id'],
                'item_id': ids_df.iloc[idx]['item_id'],
                'prediction': int(predictions[idx])
            }
            
            if probabilities is not None:
                prob = float(probabilities[idx])
                result['probability'] = prob
                
                # Determine confidence level
                if prob < 0.3 or prob > 0.7:
                    confidence = "high"
                elif prob < 0.4 or prob > 0.6:
                    confidence = "medium"
                else:
                    confidence = "low"
                result['confidence'] = confidence
            
            results.append(result)
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None


# ============================================================================
# Global Model Manager Instance
# ============================================================================

# Initialize with default model path (will be loaded on first request if exists)
DEFAULT_MODEL_PATH = Path("models")
model_manager = ModelManager()

# Try to load the latest model
if DEFAULT_MODEL_PATH.exists():
    model_files = sorted(DEFAULT_MODEL_PATH.glob("*.pkl"))
    if model_files:
        latest_model = model_files[-1]
        try:
            model_manager.load_model(latest_model)
        except Exception as e:
            logger.warning(f"Could not load default model: {str(e)}")


# ============================================================================
# API Startup/Shutdown Events
# ============================================================================

start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("=" * 80)
    logger.info("🚀 Starting E-Commerce Purchase Prediction API")
    logger.info("=" * 80)
    logger.info(f"Model loaded: {model_manager.is_loaded()}")
    if model_manager.is_loaded():
        logger.info(f"Model version: {model_manager.model_version}")
    logger.info("API ready to serve requests")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down API")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "E-Commerce Purchase Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint for monitoring and load balancers.
    
    Returns:
        Health status including model state and uptime
    """
    uptime = time.time() - start_time
    
    return HealthResponse(
        status="healthy" if model_manager.is_loaded() else "degraded",
        model_loaded=model_manager.is_loaded(),
        model_version=model_manager.model_version,
        uptime_seconds=uptime,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make purchase predictions for user-item pairs.
    
    Args:
        request: Prediction request with features
        
    Returns:
        Predictions with probabilities and metadata
        
    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    start = time.time()
    
    # Check if model is loaded
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please load a model first."
        )
    
    try:
        # Make predictions
        predictions = model_manager.predict(
            features=request.features,
            return_probabilities=request.return_probabilities
        )
        
        # Apply custom threshold if different from 0.5
        if request.threshold != 0.5 and request.return_probabilities:
            for pred in predictions:
                if 'probability' in pred:
                    pred['prediction'] = int(pred['probability'] >= request.threshold)
        
        # Calculate latency
        latency_ms = (time.time() - start) * 1000
        
        # Build response
        response = PredictionResponse(
            predictions=[PredictionResult(**p) for p in predictions],
            model_version=model_manager.model_version,
            timestamp=datetime.now().isoformat(),
            latency_ms=round(latency_ms, 2)
        )
        
        logger.info(f"Prediction completed: {len(predictions)} items in {latency_ms:.2f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/load-model", tags=["Model Management"])
async def load_model(model_path: str):
    """
    Load a specific model version.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Success message with model version
    """
    try:
        path = Path(model_path)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model file not found: {model_path}"
            )
        
        model_manager.load_model(path)
        
        return {
            "status": "success",
            "message": f"Model loaded successfully",
            "model_version": model_manager.model_version,
            "loaded_at": model_manager.load_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )


@app.get("/model-info", tags=["Model Management"])
async def model_info():
    """
    Get information about the currently loaded model.
    
    Returns:
        Model metadata and statistics
    """
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model loaded"
        )
    
    return {
        "model_version": model_manager.model_version,
        "model_path": str(model_manager.model_path),
        "loaded_at": model_manager.load_time.isoformat(),
        "model_type": type(model_manager.model).__name__,
        "has_preprocessor": model_manager.preprocessor is not None,
        "num_features": len(model_manager.feature_names) if model_manager.feature_names else None
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
