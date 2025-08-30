#!/usr/bin/env python3
"""
College Predictor FastAPI Backend

REST API server that serves college predictions based on ML model.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import json
import os
import logging
from train_model import CollegePredictorModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="College Predictor API",
    description="API for predicting college admissions based on CET/JEE percentiles",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
predictor_model: Optional[CollegePredictorModel] = None

class StudentInput(BaseModel):
    """Student input model for predictions"""
    percentile: float
    category: str
    preferred_branch: str
    exam_type: str
    max_results: Optional[int] = 20
    
    @validator('percentile')
    def validate_percentile(cls, v):
        if not 0 <= v <= 100:
            raise ValueError('Percentile must be between 0 and 100')
        return v
    
    @validator('category')
    def validate_category(cls, v):
        allowed_categories = ['General', 'OBC', 'SC', 'ST', 'EWS']
        if v not in allowed_categories:
            raise ValueError(f'Category must be one of: {allowed_categories}')
        return v
    
    @validator('exam_type')
    def validate_exam_type(cls, v):
        allowed_exams = ['CET', 'JEE']
        if v not in allowed_exams:
            raise ValueError(f'Exam type must be one of: {allowed_exams}')
        return v

class CollegePrediction(BaseModel):
    """College prediction response model"""
    college_name: str
    branch: str
    state: str
    tier: int
    fees_per_year: int
    placement_percentage: int
    admission_probability: float
    cutoff_percentile: float

class PredictionResponse(BaseModel):
    """Response model for college predictions"""
    predictions: List[CollegePrediction]
    student_info: Dict[str, Any]
    total_colleges: int

class AvailableOptions(BaseModel):
    """Available options for dropdowns"""
    branches: List[str]
    categories: List[str]
    exam_types: List[str]
    states: List[str]

def load_model():
    """Load the trained model and preprocessors"""
    global predictor_model
    
    try:
        predictor_model = CollegePredictorModel()
        model_dir = 'models'
        
        if not os.path.exists(model_dir):
            logger.warning("Model directory not found. Training new model...")
            # Train a new model if none exists
            predictor_model.train('data/colleges_data.csv')
            predictor_model.save_model(model_dir)
        else:
            predictor_model.load_model(model_dir)
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "College Predictor API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": predictor_model is not None
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if predictor_model is not None else "unhealthy",
        "model_loaded": predictor_model is not None,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.get("/options", response_model=AvailableOptions)
async def get_available_options():
    """Get available options for form dropdowns"""
    try:
        # Load college data to get available options
        colleges_df = pd.read_csv('data/colleges_data.csv')
        
        return AvailableOptions(
            branches=sorted(colleges_df['branch'].unique().tolist()),
            categories=['General', 'OBC', 'SC', 'ST', 'EWS'],
            exam_types=sorted(colleges_df['exam_type'].unique().tolist()),
            states=sorted(colleges_df['state'].unique().tolist())
        )
    except Exception as e:
        logger.error(f"Error getting options: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load available options")

@app.post("/predict", response_model=PredictionResponse)
async def predict_colleges(student_input: StudentInput):
    """Predict top colleges for a student"""
    
    if predictor_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Predicting for student: {student_input.dict()}")
        
        # Get predictions from model
        predictions = predictor_model.predict_colleges(
            student_percentile=student_input.percentile,
            category=student_input.category,
            preferred_branch=student_input.preferred_branch,
            exam_type=student_input.exam_type
        )
        
        # Limit results
        limited_predictions = predictions[:student_input.max_results]
        
        # Convert to response format
        college_predictions = [
            CollegePrediction(**pred) for pred in limited_predictions
        ]
        
        return PredictionResponse(
            predictions=college_predictions,
            student_info={
                "percentile": student_input.percentile,
                "category": student_input.category,
                "preferred_branch": student_input.preferred_branch,
                "exam_type": student_input.exam_type
            },
            total_colleges=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/colleges")
async def get_all_colleges():
    """Get all available colleges"""
    try:
        colleges_df = pd.read_csv('data/colleges_data.csv')
        return {
            "colleges": colleges_df.to_dict('records'),
            "total_count": len(colleges_df)
        }
    except Exception as e:
        logger.error(f"Error loading colleges: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to load colleges data")

@app.get("/colleges/{exam_type}")
async def get_colleges_by_exam(exam_type: str):
    """Get colleges filtered by exam type"""
    try:
        colleges_df = pd.read_csv('data/colleges_data.csv')
        filtered_colleges = colleges_df[colleges_df['exam_type'] == exam_type.upper()]
        
        if filtered_colleges.empty:
            raise HTTPException(status_code=404, detail=f"No colleges found for exam type: {exam_type}")
        
        return {
            "colleges": filtered_colleges.to_dict('records'),
            "exam_type": exam_type.upper(),
            "total_count": len(filtered_colleges)
        }
    except Exception as e:
        logger.error(f"Error filtering colleges: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to filter colleges")

@app.post("/retrain")
async def retrain_model():
    """Retrain the model (admin endpoint)"""
    global predictor_model
    
    try:
        logger.info("Retraining model...")
        predictor_model = CollegePredictorModel()
        training_results = predictor_model.train('data/colleges_data.csv')
        predictor_model.save_model('models')
        
        return {
            "message": "Model retrained successfully",
            "training_results": training_results
        }
    except Exception as e:
        logger.error(f"Retraining error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
