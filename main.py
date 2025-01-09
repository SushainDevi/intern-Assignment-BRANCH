from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

app = FastAPI(title="Loan Default Prediction API")

# Load the saved model pipeline
try:
    with open('loan_default_model.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
except Exception as e:
    raise Exception(f"Failed to load model: {str(e)}")

class UserData(BaseModel):
    user_id: int
    cash_incoming_30days: float
    gps_fix_count_scaled: float | None = None
    movement_radius_approx_scaled: float | None = None
    age: int
    var_longitude: float

class PredictionResponse(BaseModel):
    user_id: int
    default_probability: float
    prediction: bool
    confidence_score: float

@app.post("/predict", response_model=PredictionResponse)
async def predict_loan_default(user_data: UserData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([user_data.dict()])
        
        # Separate features
        gps_features = input_data[model_pipeline['gps_features']]
        other_features = input_data[model_pipeline['other_features']]
        
        # Apply preprocessing
        gps_imputed = model_pipeline['gps_imputer'].transform(gps_features)
        other_imputed = model_pipeline['numeric_imputer'].transform(other_features)
        
        # Combine features
        X = np.hstack([other_imputed, gps_imputed])
        
        # Scale features
        X_scaled = model_pipeline['scaler'].transform(X)
        
        # Make prediction
        probability = model_pipeline['model'].predict_proba(X_scaled)[0][1]
        prediction = probability >= 0.5
        
        # Calculate confidence score (distance from decision boundary)
        confidence_score = abs(probability - 0.5) * 2  # Scale to 0-1
        
        return PredictionResponse(
            user_id=user_data.user_id,
            default_probability=float(probability),
            prediction=bool(prediction),
            confidence_score=float(confidence_score)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-info")
async def get_model_info():
    return {
        "model_type": "Random Forest Classifier",
        "features": model_pipeline['features'],
        "parameters": model_pipeline['model'].get_params(),
        "feature_importance": dict(zip(
            model_pipeline['features'],
            model_pipeline['model'].feature_importances_
        ))
    }

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}