# api/main.py

from fastapi import FastAPI, HTTPException
import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root and src to PYTHONPATH for imports to work in local and container env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.inference import predict_heart_disease, load_model_pipeline
from api.schema import HeartDiseaseFeatures, PredictionResponse

# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API", version="1.0")

# Load model pipeline on startup
try:
    load_model_pipeline()
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load model during startup: {e}")


@app.get("/health", response_model=dict, tags=["Monitoring"])
def get_health():
    """Health check endpoint to ensure API is running and model is loaded."""
    model_loaded = True
    try:
        load_model_pipeline()
    except:
        model_loaded = False

    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "api_version": app.version
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_risk(features: HeartDiseaseFeatures):
    """Predicts the presence of heart disease (1) or absence (0)."""
    try:
        input_data = features.model_dump()
        result = predict_heart_disease(input_data)

        # Logging (Step 8): Structured log of request/response
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "endpoint": "/predict",
            "input": input_data,
            "output": result
        }
        print(json.dumps(log_entry))  # Prints to stdout, collected by Kubernetes/Cloud logs

        message = "Heart disease predicted" if result['prediction'] == 1 else "No heart disease predicted"

        return PredictionResponse(
            prediction=result['prediction'],
            probability=result['probability'],
            message=message
        )

    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Model artifact not found. The service is not ready.")
    except Exception as e:
        print(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Local Run Command: uvicorn api.main:app --reload