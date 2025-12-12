# src/inference.py

import joblib
import pandas as pd
from pathlib import Path
from src.config import MODEL_PATH
from typing import Dict, Union

MODEL_PIPELINE = None


def load_model_pipeline(model_path: Path = MODEL_PATH):
    """Loads the trained model pipeline from the specified path."""
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
        try:
            MODEL_PIPELINE = joblib.load(model_path)
            print(f"✅ Model pipeline loaded successfully from {model_path}.")
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")

    return MODEL_PIPELINE


def predict_heart_disease(input_data: Dict[str, Union[int, float]]):
    """Makes a prediction using the loaded model pipeline."""
    pipeline = load_model_pipeline()

    # Convert input dictionary to a Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # The pipeline handles all preprocessing
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability": round(probability, 4)
    }