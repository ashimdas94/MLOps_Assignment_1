# api/schema.py

from pydantic import BaseModel, Field


# Input Schema for the /predict endpoint
class HeartDiseaseFeatures(BaseModel):
    age: int = Field(..., description="Age in years (e.g., 63)")
    sex: int = Field(..., description="Sex (1 = male; 0 = female)")
    cp: int = Field(..., description="Chest pain type (1-4, where 4 is asymptomatic)")
    trestbps: int = Field(..., description="Resting blood pressure (mm Hg)")
    chol: int = Field(..., description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., description="Fasting blood sugar > 120 mg/dl (1=true; 0=false)")
    restecg: int = Field(..., description="Resting electrocardiographic results (0-2)")
    thalach: int = Field(..., description="Maximum heart rate achieved")
    exang: int = Field(..., description="Exercise induced angina (1=yes; 0=no)")
    oldpeak: float = Field(..., description="ST depression induced by exercise relative to rest")
    slope: int = Field(..., description="The slope of the peak exercise ST segment (1-3)")
    ca: int = Field(..., description="Number of major vessels (0-3) colored by fluoroscopy")
    thal: int = Field(..., description="Thalium stress test result (3=normal; 6=fixed defect; 7=reversible defect)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 58, "sex": 1, "cp": 3, "trestbps": 150, "chol": 270,
                    "fbs": 0, "restecg": 0, "thalach": 145, "exang": 0,
                    "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 3
                }
            ]
        }
    }


# Output Schema for the /predict endpoint
class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="The predicted class (1 for heart disease, 0 for no heart disease)")
    probability: float = Field(..., description="The probability of heart disease presence (class 1)")
    message: str