# src/config.py

from pathlib import Path

# --- General Configuration ---
PROJECT_DIR = Path(__file__).resolve().parents[1]

# --- Data Configuration ---
DATA_RAW_DIR = PROJECT_DIR / 'data' / 'raw'
DATA_PROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'
DATA_FILENAME = 'heart.csv'
# UPDATED URL: Using the raw processed.cleveland.data file from the official UCI repository.
DATA_DOWNLOAD_URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
                     "/processed.cleveland.data")

RAW_DATA_PATH = DATA_RAW_DIR / DATA_FILENAME
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / DATA_FILENAME

# Feature Definitions (used for headers since the raw file lacks them)
COLUMN_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal', 'target'
]

NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
TARGET_COLUMN = 'target'

# --- Model Configuration ---
MODEL_DIR = PROJECT_DIR / 'src' / 'model'
MODEL_FILENAME = 'final_model.pkl'
MODEL_PATH = MODEL_DIR / MODEL_FILENAME

TEST_SIZE = 0.2
RANDOM_STATE = 42
BEST_MODEL_NAME = "Random Forest"

# Define models and their hyperparameters
MODEL_CONFIGS = {
    "Logistic Regression": {
        "model": "LogisticRegression",
        "params": {"solver": 'liblinear', "random_state": RANDOM_STATE}
    },
    "Random Forest": {
        "model": "RandomForestClassifier",
        "params": {"n_estimators": 100, "max_depth": 5, "random_state": RANDOM_STATE}
    }
}

# --- MLOps Configuration ---
MLFLOW_EXPERIMENT_NAME = "Heart_Disease_Prediction_MLOps"
