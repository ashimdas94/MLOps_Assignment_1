# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from src.config import (
    RAW_DATA_PATH, MODEL_PATH, MODEL_DIR, MODEL_CONFIGS,
    BEST_MODEL_NAME, MLFLOW_EXPERIMENT_NAME
)
from src.utils import get_metrics, create_dirs
from src.preprocess import preprocess_and_split

# Map model names (strings) to actual classes
MODEL_CLASS_MAP = {
    "LogisticRegression": LogisticRegression,
    "RandomForestClassifier": RandomForestClassifier
}


def train_and_log_model(model_name: str, model_config: dict, preprocessor,
                        X_train, X_test, y_train, y_test):
    """Trains a model, tracks parameters/metrics with MLflow, and returns the trained pipeline."""

    model_class = MODEL_CLASS_MAP.get(model_config['model'])
    model = model_class(**model_config['params'])

    with mlflow.start_run(run_name=model_name):
        print(f"\n--- Starting MLflow Run for {model_name} ---")

        # 1. Create the full ML pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # 2. Train the model
        full_pipeline.fit(X_train, y_train)

        # 3. Predict and evaluate
        y_pred = full_pipeline.predict(X_test)
        y_prob = full_pipeline.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, y_pred, y_prob)

        # 4. Log Parameters and Metrics
        mlflow.log_params(model_config['params'])
        mlflow.log_metrics(metrics)
        print(f"Metrics logged for {model_name}: {metrics}")

        # 5. Save the complete pipeline as an artifact in MLflow
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            registered_model_name=f"{model_name.replace(' ', '_')}_Pipeline"
        )

        return full_pipeline, metrics


def run_training_pipeline():
    """Executes the end-to-end training process."""

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    create_dirs([MODEL_DIR])

    if not RAW_DATA_PATH.exists():
        print("Raw data not found. Please run download_dataset.py first.")
        return

    raw_df = pd.read_csv(RAW_DATA_PATH)

    # 1. Preprocess and Split
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(raw_df)

    # 2. Train and Log Models
    all_pipelines = {}
    all_metrics = {}

    for name, config in MODEL_CONFIGS.items():
        pipeline, metrics = train_and_log_model(name, config, preprocessor,
                                                X_train, X_test, y_train, y_test)
        all_pipelines[name] = pipeline
        all_metrics[name] = metrics

    # 3. Select and Save the Best Model
    best_pipeline = all_pipelines[BEST_MODEL_NAME]

    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\n🏆 Best Model: {BEST_MODEL_NAME}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Test Metrics: {all_metrics[BEST_MODEL_NAME]}")


if __name__ == "__main__":
    run_training_pipeline()
