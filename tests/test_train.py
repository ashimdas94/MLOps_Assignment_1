# tests/test_train.py

import pytest
import json
import pandas as pd
from src.config import MODEL_PATH, RAW_DATA_PATH
from src.inference import load_model_pipeline, predict_heart_disease
from src.utils import get_metrics
from src.preprocess import preprocess_and_split


# --- Fixtures ---

@pytest.fixture(scope="session")
def trained_pipeline():
    """Attempt to load the trained model pipeline."""
    if not MODEL_PATH.exists():
        pytest.fail(f"Model file not found at {MODEL_PATH}. Run src/train.py first.")
    return load_model_pipeline(MODEL_PATH)


@pytest.fixture(scope="session")
def test_data():
    """Load the test split data for metric validation."""
    # Assuming raw data exists from test_preprocess setup
    raw_df = pd.read_csv(RAW_DATA_PATH)
    _, X_test, _, y_test, _ = preprocess_and_split(raw_df)
    return X_test, y_test


@pytest.fixture(scope="session")
def sample_inputs():
    """Load sample inputs for integration tests."""
    with open("tests/sample_input.json", 'r') as f:
        return json.load(f)


# --- Model Quality Tests ---

def test_model_performance(trained_pipeline, test_data):
    """Test if the model meets a minimum acceptable performance threshold (e.g., AUC > 0.7)."""
    X_test, y_test = test_data

    # Use the pipeline to predict on the test set
    y_pred = trained_pipeline.predict(X_test)
    y_prob = trained_pipeline.predict_proba(X_test)[:, 1]

    metrics = get_metrics(y_test, y_pred, y_prob)

    # Check Minimum Performance Requirement (Assignment Step 3)
    MIN_AUC_THRESHOLD = 0.70
    assert metrics['roc_auc'] >= MIN_AUC_THRESHOLD, (
        f"Model failed minimum AUC threshold check. "
        f"Got {metrics['roc_auc']:.2f}, required {MIN_AUC_THRESHOLD:.2f}"
    )
    print(f"\nModel AUC: {metrics['roc_auc']:.2f} (Passed threshold check)")


# --- Inference/API Integration Tests ---

def test_model_loading(trained_pipeline):
    """Test that the model object is an scikit-learn Pipeline."""
    from sklearn.pipeline import Pipeline
    assert isinstance(trained_pipeline, Pipeline)
    assert 'classifier' in trained_pipeline.named_steps


def test_inference_with_sample_inputs(sample_inputs):
    """Test if the inference function predicts the expected class for known samples."""
    for sample in sample_inputs:
        result = predict_heart_disease(sample['input'])

        # Check if the predicted class matches the expected class
        assert result['prediction'] == sample['expected_class'], (
            f"Inference failed for {sample['name']}. "
            f"Expected {sample['expected_class']}, got {result['prediction']}"
        )
        # Check if the probability is reasonable (between 0.0 and 1.0)
        assert 0.0 <= result['probability'] <= 1.0

        print(f"Inference test passed for {sample['name']}. Prediction: {result['prediction']}")
