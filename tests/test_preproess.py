# tests/test_preprocess.py

import pytest
import pandas as pd
from src.preprocess import preprocess_and_split
from src.config import RAW_DATA_PATH, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES


@pytest.fixture(scope="session")
def raw_data_df():
    """Load the raw data for testing."""
    if not RAW_DATA_PATH.exists():
        pytest.fail(f"Raw data file not found at {RAW_DATA_PATH}. Please run download_dataset.py.")
    return pd.read_csv(RAW_DATA_PATH)


def test_data_loading_and_shape(raw_data_df):
    """Test that the raw data loaded has the correct number of columns."""
    # 13 features + 1 target = 14 columns
    assert raw_data_df.shape[1] == 14
    assert raw_data_df.columns[-1] == TARGET_COLUMN


def test_preprocess_output_shapes(raw_data_df):
    """Test that the split datasets have the correct shape and non-zero size."""
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(raw_data_df)

    # Check non-empty
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

    # Check feature count (13 original features)
    assert X_train.shape[1] == 13


def test_target_binary_conversion(raw_data_df):
    """Test that the target variable is correctly converted to binary (0 or 1)."""
    _, _, y_train, y_test, _ = preprocess_and_split(raw_data_df)

    combined_y = pd.concat([y_train, y_test])
    # Check that only 0s and 1s exist in the target
    assert set(combined_y.unique()).issubset({0, 1})
    # Check that both classes are present
    assert combined_y.nunique() == 2