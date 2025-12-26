# src/preprocess.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.utils import create_dirs
from src.config import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_COLUMN,
    TEST_SIZE, RANDOM_STATE, DATA_PROCESSED_DIR,
    RAW_DATA_PATH  # Import RAW_DATA_PATH from config
)


def preprocess_and_split(raw_df: pd.DataFrame):
    """
    Cleans, transforms, and splits the raw data, handling specific UCI dataset quirks.
    """

    # 1. Data Cleaning
    # (Handling '?' as missing values - already loaded as NaN in download_dataset.py)
    # The download script loads '?' as NaN. We drop rows with ANY missing value (ca or thal).
    df = raw_df.copy()
    initial_shape = df.shape[0]

    # Drop rows containing missing values (NaN resulted from '?' in ca and thal)
    df.dropna(inplace=True)

    print(f"Data Cleaning: Dropped {initial_shape - df.shape[0]} "
          f"rows containing missing values ('?').")

    # 2. Type Conversion (Crucial for UCI Cleveland data)
    # The 'ca' (number of major vessels) and 'thal' (thallium stress test) columns # noqa: E501
    # were loaded as objects/strings due to the original '?' values. We must convert them to numeric. # noqa: E501
    # We must also convert to integer type to be treated as categorical features in the pipeline. # noqa: E501
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce').astype(int)
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce').astype(int)

    # 3. Target Encoding: Convert 0-4 targets to binary (0=No Disease, 1=Disease)
    # The 'target' column in the UCI Cleveland data is 0-4.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].apply(lambda x: 1 if x > 0 else 0)

    print(f"Target distribution after binarization (0=No Disease, 1=Disease):\n"
          f"{df[TARGET_COLUMN].value_counts()}")

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    # 4. Define Preprocessing Transformer (Scaling and Encoding)
    # This remains robust due to scikit-learn's ColumnTransformer structure.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

    preprocessor.fit(X)

    create_dirs([DATA_PROCESSED_DIR])
    print("Preprocessor fitted successfully.")

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == '__main__':
    if not RAW_DATA_PATH.exists():
        print("Raw data not found. Please run download_dataset.py first.")
    else:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(raw_df)
        print(f"\nTraining set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
