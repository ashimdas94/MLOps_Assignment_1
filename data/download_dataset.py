# data/download_dataset.py

import pandas as pd
import requests
from io import StringIO
from pathlib import Path
from src.config import RAW_DATA_PATH, DATA_DOWNLOAD_URL, DATA_RAW_DIR, COLUMN_NAMES
from src.utils import create_dirs


def download_data(url: str, output_path: Path):
    """Downloads the raw UCI data, applies headers, and saves it as CSV."""

    create_dirs([DATA_RAW_DIR])
    print(f"Attempting to download data from: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()

        # 1. Read the raw data (which lacks headers)
        data = StringIO(response.text)

        # Load data, specifying no header, and marking '?' as missing values
        df = pd.read_csv(data, sep=',', header=None, na_values='?')

        # 2. Apply the correct column names from config
        df.columns = COLUMN_NAMES

        # 3. Save the structured data (The '?' handling is still done later in preprocess.py,
        # but the headers are crucial here.)
        df.to_csv(output_path, index=False)
        print(f"Data successfully downloaded and saved to: {output_path}")
        print(f"Dataset shape: {df.shape}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading data: {e}")
        raise RuntimeError(f"Data download failed: {e}")


if __name__ == "__main__":
    download_data(DATA_DOWNLOAD_URL, RAW_DATA_PATH)