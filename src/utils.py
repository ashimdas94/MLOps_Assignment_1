# src/utils.py

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from pathlib import Path

def get_metrics(y_true, y_pred, y_prob):
    """Calculates and returns key classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }

def create_dirs(paths: list[Path]):
    """Creates directories if they do not exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)