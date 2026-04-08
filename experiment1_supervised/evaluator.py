"""Evaluation helpers for experiment metrics and reports."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def predict_binary(model, x_split) -> Tuple[np.ndarray, np.ndarray]:
    probs = model.predict(x_split, verbose=0).ravel()
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def evaluate_predictions(y_true, y_pred) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, digits=4, zero_division=0
        ),
    }
