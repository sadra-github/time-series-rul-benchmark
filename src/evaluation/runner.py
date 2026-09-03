"""Model evaluation helpers for the benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import evaluate_regression


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    """Generate predictions and evaluate a fitted model with common metrics."""
    predictions = np.asarray(model.predict(X_test), dtype=float)
    return evaluate_regression(np.asarray(y_test, dtype=float), predictions)
