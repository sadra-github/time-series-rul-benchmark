"""Model evaluation helpers for the benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np

from .metrics import evaluate_regression


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
    """Generate predictions and evaluate a fitted model with common metrics."""
    X_test = np.asarray(X_test)
    if X_test.ndim != 3:
        raise ValueError("X_test must have shape (samples, timesteps, features).")
    if len(X_test) == 0:
        raise ValueError("X_test must not be empty.")

    predictions = np.asarray(model.predict(X_test), dtype=float)
    return evaluate_regression(np.asarray(y_test, dtype=float), predictions)
