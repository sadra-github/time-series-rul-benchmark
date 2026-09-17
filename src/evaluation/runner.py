"""Model evaluation helpers for the benchmark."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

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


def evaluate_model_by_unit(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    unit_ids: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Evaluate a fitted model separately for each test unit."""
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test, dtype=float)
    unit_ids = np.asarray(unit_ids)

    if X_test.ndim != 3:
        raise ValueError("X_test must have shape (samples, timesteps, features).")
    if len(X_test) == 0:
        raise ValueError("X_test must not be empty.")
    if len(y_test) != len(X_test) or len(unit_ids) != len(X_test):
        raise ValueError("X_test, y_test, and unit_ids must have the same length.")

    predictions = np.asarray(model.predict(X_test), dtype=float)
    rows = []
    for unit_id in pd.unique(unit_ids):
        mask = unit_ids == unit_id
        metrics = evaluate_regression(y_test[mask], predictions[mask])
        rows.append({"unit_id": unit_id, **metrics, "n_samples": int(mask.sum())})

    return pd.DataFrame(rows)
