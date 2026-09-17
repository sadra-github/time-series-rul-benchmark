"""Simple, dependency-light RUL baseline models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class MeanRULBaseline:
    """Predict the mean training RUL for every sample."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MeanRULBaseline":
        """Store the mean RUL computed from the training targets."""
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        if len(y) == 0:
            raise ValueError("Training data must not be empty.")
        self._mean_rul = float(np.mean(y))
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the training-set mean RUL for every sample."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Baseline must be fitted before prediction.")
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        return np.full(len(X), self._mean_rul, dtype=float)


class LastValueBaseline:
    """Predict the target using the last observed feature value.

    This baseline is retained as a diagnostic reference. It is not the primary
    naive RUL baseline because the last feature is not intrinsically an RUL
    quantity.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LastValueBaseline":
        """Validate training arrays and return the fitted baseline."""
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the last feature value at the final timestep."""
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Baseline must be fitted before prediction.")
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        return X[:, -1, -1]


class LinearRULBaseline:
    """Linear regression on flattened temporal windows."""

    def __init__(self) -> None:
        self._model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRULBaseline":
        """Fit linear regression using only the supplied training data."""
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples.")
        self._model.fit(X.reshape(len(X), -1), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL from flattened temporal windows."""
        if X.ndim != 3:
            raise ValueError("X must have shape (samples, timesteps, features).")
        return self._model.predict(X.reshape(len(X), -1))
