"""Simple, dependency-light RUL baseline models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression


class LastValueBaseline:
    """Predict the target using the last observed feature value.

    This baseline is intentionally generic: it uses the final feature in each
    flattened window and therefore provides a simple reference point before
    introducing learned nonlinear models.
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
