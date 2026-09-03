"""Random Forest RUL baseline."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestRULBaseline:
    """Random Forest regression on flattened temporal windows."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestRULBaseline":
        """Fit the model using only the supplied training windows."""
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
