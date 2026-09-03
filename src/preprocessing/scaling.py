"""Training-only feature scaling utilities."""

from typing import Iterable

import pandas as pd
from sklearn.preprocessing import StandardScaler


class TrainingOnlyScaler:
    """Fit a StandardScaler on training data and reuse it for other splits."""

    def __init__(self, feature_columns: Iterable[str]):
        self.feature_columns = list(feature_columns)
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty.")
        self._scaler = StandardScaler()
        self._fitted = False

    def fit(self, train: pd.DataFrame) -> "TrainingOnlyScaler":
        """Fit scaling parameters using training data only."""
        self._validate_columns(train)
        self._scaler.fit(train[self.feature_columns])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform a dataframe using parameters learned from training data."""
        if not self._fitted:
            raise RuntimeError("Scaler must be fitted on training data first.")

        self._validate_columns(df)
        result = df.copy()
        result.loc[:, self.feature_columns] = self._scaler.transform(
            df[self.feature_columns]
        )
        return result

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        """Fit on training data and transform that same training split."""
        self.fit(train)
        return self.transform(train)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        if df.empty:
            raise ValueError("Cannot scale an empty dataframe.")
