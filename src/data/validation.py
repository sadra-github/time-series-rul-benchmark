"""Unit-level validation splitting for development experiments."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def split_train_validation(
    train_df: pd.DataFrame,
    validation_fraction: float = 0.2,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the official training partition into disjoint development units.

    The split is performed at the unit level, so observations from one engine
    cannot appear in both training and validation sets. The official C-MAPSS
    test partition is intentionally not involved in this function.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if "unit_id" not in train_df.columns:
        raise ValueError("The dataframe must contain 'unit_id'.")

    units = train_df["unit_id"].dropna().unique()
    if len(units) < 2:
        raise ValueError("At least two units are required for validation splitting.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=validation_fraction,
        random_state=random_seed,
    )
    train_idx, validation_idx = next(
        splitter.split(train_df, groups=train_df["unit_id"])
    )

    train = train_df.iloc[train_idx].copy()
    validation = train_df.iloc[validation_idx].copy()

    if set(train["unit_id"]).intersection(validation["unit_id"]):
        raise RuntimeError("Unit leakage detected between training and validation.")

    return train, validation
