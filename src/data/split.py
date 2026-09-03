"""Leakage-aware train/test splitting utilities."""

import pandas as pd


def split_by_unit(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete trajectories by unit identifier.

    Units, rather than individual observations, are assigned to train or test.
    This prevents observations from the same trajectory appearing in both
    partitions.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    if "unit_id" not in df.columns:
        raise ValueError("The dataframe must contain 'unit_id'.")

    units = pd.Index(df["unit_id"].dropna().unique()).sort_values()
    if len(units) < 2:
        raise ValueError("At least two units are required for a split.")

    n_test = max(1, int(round(len(units) * test_fraction)))
    n_test = min(n_test, len(units) - 1)

    test_units = set(units[-n_test:])
    test_mask = df["unit_id"].isin(test_units)

    train = df.loc[~test_mask].copy()
    test = df.loc[test_mask].copy()

    if set(train["unit_id"]).intersection(test["unit_id"]):
        raise RuntimeError("Unit leakage detected between train and test sets.")

    return train, test
