"""Data loading and temporal validation utilities for C-MAPSS."""

from pathlib import Path
from typing import Iterable

import pandas as pd


COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def load_cmapss_txt(path: str | Path) -> pd.DataFrame:
    """Load a whitespace-delimited C-MAPSS FD001-style file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"Expected {len(COLUMNS)} columns, found {df.shape[1]} in {path}."
        )

    df.columns = COLUMNS
    return df


def validate_cmapss_schema(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
) -> None:
    """Validate structural assumptions required by the benchmark."""
    required = list(required_columns or COLUMNS)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("The dataset is empty.")
    if df["unit_id"].isna().any() or df["cycle"].isna().any():
        raise ValueError("unit_id and cycle must not contain missing values.")
    if not pd.api.types.is_numeric_dtype(df["unit_id"]):
        raise TypeError("unit_id must be numeric.")
    if not pd.api.types.is_numeric_dtype(df["cycle"]):
        raise TypeError("cycle must be numeric.")
    if (df["cycle"] < 1).any():
        raise ValueError("cycle values must be positive.")


def validate_temporal_order(df: pd.DataFrame) -> None:
    """Check that input rows are chronologically ordered within each unit.

    Unlike window construction, this validator intentionally does not sort the
    dataframe first. It therefore detects an input ordering violation rather
    than merely checking whether the cycles can be sorted into a valid order.
    """
    required = {"unit_id", "cycle"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cycle_diff = df.groupby("unit_id", sort=False)["cycle"].diff()
    if (cycle_diff.dropna() <= 0).any():
        raise ValueError(
            "Cycle indices must be strictly increasing within each unit."
        )
