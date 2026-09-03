"""RUL label construction for truncated C-MAPSS test trajectories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cmapss_rul(path: str | Path) -> pd.Series:
    """Load one terminal RUL value for each C-MAPSS test unit."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")

    if values.shape[1] != 1:
        raise ValueError(f"Expected one RUL column, found {values.shape[1]} in {path}.")

    rul = pd.to_numeric(values.iloc[:, 0], errors="raise")
    if rul.empty:
        raise ValueError("The RUL file is empty.")
    if (rul < 0).any():
        raise ValueError("Terminal RUL values must be non-negative.")

    rul.index = pd.RangeIndex(start=1, stop=len(rul) + 1, name="unit_id")
    rul.name = "terminal_rul"
    return rul


def add_test_rul(test_df: pd.DataFrame, terminal_rul: pd.Series) -> pd.DataFrame:
    """Add row-level RUL labels to truncated test trajectories.

    The supplied terminal RUL corresponds to the final observed cycle of each
    test unit. Earlier-cycle labels are reconstructed without using future
    sensor observations.
    """
    required = {"unit_id", "cycle"}
    missing = required.difference(test_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if terminal_rul.index.name != "unit_id":
        terminal_rul = terminal_rul.copy()
        terminal_rul.index.name = "unit_id"

    units = pd.Index(test_df["unit_id"].unique())
    missing_rul = units.difference(terminal_rul.index)
    if len(missing_rul):
        raise ValueError(f"Missing terminal RUL values for test units: {missing_rul.tolist()}")

    result = test_df.copy()
    terminal = result["unit_id"].map(terminal_rul)
    final_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = terminal + (final_cycle - result["cycle"])
    return result
