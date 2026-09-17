"""RUL label construction for truncated C-MAPSS test trajectories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_cmapss_rul(path: str | Path) -> pd.Series:
    """Load one terminal RUL value for each C-MAPSS test unit."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"RUL file not found: {path}")

    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(
        axis=1, how="all"
    )

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


def validate_test_rul_alignment(
    test_df: pd.DataFrame,
    terminal_rul: pd.Series,
) -> None:
    """Validate one terminal RUL value for every test unit, with no extras."""
    if "unit_id" not in test_df.columns:
        raise ValueError("The test dataframe must contain 'unit_id'.")

    if terminal_rul.index.has_duplicates:
        raise ValueError("Terminal RUL unit identifiers must be unique.")

    test_units = pd.Index(test_df["unit_id"].unique()).sort_values()
    rul_units = pd.Index(terminal_rul.index).sort_values()

    missing_rul = test_units.difference(rul_units)
    extra_rul = rul_units.difference(test_units)

    if len(missing_rul) or len(extra_rul):
        details = []
        if len(missing_rul):
            details.append(f"missing terminal RUL for test units: {missing_rul.tolist()}")
        if len(extra_rul):
            details.append(f"RUL values supplied for non-test units: {extra_rul.tolist()}")
        raise ValueError("Test RUL alignment mismatch; " + "; ".join(details))


def add_test_rul(test_df: pd.DataFrame, terminal_rul: pd.Series) -> pd.DataFrame:
    """Add row-level RUL labels to truncated C-MAPSS test trajectories.

    The supplied terminal RUL corresponds to the final observed cycle of each
    test unit. Earlier-cycle labels are reconstructed from cycle indices.
    """
    required = {"unit_id", "cycle"}
    missing = required.difference(test_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if terminal_rul.index.name != "unit_id":
        terminal_rul = terminal_rul.copy()
        terminal_rul.index.name = "unit_id"

    validate_test_rul_alignment(test_df, terminal_rul)

    result = test_df.copy()
    terminal = result["unit_id"].map(terminal_rul)
    final_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = terminal + (final_cycle - result["cycle"])
    return result
