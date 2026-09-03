"""RUL label construction for complete training trajectories."""

import pandas as pd


def add_training_rul(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL labels from the final observed cycle of each training unit.

    This function is intended for complete training trajectories. It does not
    infer labels for an operational test trajectory whose failure cycle is
    unknown.
    """
    required = {"unit_id", "cycle"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    result = df.copy()
    last_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = last_cycle - result["cycle"]
    return result
