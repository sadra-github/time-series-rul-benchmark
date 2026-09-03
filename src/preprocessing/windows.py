"""Leakage-aware temporal window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_sequence_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
    window_size: int = 20,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct fixed-length windows independently within each unit.

    Windows never cross unit boundaries and each target is aligned with the
    final observation in its corresponding input window.
    """
    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")

    required = {"unit_id", "cycle", target_column, *feature_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Cannot construct windows from an empty dataframe.")

    X_windows: list[np.ndarray] = []
    y_values: list[float] = []
    metadata: list[dict[str, int | float]] = []

    ordered = df.sort_values(["unit_id", "cycle"])

    for unit_id, unit_df in ordered.groupby("unit_id", sort=False):
        unit_df = unit_df.sort_values("cycle")
        features = unit_df[feature_columns].to_numpy(dtype=float)
        targets = unit_df[target_column].to_numpy(dtype=float)
        cycles = unit_df["cycle"].to_numpy()

        if len(unit_df) < window_size:
            continue

        for start in range(0, len(unit_df) - window_size + 1, stride):
            end = start + window_size
            X_windows.append(features[start:end])
            y_values.append(targets[end - 1])
            metadata.append(
                {
                    "unit_id": unit_id,
                    "start_cycle": cycles[start],
                    "end_cycle": cycles[end - 1],
                }
            )

    if not X_windows:
        raise ValueError("No complete windows can be constructed with this window_size.")

    return (
        np.stack(X_windows),
        np.asarray(y_values, dtype=float),
        pd.DataFrame(metadata),
    )
