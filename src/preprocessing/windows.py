"""Leakage-aware temporal window construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.loader import validate_finite_values, validate_temporal_order


def make_sequence_windows(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "rul",
    window_size: int = 20,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Construct fixed-length windows independently within each unit.

    Windows follow the supplied temporal order and never cross unit
    boundaries. Each target is aligned with the final observation in its
    corresponding input window. Cycle gaps are retained in metadata rather
    than treated as missing observations.
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

    validate_temporal_order(df)
    validate_finite_values(df, [*feature_columns, target_column])

    X_windows: list[np.ndarray] = []
    y_values: list[float] = []
    metadata: list[dict[str, int | float]] = []

    for unit_id, unit_df in df.groupby("unit_id", sort=False):
        features = unit_df[feature_columns].to_numpy(dtype=float)
        targets = unit_df[target_column].to_numpy(dtype=float)
        cycles = unit_df["cycle"].to_numpy()

        if len(unit_df) < window_size:
            continue

        for start in range(0, len(unit_df) - window_size + 1, stride):
            end = start + window_size
            window_cycles = cycles[start:end]
            cycle_diffs = np.diff(window_cycles)
            X_windows.append(features[start:end])
            y_values.append(targets[end - 1])
            metadata.append(
                {
                    "unit_id": unit_id,
                    "start_cycle": window_cycles[0],
                    "end_cycle": window_cycles[-1],
                    "cycle_span": window_cycles[-1] - window_cycles[0],
                    "max_cycle_gap": (
                        float(cycle_diffs.max()) if len(cycle_diffs) else 0.0
                    ),
                }
            )

    if not X_windows:
        raise ValueError("No complete windows can be constructed with this window_size.")

    return (
        np.stack(X_windows),
        np.asarray(y_values, dtype=float),
        pd.DataFrame(metadata),
    )
