"""Validation-only experiments for temporal window sensitivity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pipeline import run_baselines


def compare_window_sizes(
    train_path: str | Path,
    window_sizes: list[int],
    validation_fraction: float = 0.2,
    validation_seed: int = 42,
    stride: int = 1,
) -> pd.DataFrame:
    """Evaluate fixed window sizes on the development validation partition.

    The official C-MAPSS test partition is not used here. This keeps temporal
    representation selection separate from the final test evaluation.
    """
    if not window_sizes:
        raise ValueError("window_sizes must not be empty.")
    if any(size < 1 for size in window_sizes):
        raise ValueError("All window sizes must be positive.")

    results = []
    for window_size in window_sizes:
        metrics = run_baselines(
            train_path,
            validation_fraction=validation_fraction,
            validation_seed=validation_seed,
            window_size=window_size,
            stride=stride,
        )
        metrics.insert(0, "window_size", window_size)
        results.append(metrics)

    return pd.concat(results, ignore_index=True)
