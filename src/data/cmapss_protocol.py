"""Protocol helpers for the official C-MAPSS train/test organization."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader import load_cmapss_txt, validate_cmapss_schema, validate_temporal_order


def load_cmapss_fd001_protocol(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the official FD001 train/test partitions.

    C-MAPSS train and test files use local unit identifiers. Therefore matching
    numeric unit IDs across the two files do not indicate leakage. The files
    remain separate and are never merged or resplit by this helper.
    """
    train = load_cmapss_txt(train_path)
    test = load_cmapss_txt(test_path)

    for name, frame in (("train", train), ("test", test)):
        validate_cmapss_schema(frame)
        validate_temporal_order(frame)
        if name == "train" and frame["unit_id"].nunique() < 2:
            raise ValueError("The training partition must contain multiple units.")

    return train, test
