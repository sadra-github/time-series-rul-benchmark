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

    The supplied C-MAPSS training and test files remain separate. This helper
    does not merge, resplit, or fit any preprocessing operation across them.
    """
    train = load_cmapss_txt(train_path)
    test = load_cmapss_txt(test_path)

    for name, frame in (("train", train), ("test", test)):
        validate_cmapss_schema(frame)
        validate_temporal_order(frame)
        if name == "train" and frame["unit_id"].nunique() < 2:
            raise ValueError("The training partition must contain multiple units.")

    if set(train["unit_id"]).intersection(test["unit_id"]):
        raise ValueError("Train and test unit identifiers must be disjoint.")

    return train, test
