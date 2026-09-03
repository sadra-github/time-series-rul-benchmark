"""End-to-end baseline pipeline for the C-MAPSS RUL benchmark."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cmapss_protocol import load_cmapss_fd001_protocol
from src.data.rul import add_training_rul
from src.data.validation import split_train_validation
from src.preprocessing.scaling import TrainingOnlyScaler
from src.preprocessing.windows import make_sequence_windows
from src.models import LastValueBaseline, LinearRULBaseline
from src.evaluation.runner import evaluate_model


FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def prepare_development_data(
    train_path: str | Path,
    validation_fraction: float = 0.2,
    validation_seed: int = 42,
    window_size: int = 20,
    stride: int = 1,
) -> tuple[tuple, tuple]:
    """Prepare training and validation windows without touching official test data."""
    train_df, _ = load_cmapss_fd001_protocol(train_path, train_path)
    train_df = add_training_rul(train_df)
    train_part, validation_part = split_train_validation(
        train_df,
        validation_fraction=validation_fraction,
        random_seed=validation_seed,
    )

    scaler = TrainingOnlyScaler(FEATURE_COLUMNS)
    train_part = scaler.fit_transform(train_part)
    validation_part = scaler.transform(validation_part)

    train_windows = make_sequence_windows(
        train_part,
        FEATURE_COLUMNS,
        window_size=window_size,
        stride=stride,
    )
    validation_windows = make_sequence_windows(
        validation_part,
        FEATURE_COLUMNS,
        window_size=window_size,
        stride=stride,
    )

    return train_windows, validation_windows


def run_baselines(
    train_path: str | Path,
    validation_fraction: float = 0.2,
    validation_seed: int = 42,
    window_size: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    """Train the initial baselines and return a common metric table."""
    train_windows, validation_windows = prepare_development_data(
        train_path,
        validation_fraction=validation_fraction,
        validation_seed=validation_seed,
        window_size=window_size,
        stride=stride,
    )
    X_train, y_train, _ = train_windows
    X_validation, y_validation, _ = validation_windows

    models = {
        "last_value": LastValueBaseline(),
        "linear_regression": LinearRULBaseline(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_validation, y_validation)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows)
