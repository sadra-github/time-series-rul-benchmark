"""End-to-end baseline pipeline for C-MAPSS development and test experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.cmapss_protocol import load_cmapss_fd001_protocol
from src.data.loader import (
    load_cmapss_txt,
    validate_cmapss_schema,
    validate_temporal_order,
)
from src.data.rul import add_training_rul
from src.data.test_rul import add_test_rul, load_cmapss_rul
from src.data.validation import split_train_validation
from src.preprocessing.scaling import TrainingOnlyScaler
from src.preprocessing.windows import make_sequence_windows
from src.models import (
    LastValueBaseline,
    LinearRULBaseline,
    MeanRULBaseline,
    RandomForestRULBaseline,
)
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
    """Prepare training and validation windows from train_FD001 only."""
    train_df = load_cmapss_txt(train_path)
    validate_cmapss_schema(train_df)
    validate_temporal_order(train_df)
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


def prepare_official_test_data(
    train_path: str | Path,
    test_path: str | Path,
    rul_path: str | Path,
    window_size: int = 20,
    stride: int = 1,
) -> tuple[tuple, tuple]:
    """Prepare full-training and official FD001 test windows.

    Scaling parameters are fitted on the complete training partition only.
    The official test partition is kept separate and receives its terminal RUL
    values from the supplied C-MAPSS RUL file.
    """
    train_df, test_df = load_cmapss_fd001_protocol(train_path, test_path)
    terminal_rul = load_cmapss_rul(rul_path)
    train_df = add_training_rul(train_df)
    test_df = add_test_rul(test_df, terminal_rul)

    scaler = TrainingOnlyScaler(FEATURE_COLUMNS)
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    train_windows = make_sequence_windows(
        train_df,
        FEATURE_COLUMNS,
        window_size=window_size,
        stride=stride,
    )
    test_windows = make_sequence_windows(
        test_df,
        FEATURE_COLUMNS,
        window_size=window_size,
        stride=stride,
    )

    return train_windows, test_windows


def run_baselines(
    train_path: str | Path,
    validation_fraction: float = 0.2,
    validation_seed: int = 42,
    window_size: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    """Train initial baselines and return a common validation metric table."""
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
        "mean_rul": MeanRULBaseline(),
        "linear_regression": LinearRULBaseline(),
        "random_forest": RandomForestRULBaseline(),
        "last_value": LastValueBaseline(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_validation, y_validation)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows)


def run_official_test_benchmark(
    train_path: str | Path,
    test_path: str | Path,
    rul_path: str | Path,
    window_size: int = 20,
    stride: int = 1,
) -> pd.DataFrame:
    """Fit baseline models on all training units and evaluate official FD001 test."""
    train_windows, test_windows = prepare_official_test_data(
        train_path,
        test_path,
        rul_path,
        window_size=window_size,
        stride=stride,
    )
    X_train, y_train, _ = train_windows
    X_test, y_test, _ = test_windows

    models = {
        "mean_rul": MeanRULBaseline(),
        "linear_regression": LinearRULBaseline(),
        "random_forest": RandomForestRULBaseline(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        rows.append({"model": name, **metrics})

    return pd.DataFrame(rows)
