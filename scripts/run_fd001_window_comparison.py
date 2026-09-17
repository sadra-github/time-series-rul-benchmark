"""Stage 8: Temporal window comparison on the FD001 internal validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import (
    load_cmapss_txt,
    validate_cmapss_schema,
    validate_temporal_order,
)
from src.data.rul import add_training_rul
from src.data.validation import split_train_validation
from src.evaluation.runner import evaluate_model
from src.models import (
    LinearRULBaseline,
    MeanRULBaseline,
    RandomForestRULBaseline,
)
from src.preprocessing.scaling import TrainingOnlyScaler
from src.preprocessing.windows import make_sequence_windows


FEATURE_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

WINDOW_SIZES = [5, 10, 20, 30, 50]
STRIDE = 1
VALIDATION_FRACTION = 0.20
VALIDATION_SEED = 42


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare temporal window sizes for C-MAPSS FD001 RUL."
    )
    parser.add_argument(
        "--train",
        required=True,
        help="Path to train_FD001.txt",
    )
    parser.add_argument(
        "--results",
        default="results",
        help="Directory for Stage 8 results.",
    )
    return parser.parse_args()


def prepare_fixed_split(
    train_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FD001 training data and create one fixed unit-level split."""
    train_df = load_cmapss_txt(train_path)

    validate_cmapss_schema(train_df)
    validate_temporal_order(train_df)

    train_df = add_training_rul(train_df)

    train_part, validation_part = split_train_validation(
        train_df,
        validation_fraction=VALIDATION_FRACTION,
        random_seed=VALIDATION_SEED,
    )

    train_units = set(train_part["unit_id"].unique())
    validation_units = set(validation_part["unit_id"].unique())

    if train_units.intersection(validation_units):
        raise RuntimeError(
            "Unit leakage detected between development train and validation."
        )

    return train_part, validation_part


def run_window_comparison(
    train_part: pd.DataFrame,
    validation_part: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all predefined windows using the same train/validation units."""
    rows: list[dict[str, float | int | str]] = []
    window_rows: list[dict[str, float | int]] = []

    for window_size in WINDOW_SIZES:
        scaler = TrainingOnlyScaler(FEATURE_COLUMNS)

        scaled_train = scaler.fit_transform(train_part)
        scaled_validation = scaler.transform(validation_part)

        train_windows = make_sequence_windows(
            scaled_train,
            FEATURE_COLUMNS,
            window_size=window_size,
            stride=STRIDE,
        )

        validation_windows = make_sequence_windows(
            scaled_validation,
            FEATURE_COLUMNS,
            window_size=window_size,
            stride=STRIDE,
        )

        X_train, y_train, train_metadata = train_windows
        X_validation, y_validation, validation_metadata = validation_windows

        window_rows.append(
            {
                "window_size": window_size,
                "stride": STRIDE,
                "train_units": train_part["unit_id"].nunique(),
                "validation_units": validation_part["unit_id"].nunique(),
                "train_windows": len(X_train),
                "validation_windows": len(X_validation),
                "train_mean_cycle_span": float(
                    train_metadata["cycle_span"].mean()
                ),
                "validation_mean_cycle_span": float(
                    validation_metadata["cycle_span"].mean()
                ),
                "train_max_cycle_gap": float(
                    train_metadata["max_cycle_gap"].max()
                ),
                "validation_max_cycle_gap": float(
                    validation_metadata["max_cycle_gap"].max()
                ),
            }
        )

        models = {
            "mean_rul": MeanRULBaseline(),
            "linear_regression": LinearRULBaseline(),
            "random_forest": RandomForestRULBaseline(),
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            metrics = evaluate_model(
                model,
                X_validation,
                y_validation,
            )

            rows.append(
                {
                    "window_size": window_size,
                    "stride": STRIDE,
                    "model": model_name,
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(window_rows)


def save_metric_figures(
    results_df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Save validation metric versus temporal window size figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_specs = [
        ("mae", "Validation MAE", "fd001_validation_mae_vs_window.png"),
        ("rmse", "Validation RMSE", "fd001_validation_rmse_vs_window.png"),
        ("r2", "Validation R2", "fd001_validation_r2_vs_window.png"),
    ]

    for metric, ylabel, filename in metric_specs:
        fig, ax = plt.subplots(figsize=(8, 5))

        for model_name in [
            "mean_rul",
            "linear_regression",
            "random_forest",
        ]:
            model_df = results_df[
                results_df["model"] == model_name
            ].sort_values("window_size")

            ax.plot(
                model_df["window_size"],
                model_df[metric],
                marker="o",
                label=model_name,
            )

        ax.set_xlabel("Window size")
        ax.set_ylabel(ylabel)
        ax.set_title(f"FD001 Validation {ylabel} vs Window Size")
        ax.set_xticks(WINDOW_SIZES)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        fig.savefig(figures_dir / filename, dpi=200)
        plt.close(fig)


def save_window_construction_figure(
    train_part: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Visualize how different windows cover one representative trajectory."""
    figures_dir.mkdir(parents=True, exist_ok=True)

    representative_unit = train_part["unit_id"].iloc[0]
    unit_df = train_part[
        train_part["unit_id"] == representative_unit
    ].sort_values("cycle")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        unit_df["cycle"],
        unit_df["rul"],
        label="RUL trajectory",
        linewidth=2,
    )

    colors = ["C1", "C2", "C3", "C4", "C5"]

    for window_size, color in zip(WINDOW_SIZES, colors):
        if len(unit_df) >= window_size:
            start_cycle = unit_df["cycle"].iloc[0]
            end_cycle = unit_df["cycle"].iloc[window_size - 1]

            ax.axvspan(
                start_cycle,
                end_cycle,
                alpha=0.15,
                color=color,
                label=f"window={window_size}",
            )

    ax.set_xlabel("Cycle")
    ax.set_ylabel("RUL")
    ax.set_title(
        f"FD001 Temporal Window Construction "
        f"(Representative Unit {representative_unit})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figures_dir / "fd001_window_construction_example.png",
        dpi=200,
    )
    plt.close(fig)


def main() -> None:
    """Run Stage 8 and save validation results."""
    args = parse_args()

    results_dir = Path(args.results)
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("C-MAPSS FD001 Stage 8: Temporal Window Comparison")
    print("=" * 58)
    print(f"window_sizes={WINDOW_SIZES}")
    print(f"stride={STRIDE}")
    print(f"validation_fraction={VALIDATION_FRACTION}")
    print(f"validation_seed={VALIDATION_SEED}")
    print("official_test_used=False")
    print()

    print("Preparing fixed unit-level validation split...")
    train_part, validation_part = prepare_fixed_split(args.train)

    print(
        f"Development train units: "
        f"{train_part['unit_id'].nunique()}"
    )
    print(
        f"Validation units: "
        f"{validation_part['unit_id'].nunique()}"
    )
    print()

    results_df, window_df = run_window_comparison(
        train_part,
        validation_part,
    )

    results_df = results_df.sort_values(
        ["window_size", "model"]
    ).reset_index(drop=True)

    window_df = window_df.sort_values(
        "window_size"
    ).reset_index(drop=True)

    results_path = (
        tables_dir / "fd001_window_comparison_validation.csv"
    )
    window_path = (
        tables_dir / "fd001_window_counts_validation.csv"
    )

    results_df.to_csv(results_path, index=False)
    window_df.to_csv(window_path, index=False)

    save_metric_figures(
        results_df,
        figures_dir,
    )

    save_window_construction_figure(
        train_part,
        figures_dir,
    )

    print("Validation results:")
    print(results_df.to_string(index=False))
    print()

    print("Window construction summary:")
    print(window_df.to_string(index=False))
    print()

    print(f"Results written to: {results_path}")
    print(f"Window summary written to: {window_path}")
    print(f"Figures written to: {figures_dir}")


if __name__ == "__main__":
    main()
