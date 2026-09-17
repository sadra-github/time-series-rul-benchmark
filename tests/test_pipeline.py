from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import prepare_official_test_data, run_official_test_benchmark


COLUMNS = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]


def _write_cmapss_file(path: Path, units: list[int], cycles: int) -> None:
    rows = []
    for unit_id in units:
        for cycle in range(1, cycles + 1):
            values = [unit_id, cycle, 0.1, 0.2, 0.3]
            values.extend(float(unit_id + cycle + sensor) for sensor in range(1, 22))
            rows.append(values)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(
        path,
        sep=" ",
        header=False,
        index=False,
    )


def test_prepare_official_test_data_keeps_train_and_test_separate(tmp_path):
    train_path = tmp_path / "train_FD001.txt"
    test_path = tmp_path / "test_FD001.txt"
    rul_path = tmp_path / "RUL_FD001.txt"

    _write_cmapss_file(train_path, [1, 2], cycles=6)
    _write_cmapss_file(test_path, [1, 2], cycles=5)
    rul_path.write_text("3\n4\n", encoding="utf-8")

    train_windows, test_windows = prepare_official_test_data(
        train_path,
        test_path,
        rul_path,
        window_size=3,
    )

    X_train, y_train, train_meta = train_windows
    X_test, y_test, test_meta = test_windows

    assert X_train.shape[1:] == (3, 24)
    assert X_test.shape[1:] == (3, 24)
    assert len(y_train) == len(train_meta)
    assert len(y_test) == len(test_meta)
    assert np.isfinite(X_train).all()
    assert np.isfinite(X_test).all()
    assert test_meta["unit_id"].nunique() == 2


def test_run_official_test_benchmark_returns_common_metrics(tmp_path):
    train_path = tmp_path / "train_FD001.txt"
    test_path = tmp_path / "test_FD001.txt"
    rul_path = tmp_path / "RUL_FD001.txt"

    _write_cmapss_file(train_path, [1, 2, 3], cycles=8)
    _write_cmapss_file(test_path, [1, 2], cycles=7)
    rul_path.write_text("3\n4\n", encoding="utf-8")

    results = run_official_test_benchmark(
        train_path,
        test_path,
        rul_path,
        window_size=3,
    )

    assert set(results.columns) == {"model", "mae", "rmse", "r2"}
    assert set(results["model"]) == {
        "mean_rul",
        "linear_regression",
        "random_forest",
    }
    assert results[["mae", "rmse", "r2"]].apply(np.isfinite).all().all()
