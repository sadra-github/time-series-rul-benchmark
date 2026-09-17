import numpy as np
import pandas as pd
import pytest

from src.evaluation.runner import evaluate_model, evaluate_model_by_unit


class DummyModel:
    def predict(self, X):
        return X[:, -1, 0]


def test_evaluate_model_uses_common_metrics():
    X_test = np.array(
        [
            [[1.0], [2.0]],
            [[2.0], [4.0]],
            [[3.0], [6.0]],
        ]
    )
    y_test = np.array([2.0, 3.0, 5.0])

    metrics = evaluate_model(DummyModel(), X_test, y_test)

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert np.isfinite(list(metrics.values())).all()


def test_evaluate_model_rejects_invalid_input_shape():
    with pytest.raises(ValueError, match="shape"):
        evaluate_model(DummyModel(), np.ones((3, 2)), np.ones(3))


def test_evaluate_model_rejects_empty_test_data():
    with pytest.raises(ValueError, match="must not be empty"):
        evaluate_model(DummyModel(), np.empty((0, 2, 1)), np.empty(0))


def test_evaluate_model_rejects_mismatched_target_length():
    X_test = np.ones((3, 2, 1))
    with pytest.raises(ValueError, match="same length"):
        evaluate_model(DummyModel(), X_test, np.ones(2))


def test_evaluate_model_by_unit_returns_one_row_per_unit():
    X_test = np.array(
        [
            [[1.0], [2.0]],
            [[2.0], [4.0]],
            [[3.0], [6.0]],
            [[4.0], [8.0]],
        ]
    )
    y_test = np.array([2.0, 3.0, 6.0, 7.0])
    unit_ids = pd.Series([1, 1, 2, 2])

    results = evaluate_model_by_unit(DummyModel(), X_test, y_test, unit_ids)

    assert list(results["unit_id"]) == [1, 2]
    assert set(results.columns) == {"unit_id", "mae", "rmse", "r2", "n_samples"}
    assert (results["n_samples"] == 2).all()
