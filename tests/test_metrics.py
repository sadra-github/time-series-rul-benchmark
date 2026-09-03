import numpy as np
import pytest

from src.evaluation.metrics import evaluate_regression


def test_regression_metrics_are_computed_consistently():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 3.0, 2.0])

    metrics = evaluate_regression(y_true, y_pred)

    assert np.isclose(metrics["mae"], 2.0 / 3.0)
    assert np.isclose(metrics["rmse"], np.sqrt(2.0 / 3.0))
    assert np.isclose(metrics["r2"], 0.0)


def test_metrics_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        evaluate_regression(np.array([1.0, 2.0]), np.array([1.0]))


def test_metrics_reject_non_finite_values():
    with pytest.raises(ValueError, match="finite"):
        evaluate_regression(np.array([1.0, np.nan]), np.array([1.0, 2.0]))
