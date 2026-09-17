import numpy as np
import pytest

from src.models.baseline import LastValueBaseline, LinearRULBaseline, MeanRULBaseline


def test_mean_rul_baseline_uses_training_target_mean():
    X_train = np.ones((3, 2, 1))
    y_train = np.array([10.0, 20.0, 30.0])
    X_test = np.ones((2, 2, 1))

    model = MeanRULBaseline().fit(X_train, y_train)

    np.testing.assert_array_equal(model.predict(X_test), [20.0, 20.0])


def test_mean_rul_baseline_rejects_empty_training_data():
    X = np.empty((0, 2, 1))
    y = np.empty(0)

    with pytest.raises(ValueError, match="must not be empty"):
        MeanRULBaseline().fit(X, y)


def test_last_value_baseline_uses_final_timestep_and_feature():
    X = np.array(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
        ]
    )
    y = np.array([20.0, 40.0])

    model = LastValueBaseline().fit(X, y)

    np.testing.assert_array_equal(model.predict(X), y)


def test_linear_baseline_fits_and_predicts():
    X = np.array(
        [
            [[1.0], [2.0]],
            [[2.0], [3.0]],
            [[3.0], [4.0]],
            [[4.0], [5.0]],
        ]
    )
    y = np.array([2.0, 3.0, 4.0, 5.0])

    model = LinearRULBaseline().fit(X, y)
    predictions = model.predict(X)

    np.testing.assert_allclose(predictions, y)


def test_baselines_reject_invalid_shape():
    X = np.ones((4, 2))
    y = np.ones(4)

    with pytest.raises(ValueError, match="shape"):
        MeanRULBaseline().fit(X, y)

    with pytest.raises(ValueError, match="shape"):
        LastValueBaseline().fit(X, y)

    with pytest.raises(ValueError, match="shape"):
        LinearRULBaseline().fit(X, y)
