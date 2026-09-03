import numpy as np
import pytest

from src.models.baseline import LastValueBaseline, LinearRULBaseline


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
        LastValueBaseline().fit(X, y)

    with pytest.raises(ValueError, match="shape"):
        LinearRULBaseline().fit(X, y)
