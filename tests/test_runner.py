import numpy as np

from src.evaluation.runner import evaluate_model


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
