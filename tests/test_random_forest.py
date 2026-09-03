import numpy as np

from src.models.random_forest import RandomForestRULBaseline


def test_random_forest_fit_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(12, 4, 3))
    y = X[:, -1, 0] * 2.0 + X[:, -1, 1]

    model = RandomForestRULBaseline(
        n_estimators=20,
        random_state=42,
        n_jobs=1,
    )
    model.fit(X, y)
    predictions = model.predict(X[:4])

    assert predictions.shape == (4,)
    assert np.isfinite(predictions).all()


def test_random_forest_requires_sequence_input():
    model = RandomForestRULBaseline(n_estimators=5, random_state=42, n_jobs=1)
    X = np.ones((10, 3))
    y = np.ones(10)

    try:
        model.fit(X, y)
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-sequence input.")
