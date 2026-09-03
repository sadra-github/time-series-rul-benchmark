import numpy as np
import pandas as pd
import pytest

from src.preprocessing.scaling import TrainingOnlyScaler


def test_scaler_fits_only_on_training_data():
    train = pd.DataFrame({"sensor_1": [1.0, 2.0, 3.0]})
    test = pd.DataFrame({"sensor_1": [100.0, 101.0]})

    scaler = TrainingOnlyScaler(["sensor_1"])
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    assert np.isclose(train_scaled["sensor_1"].mean(), 0.0)
    assert test_scaled["sensor_1"].mean() > 50.0


def test_transform_requires_fit():
    scaler = TrainingOnlyScaler(["sensor_1"])
    df = pd.DataFrame({"sensor_1": [1.0, 2.0]})

    with pytest.raises(RuntimeError, match="fitted on training data first"):
        scaler.transform(df)


def test_scaler_rejects_missing_features():
    scaler = TrainingOnlyScaler(["sensor_1", "sensor_2"])
    df = pd.DataFrame({"sensor_1": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Missing feature columns"):
        scaler.fit(df)
