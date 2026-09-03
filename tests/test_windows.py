import pandas as pd
import pytest

from src.preprocessing.windows import make_sequence_windows


def make_sample() -> pd.DataFrame:
    rows = []
    for unit_id in [1, 2]:
        for cycle in range(1, 6):
            rows.append(
                {
                    "unit_id": unit_id,
                    "cycle": cycle,
                    "sensor_1": float(unit_id * 10 + cycle),
                    "rul": float(5 - cycle),
                }
            )
    return pd.DataFrame(rows)


def test_windows_have_expected_shape_and_target_alignment():
    X, y, metadata = make_sequence_windows(
        make_sample(), ["sensor_1"], window_size=3
    )

    assert X.shape == (6, 3, 1)
    assert y.tolist() == [2.0, 1.0, 0.0, 2.0, 1.0, 0.0]
    assert metadata["start_cycle"].tolist() == [1, 2, 3, 1, 2, 3]
    assert metadata["end_cycle"].tolist() == [3, 4, 5, 3, 4, 5]


def test_windows_never_cross_unit_boundaries():
    X, y, metadata = make_sequence_windows(
        make_sample(), ["sensor_1"], window_size=4
    )

    assert X.shape == (4, 4, 1)
    assert metadata["unit_id"].tolist() == [1, 1, 2, 2]
    assert y.tolist() == [2.0, 1.0, 2.0, 1.0]


def test_short_trajectories_produce_no_windows():
    df = make_sample().query("unit_id == 1 and cycle <= 2").copy()

    with pytest.raises(ValueError, match="No complete windows"):
        make_sequence_windows(df, ["sensor_1"], window_size=3)


def test_invalid_window_parameters_are_rejected():
    df = make_sample()

    with pytest.raises(ValueError, match="window_size must be positive"):
        make_sequence_windows(df, ["sensor_1"], window_size=0)

    with pytest.raises(ValueError, match="stride must be positive"):
        make_sequence_windows(df, ["sensor_1"], window_size=3, stride=0)
