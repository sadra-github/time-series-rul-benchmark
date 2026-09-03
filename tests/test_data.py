import pandas as pd
import pytest

from src.data.loader import COLUMNS, validate_cmapss_schema, validate_temporal_order
from src.data.rul import add_training_rul


def make_sample() -> pd.DataFrame:
    rows = []
    for unit_id, cycles in [(1, [1, 2, 3]), (2, [1, 2])]:
        for cycle in cycles:
            rows.append([unit_id, cycle, 0.0, 0.0, 0.0, *([1.0] * 21)])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_schema_accepts_valid_data():
    df = make_sample()
    validate_cmapss_schema(df)
    validate_temporal_order(df)


def test_schema_rejects_wrong_column_count():
    df = make_sample().drop(columns=["sensor_21"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_cmapss_schema(df)


def test_temporal_order_rejects_duplicate_cycle():
    df = make_sample()
    duplicate = df.iloc[[0]].copy()
    df = pd.concat([df, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_temporal_order(df)


def test_training_rul_is_zero_at_terminal_cycle():
    result = add_training_rul(make_sample())

    terminal = result.groupby("unit_id")["rul"].min()
    assert (terminal == 0).all()


def test_training_rul_decreases_with_cycle():
    result = add_training_rul(make_sample())

    unit_1 = result[result["unit_id"] == 1].sort_values("cycle")
    assert unit_1["rul"].tolist() == [2, 1, 0]
