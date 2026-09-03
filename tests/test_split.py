import pandas as pd
import pytest

from src.data.split import split_by_unit


def test_split_has_disjoint_units():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "cycle": [1, 2] * 5,
        }
    )

    train, test = split_by_unit(df, test_fraction=0.2)

    assert set(train["unit_id"]).isdisjoint(set(test["unit_id"]))
    assert len(test["unit_id"].unique()) == 1
    assert len(train) + len(test) == len(df)


def test_split_rejects_invalid_fraction():
    df = pd.DataFrame({"unit_id": [1, 2], "cycle": [1, 1]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_by_unit(df, test_fraction=1.0)


def test_split_requires_multiple_units():
    df = pd.DataFrame({"unit_id": [1, 1], "cycle": [1, 2]})

    with pytest.raises(ValueError, match="At least two units"):
        split_by_unit(df)
