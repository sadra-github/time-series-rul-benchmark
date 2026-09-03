import pandas as pd
import pytest

from src.data.validation import split_train_validation


def test_validation_split_keeps_units_disjoint():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "cycle": [1, 2] * 5,
        }
    )

    train, validation = split_train_validation(
        df,
        validation_fraction=0.2,
        random_seed=42,
    )

    assert set(train["unit_id"]).isdisjoint(set(validation["unit_id"]))
    assert len(train) + len(validation) == len(df)
    assert len(validation["unit_id"].unique()) == 1


def test_validation_split_is_reproducible():
    df = pd.DataFrame(
        {
            "unit_id": list(range(1, 11)),
            "cycle": [1] * 10,
        }
    )

    train_a, validation_a = split_train_validation(df, random_seed=42)
    train_b, validation_b = split_train_validation(df, random_seed=42)

    assert set(train_a["unit_id"]) == set(train_b["unit_id"])
    assert set(validation_a["unit_id"]) == set(validation_b["unit_id"])


def test_validation_split_rejects_invalid_fraction():
    df = pd.DataFrame({"unit_id": [1, 2], "cycle": [1, 1]})

    with pytest.raises(ValueError, match="between 0 and 1"):
        split_train_validation(df, validation_fraction=1.0)
