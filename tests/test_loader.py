import pandas as pd
import pytest

from src.data.loader import validate_temporal_order


def test_temporal_order_accepts_chronological_rows():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [1, 2, 1, 2],
        }
    )

    validate_temporal_order(df)


def test_temporal_order_rejects_out_of_order_rows():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2, 2],
            "cycle": [2, 1, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_temporal_order(df)


def test_temporal_order_rejects_duplicate_cycles():
    df = pd.DataFrame(
        {
            "unit_id": [1, 1, 2],
            "cycle": [1, 1, 2],
        }
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_temporal_order(df)
