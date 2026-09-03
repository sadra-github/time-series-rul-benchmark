import pandas as pd
import pytest

from src.data.test_rul import add_test_rul, load_cmapss_rul


def test_load_cmapss_rul(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("5\n10\n15\n", encoding="utf-8")

    rul = load_cmapss_rul(path)

    assert rul.index.tolist() == [1, 2, 3]
    assert rul.name == "terminal_rul"
    assert rul.tolist() == [5, 10, 15]


def test_add_test_rul_reconstructs_labels():
    test_df = pd.DataFrame(
        {
            "unit_id": [1, 1, 1, 2, 2],
            "cycle": [8, 9, 10, 6, 7],
            "sensor_1": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    terminal_rul = pd.Series(
        [5, 10], index=pd.Index([1, 2], name="unit_id"), name="terminal_rul"
    )

    result = add_test_rul(test_df, terminal_rul)

    assert result["rul"].tolist() == [7, 6, 5, 11, 10]


def test_add_test_rul_requires_all_units():
    test_df = pd.DataFrame(
        {"unit_id": [1, 2], "cycle": [5, 5]}
    )
    terminal_rul = pd.Series(
        [5], index=pd.Index([1], name="unit_id"), name="terminal_rul"
    )

    with pytest.raises(ValueError, match="Missing terminal RUL"):
        add_test_rul(test_df, terminal_rul)


def test_load_cmapss_rul_rejects_negative_values(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    path.write_text("5\n-1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="non-negative"):
        load_cmapss_rul(path)
