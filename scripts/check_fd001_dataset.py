"""Validate the local NASA C-MAPSS FD001 dataset before benchmarking."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.loader import load_cmapss_txt
from src.data.validation import validate_cmapss_schema, validate_temporal_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run structural and temporal QC on C-MAPSS FD001."
    )
    parser.add_argument("--train", required=True, help="Path to train_FD001")
    parser.add_argument("--test", required=True, help="Path to test_FD001")
    parser.add_argument("--rul", required=True, help="Path to RUL_FD001")
    return parser.parse_args()


def audit_partition(name: str, frame: pd.DataFrame) -> dict[str, object]:
    validate_cmapss_schema(frame)
    validate_temporal_order(frame)

    return {
        "partition": name,
        "units": int(frame["unit_id"].nunique()),
        "observations": int(len(frame)),
        "min_cycle": int(frame["cycle"].min()),
        "max_cycle": int(frame["cycle"].max()),
        "columns": int(frame.shape[1]),
        "missing_values": int(frame.isna().sum().sum()),
        "non_finite_values": int((~frame.select_dtypes(include="number").applymap(pd.notna)).sum().sum()),
    }


def main() -> None:
    args = parse_args()

    train = load_cmapss_txt(args.train)
    test = load_cmapss_txt(args.test)

    train_audit = audit_partition("train", train)
    test_audit = audit_partition("test", test)

    rul = pd.read_csv(args.rul, sep=r"\s+", header=None, names=["rul"])
    if rul.empty:
        raise ValueError("RUL_FD001 is empty.")
    if not pd.api.types.is_numeric_dtype(rul["rul"]):
        raise ValueError("RUL_FD001 contains non-numeric RUL values.")
    if not rul["rul"].notna().all():
        raise ValueError("RUL_FD001 contains missing RUL values.")

    test_units = test["unit_id"].nunique()
    if len(rul) != test_units:
        raise ValueError(
            f"RUL length ({len(rul)}) does not match test units ({test_units})."
        )

    if list(train.columns) != list(test.columns):
        raise ValueError("Train and test schemas do not match.")

    audit = pd.DataFrame([train_audit, test_audit])

    print("C-MAPSS FD001 Dataset QC")
    print("=" * 28)
    print(audit.to_string(index=False))
    print()
    print(f"RUL reference rows: {len(rul)}")
    print(f"RUL minimum: {rul['rul'].min():.3f}")
    print(f"RUL maximum: {rul['rul'].max():.3f}")
    print(f"Train/Test schema match: {list(train.columns) == list(test.columns)}")
    print("Temporal ordering: PASS")
    print("Dataset QC: PASS")


if __name__ == "__main__":
    main()
