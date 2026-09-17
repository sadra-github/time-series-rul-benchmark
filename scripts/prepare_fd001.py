"""Prepare a local C-MAPSS FD001 dataset for benchmark experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile

REQUIRED_FILES = (
    "train_FD001.txt",
    "test_FD001.txt",
    "RUL_FD001.txt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare C-MAPSS FD001 files under data/raw/FD001."
    )
    parser.add_argument(
        "source",
        help="Directory containing the FD001 files or a ZIP archive containing them.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw/FD001",
        help="Destination directory for the three FD001 files.",
    )
    return parser.parse_args()


def find_files(source: Path) -> dict[str, Path]:
    if source.is_dir():
        candidates = {path.name: path for path in source.rglob("*") if path.is_file()}
    elif source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            names = {Path(name).name: name for name in archive.namelist()}
            missing = [name for name in REQUIRED_FILES if name not in names]
            if missing:
                raise FileNotFoundError(
                    "Missing required FD001 files in ZIP: " + ", ".join(missing)
                )

            temp_dir = source.parent / ".cmapss_fd001_extract"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir(parents=True)
            archive.extractall(temp_dir)
            candidates = {
                path.name: path for path in temp_dir.rglob("*") if path.is_file()
            }
    else:
        raise ValueError("Source must be a directory or a ZIP archive.")

    missing = [name for name in REQUIRED_FILES if name not in candidates]
    if missing:
        raise FileNotFoundError(
            "Missing required FD001 files: " + ", ".join(missing)
        )
    return {name: candidates[name] for name in REQUIRED_FILES}


def validate_text_file(path: Path, expected_columns: int) -> None:
    with path.open("r", encoding="utf-8") as handle:
        rows = [line.split() for line in handle if line.strip()]

    if not rows:
        raise ValueError(f"File is empty: {path}")

    bad_rows = [index + 1 for index, row in enumerate(rows) if len(row) != expected_columns]
    if bad_rows:
        preview = ", ".join(map(str, bad_rows[:5]))
        raise ValueError(
            f"Unexpected column count in {path}; bad row(s): {preview}"
        )


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    files = find_files(source)
    validate_text_file(files["train_FD001.txt"], expected_columns=26)
    validate_text_file(files["test_FD001.txt"], expected_columns=26)
    validate_text_file(files["RUL_FD001.txt"], expected_columns=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        shutil.copy2(path, output_dir / name)

    print("C-MAPSS FD001 dataset prepared successfully.")
    print(f"Source: {source}")
    print(f"Destination: {output_dir}")
    for name in REQUIRED_FILES:
        print(f"  {name}")


if __name__ == "__main__":
    main()
