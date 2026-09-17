"""Run the official C-MAPSS FD001 baseline benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_official_test_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline RUL models on the official C-MAPSS FD001 test set."
    )
    parser.add_argument("--train", type=Path, required=True, help="Path to train_FD001.txt")
    parser.add_argument("--test", type=Path, required=True, help="Path to test_FD001.txt")
    parser.add_argument("--rul", type=Path, required=True, help="Path to RUL_FD001.txt")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_official_test_benchmark(
        args.train,
        args.test,
        args.rul,
        window_size=args.window_size,
        stride=args.stride,
    )

    print(results.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
