"""Compare temporal window sizes on the development validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.window_comparison import compare_window_sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RUL temporal window sizes without using the official test set."
    )
    parser.add_argument("--train", type=Path, required=True, help="Path to train_FD001.txt")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[5, 10, 20, 30],
        help="Window sizes to compare.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = compare_window_sizes(
        args.train,
        args.windows,
        validation_fraction=args.validation_fraction,
        validation_seed=args.seed,
        stride=args.stride,
    )

    print(results.to_string(index=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
