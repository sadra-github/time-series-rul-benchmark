"""Run the baseline C-MAPSS FD001 benchmark and save tables and figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_cmapss_txt
from src.pipeline import run_official_test_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline RUL models on the official C-MAPSS FD001 test set."
    )
    parser.add_argument("--train", required=True, help="Path to train_FD001.txt")
    parser.add_argument("--test", required=True, help="Path to test_FD001.txt")
    parser.add_argument("--rul", required=True, help="Path to RUL_FD001.txt")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for benchmark tables and figures.",
    )
    return parser.parse_args()


def save_dataset_audit(train_path: str, test_path: str, output_dir: Path) -> None:
    train = load_cmapss_txt(train_path)
    test = load_cmapss_txt(test_path)

    audit = pd.DataFrame(
        [
            {
                "partition": "train",
                "units": train["unit_id"].nunique(),
                "observations": len(train),
                "min_cycle": train["cycle"].min(),
                "max_cycle": train["cycle"].max(),
            },
            {
                "partition": "test",
                "units": test["unit_id"].nunique(),
                "observations": len(test),
                "min_cycle": test["cycle"].min(),
                "max_cycle": test["cycle"].max(),
            },
        ]
    )
    audit.to_csv(output_dir / "fd001_dataset_audit.csv", index=False)

    train_lengths = train.groupby("unit_id")["cycle"].max()
    test_lengths = test.groupby("unit_id")["cycle"].max()

    plt.figure(figsize=(8, 5))
    plt.hist(train_lengths, bins=20, alpha=0.7, label="train")
    plt.hist(test_lengths, bins=20, alpha=0.7, label="test")
    plt.xlabel("Observed trajectory length (cycles)")
    plt.ylabel("Number of units")
    plt.title("C-MAPSS FD001 trajectory-length distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "fd001_trajectory_lengths.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    save_dataset_audit(args.train, args.test, tables_dir)

    results = run_official_test_benchmark(
        args.train,
        args.test,
        args.rul,
        window_size=args.window,
        stride=args.stride,
    )
    results.insert(0, "window_size", args.window)
    results.insert(1, "stride", args.stride)
    results.to_csv(tables_dir / "fd001_baseline_results.csv", index=False)

    print("C-MAPSS FD001 baseline benchmark")
    print(f"window_size={args.window}, stride={args.stride}")
    print(results.to_string(index=False))

    for metric in ["mae", "rmse", "r2"]:
        plt.figure(figsize=(8, 5))
        plt.bar(results["model"], results[metric])
        plt.ylabel(metric.upper())
        plt.xlabel("Model")
        plt.title(f"FD001 test {metric.upper()} by baseline model")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(figures_dir / f"fd001_test_{metric}.png", dpi=200)
        plt.close()

    print(f"Results written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
