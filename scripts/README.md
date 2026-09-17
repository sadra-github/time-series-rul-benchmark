# Benchmark Scripts

The scripts in this directory provide reproducible entry points for running benchmark experiments and generating result tables and figures.

Scripts should read the public dataset from a user-supplied path rather than storing raw C-MAPSS files in the repository.

Result-generating scripts should keep dataset partitions, preprocessing, window settings, model parameters, and random seeds explicit.

## FD001 baseline benchmark

From the repository root, run:

```bash
python scripts/run_fd001_benchmark.py --train PATH_TO/train_FD001.txt --test PATH_TO/test_FD001.txt --rul PATH_TO/RUL_FD001.txt
```

The script uses the official FD001 train/test organization, a training-only standard scaler, a default temporal window of 20 observations, and the baseline models defined by the benchmark pipeline.

Results are written to `results/tables/` and `results/figures/`.

Raw C-MAPSS data are not stored in the repository.
