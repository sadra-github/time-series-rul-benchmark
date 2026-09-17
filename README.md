# Time-Series RUL Benchmark

A reproducible benchmark for remaining useful life prediction from multivariate time-series data.

## Research Question

How does temporal representation affect the performance and generalization of machine-learning models for remaining useful life prediction?

## Objectives

- Establish leakage-aware experimental protocols for temporal prediction.
- Compare classical machine-learning and neural-network approaches.
- Evaluate the effect of temporal window construction on predictive performance.
- Analyze prediction errors and model generalization.
- Provide a reproducible implementation and evaluation workflow.

## Dataset

This project uses the publicly available NASA C-MAPSS dataset, initially restricted to the FD001 subset.

Raw dataset files are not included in this repository. The expected inputs are `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt`.

## Methodology

The current implementation follows this workflow:

```text
C-MAPSS FD001
      |
      v
Schema and temporal validation
      |
      v
Training-only preprocessing
      |
      v
Fixed temporal windows
      |
      +--------------------------+
      |                          |
      v                          v
Development validation      Official test
      |                          |
      v                          v
Baseline comparison        Final evaluation
      |                          |
      +------------+-------------+
                   |
                   v
          Error and generalization analysis
```

## Current Models

The implemented baseline set is:

- Mean RUL baseline
- Linear Regression on flattened temporal windows
- Random Forest on flattened temporal windows
- Last-value diagnostic baseline

The last-value model is retained only as a diagnostic reference because the last value of an arbitrary sensor is not intrinsically an RUL estimate.

Neural models are not added until the data, temporal-window, and evaluation pipeline has been validated with simple baselines.

## Experimental Protocol

The benchmark preserves temporal structure throughout the experiment.

Key controls include:

- Unit-level development train/validation separation.
- Separate use of the official C-MAPSS test partition for final evaluation.
- Chronological ordering within each unit.
- No future observations in an input window.
- Preprocessing fitted on training data only.
- Explicit window size and stride.
- Common evaluation metrics across model families.
- Fixed random seeds where stochastic components are used.

The C-MAPSS train and test files use local unit identifiers. Therefore, equal numeric unit IDs in the two files do not represent shared trajectories.

## Temporal Window Comparison

Window-size experiments are performed on the development validation partition. The official test set is not used to select a window size. This prevents the final test result from becoming part of model or protocol selection.

The comparison utility is available through `scripts/compare_windows.py`.

## Evaluation

The primary metrics are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

The evaluation layer validates array shape, sample counts, and finite values before calculating metrics.

Unit-level evaluation is also available for examining variation across previously unseen test trajectories.

## Running the Benchmark

After downloading the official C-MAPSS FD001 files, run the final baseline benchmark from the repository root:

```text
python scripts/run_benchmark.py --train path/to/train_FD001.txt --test path/to/test_FD001.txt --rul path/to/RUL_FD001.txt --window-size 20 --output results/fd001_baselines.csv
```

To compare temporal window sizes without using the official test set:

```text
python scripts/compare_windows.py --train path/to/train_FD001.txt --windows 5 10 20 30 --output results/window_comparison.csv
```

## Reproducibility

The benchmark configuration records the dataset subset, split strategy, preprocessing rule, temporal window definition, evaluation metrics, and seed. Model parameters are explicit in the model constructors.

The repository includes automated tests for data validation, RUL construction, temporal windows, scaling, models, evaluation, and the end-to-end benchmark pipeline.

## Project Structure

```text
.
├── data/
├── configs/
├── docs/
├── scripts/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── models/
│   ├── evaluation/
│   └── experiments/
└── tests/
```

## Documentation

- `docs/benchmark_protocol.md`: data, labels, preprocessing, temporal windows, models, and evaluation protocol.
- `docs/generalization.md`: unit-level generalization analysis.
- `configs/benchmark.yaml`: experiment configuration.

## Scope and Limitations

Version 1 is limited to NASA C-MAPSS FD001. The benchmark currently focuses on controlled classical baselines and temporal representation. Neural architectures, broader datasets, and additional prognostics metrics are reserved for later experimental stages.

## Status

The leakage-aware data pipeline, temporal preprocessing, baseline models, evaluation layer, official FD001 benchmark runner, and validation-only temporal-window comparison are implemented. Results are generated locally from the user's copy of the public C-MAPSS dataset.
