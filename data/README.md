# Data Directory

This directory defines the data interface used by the benchmark.

## Dataset

The initial benchmark uses NASA C-MAPSS, FD001.

Raw dataset files are intentionally excluded from version control. Place locally acquired raw files under:

```text
data/raw/
```

The repository `.gitignore` prevents raw data from being committed accidentally.

## Data Contract

The preprocessing pipeline will expect engine-level multivariate time-series observations with:

- An engine/unit identifier.
- A cycle/time index within each engine.
- Sensor measurements.
- Operating-condition variables when provided by the source dataset.

The target variable is Remaining Useful Life (RUL), derived from the temporal position of each observation within an engine trajectory.

## Required Properties

Before model training, the ingestion stage will validate that:

1. Engine identifiers are present.
2. Cycle indices are valid and ordered within each engine.
3. Sensor columns are numeric or explicitly transformed into numeric representations.
4. Missing values are identified before preprocessing.
5. No test information is used during training-time preprocessing.
6. RUL labels are generated only from information permitted by the experimental protocol.

## Data Flow

```text
Raw dataset
    |
    v
Data ingestion
    |
    v
Schema validation
    |
    v
Temporal ordering checks
    |
    v
Train/test separation
    |
    v
Training-only preprocessing
    |
    v
Window / feature construction
```

## Reproducibility

Dataset preparation must be deterministic given the same source data and configuration. Dataset assumptions, preprocessing decisions, and generated intermediate representations will be documented as the implementation develops.

## Privacy and Scope

This project uses public benchmark data. No private research data, unpublished experimental results, or proprietary datasets are required by the repository.
