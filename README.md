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

Raw dataset files are not included in this repository. Dataset acquisition and expected file structure will be documented separately.

## Methodology

The planned workflow is:

```text
Raw multivariate time series
        |
        v
Temporal preprocessing
        |
        v
Leakage-aware window construction
        |
        +----------------------+
        |                      |
        v                      v
 Engineered representation   Sequence representation
        |                      |
        v                      v
 Classical ML              Neural models
        |                      |
        +----------+-----------+
                   |
                   v
             RUL prediction
                   |
                   v
          Evaluation and analysis
```

## Models

The initial benchmark includes:

- Naive baseline
- Linear Regression
- Random Forest
- Gradient Boosting
- Multilayer Perceptron (MLP)
- Long Short-Term Memory (LSTM)

Additional models will only be introduced when justified by the experimental design.

## Experimental Protocol

The benchmark is designed to preserve temporal structure throughout the experiment.

Key controls include:

- Unit-aware train/test separation.
- Chronological ordering of observations.
- No use of future observations when constructing past inputs.
- Fitting preprocessing operations using training data only.
- Fixed and explicitly recorded random seeds.
- Consistent evaluation procedures across model families.

## Evaluation

The initial metrics are:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)

Additional prognostics-specific metrics may be included when their use is methodologically justified.

## Reproducibility

Experiments will be controlled through explicit configurations covering preprocessing, temporal windows, model settings, random seeds, and evaluation procedures.

The repository is intended to support repeatable experiments rather than a single model demonstration.

## Project Structure

```text
.
├── data/
├── configs/
├── notebooks/
├── src/
│   ├── data/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   └── evaluation/
├── tests/
├── results/
└── docs/
```

## Limitations

The initial scope is limited to NASA C-MAPSS FD001 and the model families defined above. Broader datasets and additional architectures may be considered in future versions.

## License

A license will be selected before the first public release of the complete implementation.

## Status

Project initialization. The experimental specification and implementation are being developed incrementally.
