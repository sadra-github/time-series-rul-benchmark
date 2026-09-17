# Benchmark Protocol

## Dataset

Version 1 uses NASA C-MAPSS FD001. The raw dataset is not stored in this repository. The user supplies the official `train_FD001.txt`, `test_FD001.txt`, and `RUL_FD001.txt` files locally.

## RUL construction

Training trajectories are treated as complete trajectories, so the training label is defined as the difference between the final observed cycle of a unit and the current cycle.

The C-MAPSS test trajectories are truncated. Their terminal RUL values are supplied separately in `RUL_FD001.txt`. Earlier test labels are reconstructed from the terminal RUL and the cycle distance from the final observed test cycle.

The train and test files use local unit identifiers. The same numeric unit identifier can therefore appear in both files without representing the same physical trajectory.

## Data separation

Development validation is performed at the unit level. A complete unit is assigned to either training or validation, never both.

For the final benchmark, all labeled training units are used for model fitting and the official C-MAPSS test partition is evaluated separately.

## Preprocessing

Feature scaling is fitted on the training partition only. The fitted transformation is then applied to validation or test data. No validation or test observations are used to estimate preprocessing parameters.

## Temporal windows

Fixed-length windows are created independently within each unit. A window never crosses a unit boundary. The target is aligned with the final observation in the window.

Windows are observation-based. Cycle gaps are retained in metadata and are not silently interpreted as regularly sampled missing observations.

## Models

The initial benchmark contains three primary baselines:

- Mean RUL: predicts the mean training RUL for every sample.
- Linear Regression: fits a linear model to flattened temporal windows.
- Random Forest: fits a tree ensemble to flattened temporal windows.

A last-value diagnostic baseline is retained separately because the last value of an arbitrary sensor is not intrinsically an RUL estimate.

## Metrics

All primary models are evaluated with the same three regression metrics:

- MAE
- RMSE
- R2

The evaluation code validates prediction shape, sample count, and finite values before calculating metrics.

## Reproducibility

The benchmark configuration records the dataset subset, split strategy, preprocessing rule, window definition, metrics, and random seed. Model parameters are explicit in the model constructors.

The benchmark runner accepts dataset paths from the command line and can write the resulting metric table to CSV.
