# Stage 8: Temporal Window Comparison

## Status

Stage 8 validation experiment completed for NASA C-MAPSS FD001.

Official test data were not used for window selection.

## Research question

How does temporal representation, expressed through fixed-length window size, affect RUL prediction performance?

## Protocol

- Dataset: NASA C-MAPSS FD001
- Development partition: `train_FD001.txt` only
- Unit-level split: 80 training units / 20 validation units
- Validation fraction: 0.20
- Validation seed: 42
- Window sizes: 5, 10, 20, 30, 50 observations
- Stride: 1
- Target alignment: final observation in each window
- Feature scaling: standard scaling fitted on development-training units only
- Models: Mean RUL, Linear Regression, Random Forest
- Primary selection metric: validation MAE
- Complementary metrics: RMSE and R2
- Official test set: not accessed
- Official `RUL_FD001.txt`: not accessed

Windows are constructed independently within each unit. Consecutive windows overlap because stride is fixed at 1. This overlap occurs only within a partition; training and validation units remain disjoint.

## Validation results

| Window | Model | MAE | RMSE | R2 |
|---:|---|---:|---:|---:|
| 5 | Linear Regression | 28.790396 | 36.605870 | 0.679076 |
| 5 | Mean RUL | 54.460967 | 64.682864 | -0.002028 |
| 5 | Random Forest | 23.853534 | 33.030361 | 0.738707 |
| 10 | Linear Regression | 27.639461 | 35.214902 | 0.690801 |
| 10 | Mean RUL | 53.338511 | 63.397753 | -0.002151 |
| 10 | Random Forest | 23.241802 | 32.101969 | 0.743050 |
| 20 | Linear Regression | 25.098168 | 31.676054 | 0.728338 |
| 20 | Mean RUL | 51.119079 | 60.847575 | -0.002429 |
| 20 | Random Forest | 21.852327 | 30.179952 | 0.753394 |
| 30 | Linear Regression | 23.185002 | 29.206968 | 0.748559 |
| 30 | Mean RUL | 48.934785 | 58.326648 | -0.002760 |
| 30 | Random Forest | 20.381818 | 28.116547 | 0.766983 |
| 50 | Linear Regression | 20.720773 | 25.980119 | 0.762317 |
| 50 | Mean RUL | 44.708493 | 53.386163 | -0.003630 |
| 50 | Random Forest | 17.444854 | 24.022515 | 0.796786 |

## Interpretation at this checkpoint

Within the predefined validation protocol, validation error decreased as the temporal window increased from 5 to 50 observations for both Linear Regression and Random Forest. For Random Forest, validation MAE decreased from 23.853534 at window 5 to 17.444854 at window 50, while R2 increased from 0.738707 to 0.796786.

This result supports the benchmark research question that temporal representation affects RUL prediction performance. It does not establish that window 50 is universally optimal, because only the predefined range was evaluated and official test performance has not yet been assessed.

The number of windows decreases with larger window sizes, while validation performance improves. Therefore the observed improvement cannot be attributed simply to having more training windows.

## Next stage

Stage 9 will examine generalization across units before the final official FD001 test evaluation. Window selection must remain independent of the official test partition.
