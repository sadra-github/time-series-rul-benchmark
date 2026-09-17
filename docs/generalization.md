# Generalization Analysis

The benchmark treats generalization as performance across previously unseen units, not as a random row-level split.

For development, complete units are separated into training and validation groups. This prevents windows from the same trajectory appearing in both groups.

For the final experiment, the official C-MAPSS test partition remains untouched during model fitting and temporal-window selection. After a protocol is fixed, performance can be reported globally and separately for each test unit.

Unit-level analysis is useful because an aggregate metric can hide large differences between trajectories. The evaluation runner therefore provides `evaluate_model_by_unit`, which reports MAE, RMSE, R2, and the number of evaluated windows for each unit.

A unit-level R2 requires at least two evaluated windows because R2 is undefined for a single observation. Window sizes that produce too few observations for a unit should therefore be documented rather than silently converted into a numerical score.

The repository does not interpret unit-level variation as evidence of a particular physical cause. It is reported as an empirical generalization pattern that can be investigated in later experiments.
