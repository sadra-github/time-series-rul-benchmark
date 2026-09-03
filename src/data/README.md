# Data Module

This module contains dataset ingestion, structural validation, and target-label construction for the benchmark.

## Design Principles

- Loading and validation are separate operations.
- The loader does not scale, impute, filter, or otherwise transform measurements.
- Temporal ordering is checked explicitly.
- Training RUL labels are derived only from complete training trajectories.
- Test-time RUL handling is kept separate because the terminal failure cycle is not directly observed.

## Components

- `loader.py`: C-MAPSS file loading and structural validation.
- `rul.py`: RUL construction for complete training trajectories.

Preprocessing and temporal window construction will be implemented as separate modules so that each experimental transformation remains explicit and testable.
