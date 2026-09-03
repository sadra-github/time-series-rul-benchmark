"""Baseline and predictive models for the RUL benchmark."""

from .baseline import LastValueBaseline, LinearRULBaseline

__all__ = ["LastValueBaseline", "LinearRULBaseline"]
