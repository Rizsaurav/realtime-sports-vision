"""Benchmark harness: the numbers that make this project jaw-dropping.

Every metric lands in results/*.json. Numbers are never re-run.
"""

from .metrics import compute_detection_metrics, compute_tracking_metrics
from .run_study import run_study

__all__ = ["compute_detection_metrics", "compute_tracking_metrics", "run_study"]
