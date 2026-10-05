"""Tracker wrappers: ByteTrack (FoundationVision) and OC-SORT.
Both are motion-only (no learned weights) -> nothing to train, CPU-friendly.
"""

from .tracker import Tracker

__all__ = ["Tracker"]
