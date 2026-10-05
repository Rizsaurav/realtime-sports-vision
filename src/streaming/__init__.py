"""Streaming pipeline: the real-time loop and the frame feature cache.

This module is the novel contribution. Everything else is building blocks.
"""

from .pipeline import StreamingPipeline
from .cache import FrameFeatureCache

__all__ = ["StreamingPipeline", "FrameFeatureCache"]
