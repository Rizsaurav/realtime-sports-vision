"""Detector wrappers. One interface, many backends:
PyTorch (dev/CPU smoke) -> ONNX -> TensorRT (GPU sessions only).
"""

from .detector import Detector

__all__ = ["Detector"]
