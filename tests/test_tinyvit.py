"""Tests for the TinyViT-5M backbone (feat/timm-tinyvit).

Skipped when weights/ has no local safetensors (e.g. a fresh checkout
before scripts/download_weights.py runs) so the CPU suite stays green.
"""
import numpy as np
import pytest
import torch

from models.efficient_vit import TinyViTBackbone, EfficientViTDetector, \
    DEFAULT_WEIGHTS_PATH

needs_weights = pytest.mark.skipif(
    not DEFAULT_WEIGHTS_PATH.exists(),
    reason="TinyViT weights not downloaded (scripts/download_weights.py)")


@needs_weights
def test_feature_map_shapes():
    bb = TinyViTBackbone(device="cpu")
    assert bb.feature_map_shapes(640) == [
        (64, 160, 160),
        (128, 80, 80),
        (160, 40, 40),
        (320, 20, 20),
    ]


@needs_weights
def test_backbone_features_bgr_frame():
    bb = TinyViTBackbone(device="cpu")
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)  # BGR
    feats = bb.backbone_features(frame)
    assert len(feats) == 4
    assert all(isinstance(f, torch.Tensor) and f.dtype == torch.float32
               for f in feats)
    assert [tuple(f.shape) for f in feats] == [
        (1, 64, 120, 160), (1, 128, 60, 80), (1, 160, 30, 40),
        (1, 320, 15, 20)]


@needs_weights
def test_detector_exposes_cache_ready_features():
    det = EfficientViTDetector("tinyvit_5m", device="cpu")
    frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    feats = det.backbone_features(frame)
    assert len(feats) == 4
    with pytest.raises(NotImplementedError):
        det.infer(frame)  # detection head is feat/detection-head
