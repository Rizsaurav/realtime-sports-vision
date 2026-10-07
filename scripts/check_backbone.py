#!/usr/bin/env python3
"""Acceptance check for feat/timm-tinyvit.

Loads TinyViT-5M pretrained weights on CPU (auto-downloading them if
missing) and prints the backbone's feature-map output shapes, exactly what
the frame cache will store on keyframes.

Usage: python scripts/check_backbone.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from models.efficient_vit import (  # noqa: E402
    TinyViTBackbone,
    EfficientViTDetector,
    DEFAULT_WEIGHTS_PATH,
)


def main():
    if not DEFAULT_WEIGHTS_PATH.exists():
        print("[check] weights missing, downloading...")
        import subprocess
        subprocess.run([sys.executable, "scripts/download_weights.py"],
                       check=True)
    print(f"[check] loading TinyViT-5M pretrained weights on CPU "
          f"({DEFAULT_WEIGHTS_PATH})")
    bb = TinyViTBackbone(device="cpu")
    print(f"[check] params: {bb.num_params / 1e6:.2f}M")

    print("[check] backbone feature-map output shapes (imgsz=640):")
    shapes = bb.feature_map_shapes(640)
    for i, (c, h, w) in enumerate(shapes):
        print(f"  stage {i}: stride {bb.strides[i]:>2}  channels {c:>3}  "
              f"shape (1, {c}, {h}, {w})")

    frame = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)  # BGR
    feats = bb.backbone_features(frame)
    assert [tuple(f.shape) for f in feats] == [(1, *s) for s in shapes], \
        "feature forward shapes mismatch"
    print("[check] forward pass OK: shapes match feature_map_shapes()")

    det = EfficientViTDetector("tinyvit_5m", device="cpu")
    cached = det.backbone_features(frame)
    assert len(cached) == 4, "EfficientViTDetector must expose 4 stage maps"
    print("[check] EfficientViTDetector.backbone_features() OK "
          f"({len(cached)} maps, cache-ready)")
    print("[check] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
