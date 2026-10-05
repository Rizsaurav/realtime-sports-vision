"""Efficient-ViT detector variants (Week 2 module).

Wraps a lightweight ViT backbone (MobileViT-S via apple/ml-cvnets, or
TinyViT-5M via timm) with a detection head, exposing the same interface
as Detector. The backbone's intermediate feature maps are exposed so
src/streaming/cache.py can cache them on keyframes (the novel bit).

Status: scaffold. Week 2 implements the head + feature taps.
Pretrained weights: timm (TinyViT) / ml-cvnets (MobileViT) — CPU-downloadable.
"""


class EfficientViTDetector:
    BACKBONES = ("mobilevit_s", "tinyvit_5m")

    def __init__(self, backbone, device="cpu"):
        assert backbone in self.BACKBONES, backbone
        self.backbone_name = backbone
        self.device = device
        # TODO Week 2:
        #  - load backbone from timm (tinyvit_5m) or ml-cvnets (mobilevit_s)
        #  - attach a lightweight detection head (e.g., anchor-free, YOLO-style)
        #  - expose self.backbone_features(x) for the frame cache
        raise NotImplementedError("Week 2: wire backbone + detection head")

    def infer(self, frame):
        raise NotImplementedError("Week 2")

    def backbone_features(self, frame):
        """Return intermediate feature maps for FrameFeatureCache."""
        raise NotImplementedError("Week 2")

    def to_onnx(self, path, opset=17):
        raise NotImplementedError("Week 2")
