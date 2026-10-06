"""Efficient-ViT detector variants (Week 2 module).

Wraps a lightweight ViT backbone (TinyViT-5M via timm, or MobileViT-S via
apple/ml-cvnets) with a detection head, exposing the same interface
as Detector. The backbone's intermediate feature maps are exposed so
src/streaming/cache.py can cache them on keyframes (the novel bit).

Backbone status:
  - TinyViT-5M (timm): wired. Loads ImageNet-1k pretrained weights on CPU
    via scripts/download_weights.py (see feat/timm-tinyvit).
  - MobileViT-S (ml-cvnets): pending (feat/mobilevit).
Detection head: pending (feat/detection-head).
"""

import os
from pathlib import Path

import numpy as np
import torch
import timm

TIMM_MODEL = "tiny_vit_5m_224"
TIMM_PRETRAINED_TAG = "tiny_vit_5m_224.in1k"
WEIGHTS_URL = (
    "https://huggingface.co/timm/tiny_vit_5m_224.in1k"
    "/resolve/main/model.safetensors"
)
DEFAULT_WEIGHTS_PATH = (
    Path(os.environ.get("TINYVIT_WEIGHTS", "weights/tiny_vit_5m_224.in1k.safetensors"))
)


def _adapt_checkpoint_keys(model, state_dict):
    """Remap a timm classification checkpoint to the features_only naming.

    timm's FeatureListNet (used with features_only=True) renames the
    submodule ``stages`` -> ``stages_`` to avoid clashing with the original
    forward pass, so checkpoint keys ``stages.<i>...`` must become
    ``stages_<i>...``. The classification ``head.*`` keys are dropped.
    """
    adapted = {}
    for k, v in state_dict.items():
        if k.startswith("head."):
            continue
        if k.startswith("stages."):
            k = "stages_" + k[len("stages."):]
        adapted[k] = v
    missing, unexpected = model.load_state_dict(adapted, strict=False)
    # BatchNorm running stats are stored as plain buffers without
    # num_batches_tracked in this build; everything else must match.
    bad = [k for k in missing if "num_batches_tracked" not in k]
    if bad:
        raise RuntimeError(f"{len(bad)} backbone weights missing after remap, "
                           f"e.g. {bad[:5]}")
    return model


class TinyViTBackbone:
    """TinyViT-5M backbone on CPU with timm, features_only.

    Emits 4 hierarchical feature maps (strides 4/8/16/32) that the
    FrameFeatureCache can hold on keyframes and the detection head
    (feat/detection-head) will fuse.
    """

    def __init__(self, model_name=TIMM_MODEL,
                 weights_path=DEFAULT_WEIGHTS_PATH, device="cpu"):
        self.device = torch.device(device)
        self.model_name = model_name
        self.model = timm.create_model(
            model_name, pretrained=False, features_only=True)
        self._load_weights(weights_path)
        self.model.eval().to(self.device)
        cfg = self.model.default_cfg
        self._mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
        self._std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)

    def _load_weights(self, weights_path):
        if weights_path is not None and Path(weights_path).exists():
            from safetensors.torch import load_file
            _adapt_checkpoint_keys(
                self.model, load_file(str(weights_path), device="cpu"))
            return
        # No local weights: try timm's own pretrained path (needs working
        # HF hub access). Fails on machines with broken proxy env; then
        # tell the user to run scripts/download_weights.py.
        try:
            pretrained_model = timm.create_model(
                self.model_name, pretrained=TIMM_PRETRAINED_TAG,
                features_only=True)
            self.model.load_state_dict(
                pretrained_model.state_dict(), strict=False)
            del pretrained_model
        except Exception as e:  # noqa: BLE001 - we re-raise with context
            raise RuntimeError(
                f"TinyViT weights not found at {weights_path} and timm's "
                f"pretrained download failed ({e}). Run "
                f"`python scripts/download_weights.py` first.") from e

    @property
    def feature_channels(self):
        return self.model.feature_info.channels()

    @property
    def strides(self):
        return self.model.feature_info.reduction()

    def feature_map_shapes(self, imgsz=640):
        """[(C, H, W), ...] per stage for an imgsz x imgsz input."""
        return [(c, imgsz // s, imgsz // s)
                for c, s in zip(self.feature_channels, self.strides)]

    @property
    def num_params(self):
        return sum(p.numel() for p in self.model.parameters())

    def backbone_features(self, frame):
        """frame: HxWx3 BGR numpy array (uint8).

        Returns a list of 4 CPU torch tensors (N=1, C, H, W), one per
        stage — the format FrameFeatureCache stores on keyframes.
        """
        if not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be an HxWx3 numpy array")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
        x = (torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
             / 255.0)
        x = (x - self._mean) / self._std
        with torch.inference_mode():
            feats = self.model(x.to(self.device))
        return [f.cpu() for f in feats]


class EfficientViTDetector:
    BACKBONES = ("mobilevit_s", "tinyvit_5m")

    def __init__(self, backbone, device="cpu"):
        assert backbone in self.BACKBONES, backbone
        self.backbone_name = backbone
        self.device = device
        self.backbone = None
        if backbone == "tinyvit_5m":
            self.backbone = TinyViTBackbone(device=device)
        # mobilevit_s: pending (feat/mobilevit)
        # TODO Week 2 (feat/detection-head): attach lightweight head on the
        # backbone features and implement infer().

    def infer(self, frame):
        raise NotImplementedError(
            "detection head pending (feat/detection-head)")

    def backbone_features(self, frame):
        """Return intermediate feature maps for FrameFeatureCache."""
        if self.backbone is None:
            raise NotImplementedError(
                f"{self.backbone_name}: backbone not wired yet")
        return self.backbone.backbone_features(frame)

    def to_onnx(self, path, opset=17):
        raise NotImplementedError("feat/detection-head")
