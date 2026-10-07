"""Unified detector interface.

Backends:
  - "yolo26s" / "yolo26m" / "yolo26n": ultralytics (AGPL-3.0, fine for portfolio)
  - "rtdetr": RT-DETR via ultralytics model hub (rtdetr-l)
  - "mobilevit_s" / "tinyvit_5m": efficient-ViT backbones (Week 2; wired here,
    implemented in src/models/efficient_vit.py)
  - "trt:<engine_path>": TensorRT engine built by scripts/build_trt.py (GPU only)

All backends return the same format so tracking/benchmark code never changes
between arms of the tradeoff study.
"""

import numpy as np


class Detector:
    def __init__(self, name="yolo26s", imgsz=640, conf=0.25, device="cpu",
                 classes=None, iou=None, weights=None):
        self.classes = classes  # e.g. [0] = person only (SportsMOT GT is players)
        self.iou = iou
        self.weights = None if weights in (None, "pretrained") else weights
        self.name = name
        self.imgsz = imgsz
        self.conf = conf
        self.device = device
        self._model = None
        self._load()

    def _load(self):
        if self.name.startswith("trt:"):
            raise RuntimeError(
                "TensorRT engines need scripts/build_trt.py on the GPU machine. "
                "See docs/GPU_RUN_PLAN.md Session B."
            )
        if self.name in ("mobilevit_s", "tinyvit_5m"):
            from .efficient_vit import EfficientViTDetector  # Week 2 module
            self._model = EfficientViTDetector(self.name, self.device)
            return
        from ultralytics import YOLO
        model_id = {"yolo26n": "yolo26n.pt", "yolo26s": "yolo26s.pt",
                    "yolo26m": "yolo26m.pt", "rtdetr": "rtdetr-l.pt"}[self.name]
        self._model = YOLO(self.weights or model_id)
        # ultralytics handles device placement per-predict call

    def infer(self, frame):
        """frame: HxWx3 BGR numpy array.
        Returns list of (x1, y1, x2, y2, score, cls)."""
        if self.name in ("mobilevit_s", "tinyvit_5m"):
            return self._model.infer(frame)
        res = self._model.predict(frame, imgsz=self.imgsz, conf=self.conf,
                                  device=self.device, classes=self.classes,
                                  verbose=False,
                                  **({"iou": self.iou} if self.iou else {}))[0]
        out = []
        if res.boxes is not None and len(res.boxes):
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            clses = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), s, c in zip(boxes, scores, clses):
                out.append((float(x1), float(y1), float(x2), float(y2),
                            float(s), int(c)))
        return out

    def to_onnx(self, path, opset=17):
        """Export for the TensorRT path. CPU-OK; engine build needs GPU."""
        if self.name in ("mobilevit_s", "tinyvit_5m"):
            return self._model.to_onnx(path, opset=opset)
        import shutil
        from pathlib import Path
        out = self._model.export(format="onnx", imgsz=self.imgsz, opset=opset)
        src = Path(out) if out and Path(out).exists() else None
        if src is None:  # fallbacks across ultralytics versions
            ckpt = getattr(self._model, "ckpt_path", None)
            cand = Path(ckpt).with_suffix(".onnx") if ckpt else None
            if cand is not None and cand.exists():
                src = cand
        if src is None:
            cands = sorted(Path(".").glob("*.onnx"),
                           key=lambda p: p.stat().st_mtime)
            src = cands[-1] if cands else None
        if src is None:
            raise RuntimeError("ONNX export produced no .onnx file")
        shutil.move(str(src), path)
        return path
