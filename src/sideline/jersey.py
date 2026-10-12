"""Jersey number evidence per tracked box: legibility classifier + PARSeq reader.

Models: Koshkina & Elder's SoccerNet legibility ResNet34 and SoccerNet
fine-tuned PARSeq (non-commercial license), exported to ONNX once with
scripts/export_jersey_onnx.py and run here as TensorRT engines, built on first
use next to the ONNX files.

For each player or goalkeeper box at least `min_h` pixels tall:
  leg  = P(number legible) on the whole-player crop
  P[n] = number distribution (0-99) from PARSeq's first three tokens on the
         torso crop, which spans the shoulder and hip keypoints when the pose
         is confident and otherwise 10-55% of the box height
The evidence (P, 1 - leg) is gated: reads below the legibility floor or the
read-confidence threshold count as no evidence.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def iou_matrix(A, B):
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (a[:, None] + b[None, :] - inter + 1e-9)


def pose_torso(kp, pad=5, min_conf=0.3):
    """Shoulders-to-hips box from COCO keypoints (5,6 shoulders; 11,12 hips), or None."""
    pts = kp[[5, 6, 11, 12]]
    if (pts[:, 2] >= min_conf).sum() < 3 or pts[[0, 1], 2].max() < min_conf:
        return None
    v = pts[pts[:, 2] >= min_conf]
    x1, y1 = v[:, 0].min() - pad, v[:, 1].min() - pad
    x2, y2 = v[:, 0].max() + pad, v[:, 1].max() + pad
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    return x1, y1, x2, y2


def number_dist(logits):
    """(N, steps, C) PARSeq logits -> (N, 100) distribution over jersey numbers."""
    p = logits[:, :3, :11].float().softmax(-1).cpu().numpy()     # (N,3,{end,0..9})
    end, dig = 0, slice(1, 11)
    out = np.zeros((len(p), 100), np.float32)
    out[:, :10] = p[:, 0, dig] * p[:, 1, end:end + 1]
    two = p[:, 0, dig][:, :, None] * p[:, 1, dig][:, None, :] * p[:, 2, end][:, None, None]
    out[:, 10:] = two.reshape(len(p), 100)[:, 10:]               # a leading zero is not a number
    return out / np.maximum(out.sum(1, keepdims=True), 1e-9)


def _engine(onnx_path, fp16):
    from .trt import TrtModel, build_engine
    eng = Path(onnx_path).with_suffix(".fp16.engine" if fp16 else ".fp32.engine")
    if not eng.exists():
        print(f"[jersey] building TensorRT engine {eng.name} (first run only)", flush=True)
        build_engine(onnx_path, eng, fp16=fp16)
    return TrtModel(eng)


class JerseyReader:
    def __init__(self, model_dir="models/jersey", min_h=40, leg_thr=0.3, read_conf=0.97,
                 parseq_fp16=False, device="cuda"):
        d = Path(model_dir)
        self.leg = _engine(d / "legibility.onnx", fp16=True)
        self.stm = _engine(d / "parseq.onnx", fp16=parseq_fp16)
        self.min_h, self.leg_thr, self.read_conf = min_h, leg_thr, read_conf
        self.device = torch.device(device)
        self.mean, self.std = MEAN.to(self.device), STD.to(self.device)

    def _batch(self, imgs, hw):
        t = torch.from_numpy(np.stack([cv2.resize(i, (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC)
                                       for i in imgs])).to(self.device, non_blocking=True)
        return t.permute(0, 3, 1, 2).float().div_(255)

    @torch.no_grad()
    def read(self, frame, boxes, classes, pose_boxes=None, pose_kps=None):
        """boxes: (N,4) xyxy; classes: detector class names.
        -> list of (P[100], Pnone) or None per box (None = not read)."""
        H, W = frame.shape[:2]
        out = [None] * len(boxes)
        if len(boxes) == 0:
            return out
        kp_for = [None] * len(boxes)
        if pose_boxes is not None and len(pose_boxes):
            iou = iou_matrix(np.asarray(boxes, np.float32), np.asarray(pose_boxes, np.float32))
            for i in range(len(boxes)):
                j = int(iou[i].argmax())
                if iou[i, j] >= 0.5:
                    kp_for[i] = pose_kps[j]
        idx, whole, torso = [], [], []
        rgb = None
        for i, (b, c) in enumerate(zip(boxes, classes)):
            x1, y1, x2, y2 = map(float, b)
            h = y2 - y1
            if c not in ("player", "goalkeeper") or h < self.min_h:
                continue
            if rgb is None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pt = pose_torso(kp_for[i]) if kp_for[i] is not None else None
            if pt is not None:
                tx1, ty1, tx2, ty2 = pt
            else:
                tx1, tx2, ty1, ty2 = x1, x2, y1 + 0.10 * h, y1 + 0.55 * h
            wc = rgb[int(max(0, y1)):int(min(H, y2)), int(max(0, x1)):int(min(W, x2))]
            tc = rgb[int(max(0, ty1)):int(min(H, ty2)), int(max(0, tx1)):int(min(W, tx2))]
            if wc.size == 0 or tc.size == 0:
                continue
            idx.append(i)
            whole.append(wc)
            torso.append(tc)
        if not idx:
            return out
        legs = self.leg((self._batch(whole, (256, 256)) - self.mean) / self.std).view(-1).cpu().numpy()
        P = np.zeros((len(idx), 100), np.float32)
        run = np.flatnonzero(legs >= self.leg_thr)
        if len(run):
            x = (self._batch([torso[j] for j in run], (32, 128)) - 0.5) / 0.5
            P[run] = number_dist(self.stm(x))
        keep = (legs >= self.leg_thr) & (P.max(1) >= self.read_conf)
        for k, i in enumerate(idx):
            out[i] = (P[k], float(1.0 - legs[k])) if keep[k] else (np.zeros(100, np.float32), 1.0)
        return out
