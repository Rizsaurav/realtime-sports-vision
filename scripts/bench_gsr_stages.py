#!/usr/bin/env python3
"""Per-stage latency of the live GSR pipeline on one clip (idle-gated).

Main-env stages, per frame: decode, detector, pose, camera motion +
calibration propagation, tracker (McByte with shared CMC), team colour
features, and the keyframe calibrator (PnLCalib, every k frames, with the
resize done before the tensor conversion). Jersey stages run in their own env
(bench_jersey_stage.py). Prints medians/p95 per stage and the per-frame budget.

Usage:
  python scripts/bench_gsr_stages.py --seq data/raw/soccernet/gsr/valid/SNGS-021 \
      --weights runs/finetune/yolo26s_gsr_1280/weights/best.pt --frames 300 --keyframe-k 10
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark.timing import _phys_gpu_index, wait_for_idle_gpu
from calibration.propagation import HomographyPropagator
from gsr_perceive import SharedCMC
from gsr_team_eval import torso_feature


def sync():
    import torch
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pose-model", default="yolo26s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--keyframe-k", type=int, default=10)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", default="results/bench_gsr_stages.json")
    args = ap.parse_args()

    import supervision as sv
    import trackers
    from ultralytics import YOLO
    from gsr_calib_eval import PnLCalibrator

    gi = _phys_gpu_index()
    gate = wait_for_idle_gpu(gi, max_wait_s=600)
    det, pose = YOLO(args.weights), YOLO(args.pose_model)
    pnl = PnLCalibrator("/data/saurav/repos/PnLCalib", "/data/saurav/repos/calib_weights/SV_kp",
                        "/data/saurav/repos/calib_weights/SV_lines")
    paths = sorted(Path(args.seq, "img1").glob("*.jpg"))[: args.frames + args.warmup]
    trk = trackers.McByteTracker(frame_rate=25, enable_cmc=True)
    shared = SharedCMC()
    trk.cmc = shared
    prop = HomographyPropagator()
    T = {k: [] for k in ("decode", "detect", "pose", "motion_prop", "track", "team_feat", "keyframe_calib")}
    prev_boxes = None
    for i, p in enumerate(paths):
        rec = i >= args.warmup
        t = time.perf_counter(); frame = cv2.imread(str(p)); t_dec = time.perf_counter() - t
        sync(); t = time.perf_counter()
        r = det.predict(frame, imgsz=args.imgsz, conf=0.1, device=0, half=args.half, verbose=False)[0]
        sync(); t_det = time.perf_counter() - t
        t = time.perf_counter()
        pose.predict(frame, imgsz=args.imgsz, conf=0.1, device=0, half=args.half, verbose=False)
        sync(); t_pose = time.perf_counter() - t
        xyxy, conf, cls = (r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                           r.boxes.cls.cpu().numpy().astype(int))
        keep = cls != 3
        t = time.perf_counter()
        M = prop.motion(frame, prev_boxes)
        shared.H = prop.cmc_affine(M)
        if i % args.keyframe_k:
            prop.advance(M)
        t_mot = time.perf_counter() - t
        t = time.perf_counter()
        out = trk.update(sv.Detections(xyxy=xyxy[keep].astype(np.float32),
                                       confidence=conf[keep].astype(np.float32),
                                       class_id=cls[keep]), frame)
        t_trk = time.perf_counter() - t
        prev_boxes = out.xyxy
        t = time.perf_counter()
        for b in out.xyxy:
            torso_feature(frame, {"x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1]})
        t_team = time.perf_counter() - t
        t_key = 0.0
        if i % args.keyframe_k == 0:
            small = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
            sync(); t = time.perf_counter()
            H = pnl.homography(small)
            sync(); t_key = time.perf_counter() - t
            prop.keyframe(frame, H, out.xyxy)
        if rec:
            for k, v in zip(T, (t_dec, t_det, t_pose, t_mot, t_trk, t_team, t_key)):
                T[k].append(v * 1000)
    res = {"gpu_gate": gate, "frames": args.frames, "imgsz": args.imgsz, "half": args.half,
           "keyframe_k": args.keyframe_k}
    for k, v in T.items():
        v = np.asarray(v)
        if k == "keyframe_calib":
            kv = v[v > 0]
            res[k] = {"per_keyframe_p50": round(float(np.median(kv)), 2),
                      "amortised_per_frame": round(float(v.mean()), 2)}
        else:
            res[k] = {"p50": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2)}
    budget = sum(res[k]["p50"] for k in T if k != "keyframe_calib") + res["keyframe_calib"]["amortised_per_frame"]
    res["sequential_ms_per_frame_excl_jersey"] = round(budget, 2)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
