#!/usr/bin/env python3
"""Live streaming runner: measures sustained FPS and per-frame latency of the
GSR perception pipeline with CPU and GPU stages overlapped.

Threads:
  gpu   : decode -> detector -> pose          (TensorRT engines or PyTorch)
  cpu   : camera motion + calibration propagation -> tracker (shared CMC) ->
          team colour features; requests keyframe calibrations
  calib : accurate calibrator (PnLCalib) on keyframes, asynchronous; a late
          result is carried to the current frame with the stored motions
The jersey reader lives in another env and is load-tested concurrently by
bench_jersey_stage.py; the system's sustained rate is the slower of the two.

Usage:
  python scripts/live_runner.py --seq data/raw/soccernet/gsr/valid/SNGS-021 \
      --det engines/det_yolo26s_1280_fp16.engine --pose engines/pose_yolo26s_1280_fp16.engine
"""
import argparse
import json
import queue
import sys
import threading
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

STOP = object()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--det", required=True)
    ap.add_argument("--pose", default="")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--keyframe-k", type=int, default=10)
    ap.add_argument("--frames", type=int, default=750)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="results/live_runner.json")
    args = ap.parse_args()

    import supervision as sv
    import trackers
    import torch
    from ultralytics import YOLO
    from gsr_calib_eval import PnLCalibrator

    gate = wait_for_idle_gpu(_phys_gpu_index(), max_wait_s=600)
    det = YOLO(args.det, task="detect")
    pose = YOLO(args.pose, task="pose") if args.pose else None
    pnl = PnLCalibrator("/data/saurav/repos/PnLCalib", "/data/saurav/repos/calib_weights/SV_kp",
                        "/data/saurav/repos/calib_weights/SV_lines")
    paths = sorted(Path(args.seq, "img1").glob("*.jpg"))[: args.frames]
    blank = cv2.imread(str(paths[0]))
    for _ in range(10):                                   # engine / cudnn warm-up
        det.predict(blank, imgsz=args.imgsz, device=0, verbose=False)
        if pose:
            pose.predict(blank, imgsz=args.imgsz, device=0, verbose=False)
        pnl.homography(cv2.resize(blank, (960, 540)))

    q_cpu, q_cal = queue.Queue(maxsize=8), queue.Queue(maxsize=4)
    t_in, t_out = {}, {}
    cal_results, lock = {}, threading.Lock()

    def gpu_stage():
        for i, p in enumerate(paths):
            t_in[i] = time.perf_counter()
            frame = cv2.imread(str(p))
            r = det.predict(frame, imgsz=args.imgsz, conf=0.1, device=0, verbose=False)[0]
            if pose:
                pose.predict(frame, imgsz=args.imgsz, conf=0.1, device=0, verbose=False)
            q_cpu.put((i, frame, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                       r.boxes.cls.cpu().numpy().astype(int)))
        q_cpu.put(STOP)

    def calib_stage():
        while True:
            item = q_cal.get()
            if item is STOP:
                return
            i, small = item
            H = pnl.homography(small)
            with lock:
                cal_results[i] = H

    def cpu_stage():
        trk = trackers.McByteTracker(frame_rate=25, enable_cmc=True)
        shared = SharedCMC()
        trk.cmc = shared
        prop = HomographyPropagator()
        motions, H_cur, H_from, prev_boxes = {}, None, None, None
        while True:
            item = q_cpu.get()
            if item is STOP:
                q_cal.put(STOP)
                return
            i, frame, xyxy, conf, cls = item
            M = prop.motion(frame, prev_boxes)
            motions[i] = M
            shared.H = prop.cmc_affine(M)
            keep = cls != 3
            out = trk.update(sv.Detections(xyxy=xyxy[keep].astype(np.float32),
                                           confidence=conf[keep].astype(np.float32),
                                           class_id=cls[keep]), frame)
            prev_boxes = out.xyxy
            for b in out.xyxy:
                torso_feature(frame, {"x": b[0], "y": b[1], "w": b[2] - b[0], "h": b[3] - b[1]})
            if i % args.keyframe_k == 0:
                try:
                    q_cal.put_nowait((i, cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)))
                except queue.Full:
                    pass
            # carry the current calibration one step (frame i-1 -> i) ...
            if H_cur is not None:
                H_cur = H_cur @ M if M is not None else None
            # ... unless a newer keyframe finished: rebuild it, chaining frame k -> i
            with lock:
                done = [k for k, H in cal_results.items()
                        if H is not None and (H_from is None or k > H_from)]
                k = max(done) if done else None
                Hk = cal_results[k] if done else None
            if k is not None:
                H = Hk
                for j in range(k + 1, i + 1):
                    if motions.get(j) is None:
                        H = None
                        break
                    H = H @ motions[j]
                if H is not None:
                    H_cur, H_from = H, k
            motions.pop(i - 4 * args.keyframe_k, None)
            t_out[i] = time.perf_counter()

    threads = [threading.Thread(target=f) for f in (gpu_stage, cpu_stage, calib_stage)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    idx = [i for i in sorted(t_out) if i >= args.warmup]
    wall = t_out[idx[-1]] - t_out[idx[0]]
    lat = np.array([(t_out[i] - t_in[i]) * 1000 for i in idx])
    res = {"gpu_gate": gate, "seq": Path(args.seq).name, "frames": len(idx), "det": args.det,
           "pose": args.pose or None, "keyframe_k": args.keyframe_k,
           "sustained_fps": round((len(idx) - 1) / wall, 2),
           "latency_ms_p50": round(float(np.median(lat)), 1),
           "latency_ms_p95": round(float(np.percentile(lat, 95)), 1),
           "keyframes_calibrated": len(cal_results),
           "total_wall_s": round(time.perf_counter() - t0, 1)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
