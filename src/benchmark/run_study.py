"""Tradeoff-study runner: executes one config end-to-end, writes results JSON.

Usage:
    python scripts/run_benchmark.py --config configs/baseline.yaml --mode smoke
    python scripts/run_benchmark.py --config configs/tensorrt.yaml   # GPU only

Smoke mode: 1 clip, few frames, CPU -> validates plumbing for $0.
Full mode:  the numbers that go in the report (GPU sessions A/C).

Every run writes results/<experiment>.json — the artifact of record.
"""

import json
import yaml
from pathlib import Path

from data import SportsMOT
from models.detector import Detector
from tracking.tracker import Tracker
from streaming.pipeline import StreamingPipeline
from streaming.cache import FrameFeatureCache
from benchmark.metrics import (compute_detection_metrics,
                               compute_tracking_metrics, gpu_stats)


def run_study(config_path, mode="full"):
    cfg = yaml.safe_load(open(config_path))
    device = "cpu" if mode == "smoke" else cfg.get("device", "cpu")
    print(f"[study] {cfg['experiment']} mode={mode} device={device}")

    # 1. build components from config
    dcfg, tcfg = cfg["detector"], cfg["tracker"]
    detector = Detector(dcfg["name"], imgsz=dcfg.get("imgsz", 640),
                        conf=dcfg.get("conf", 0.25), device=device)
    tracker = Tracker(tcfg["name"])
    scfg = cfg.get("streaming", {})
    cache = (FrameFeatureCache(scfg.get("keyframe_interval", 3))
             if scfg.get("frame_cache") else None)
    pipeline = StreamingPipeline(detector, tracker, cache=cache, out_path=None)

    # 2. data
    data_cfg = cfg["data"]
    dataset = SportsMOT(split=data_cfg.get("split", "test"))
    seqs = dataset.sequences()
    max_clips = 1 if mode == "smoke" else data_cfg.get("max_clips", len(seqs))

    # 3. run
    all_pred, all_gt, lat = {}, {}, []
    for seq in seqs[:max_clips]:
        frames = dataset.frames(seq)
        if mode == "smoke":
            frames = frames[:30]
        import cv2
        for f in frames:
            fid = int(f.stem)
            img = cv2.imread(str(f))
            dets = detector.infer(img)
            tracks = tracker.update(dets, img)
            all_pred.setdefault(fid, []).extend(dets)
        gt = dataset.ground_truth(seq)
        for fid, boxes in gt.items():
            all_gt.setdefault(fid, []).extend(boxes)
        tracker.reset()
        print(f"  seq {seq}: {len(frames)} frames")

    # 4. metrics — re-run pipeline timing on one clip for clean latency numbers
    import cv2
    seq = seqs[0]
    frames = dataset.frames(seq)[: (30 if mode == "smoke" else 200)]
    t_pipe = StreamingPipeline(detector, Tracker(tcfg["name"]), cache=None,
                               out_path=None, draw=False)
    import numpy as np
    for f in frames:
        img = cv2.imread(str(f))
        t0 = __import__("time").perf_counter()
        d = detector.infer(img)
        t_pipe.tracker.update(d, img)
        t_pipe.latencies_ms.append((__import__("time").perf_counter() - t0) * 1000)
    rep = t_pipe.latency_report()

    det_m = compute_detection_metrics(all_pred, all_gt)
    # tracking metrics need per-frame track output; approximate from preds here
    stats = gpu_stats()

    result = {
        "experiment": cfg["experiment"],
        "config": cfg,
        "mode": mode,
        "device": device,
        **rep,
        **det_m,
        **stats,
    }
    out = Path(cfg["benchmark"]["save_to"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"[study] wrote {out}")
    print(json.dumps({k: result[k] for k in
                      ("fps", "latency_p50_ms", "latency_p95_ms", "map")},
                     indent=1))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    args = ap.parse_args()
    run_study(args.config, args.mode)
