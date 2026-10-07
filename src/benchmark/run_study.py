"""Tradeoff-study runner: executes one config end-to-end, writes results JSON.

Usage:
    python scripts/run_benchmark.py --config configs/baseline.yaml --mode smoke --cpu
    python scripts/run_benchmark.py --config configs/tensorrt.yaml   # GPU only

Smoke mode: 1 clip, 30 frames, CPU -> validates plumbing for $0.
Full mode:  the numbers that go in the report (GPU sessions A/C).

Accuracy: HOTA/DetA/AssA/IDF1/MOTA from roboflow `trackers eval` (TrackEval-
compatible) and COCO mAP from torchmetrics (faster_coco_eval). Speed: see
benchmark/timing.py (warmup + repeats + contention flags + Zeus energy).
Every run writes the results JSON in cfg["benchmark"]["save_to"].
"""

import json
import shutil
import yaml
from pathlib import Path

import cv2

from data import SportsMOT
from models.detector import Detector
from tracking.tracker import Tracker
from benchmark.metrics import gpu_stats
from benchmark.mot_eval import (coco_map, run_trackeval, write_gt_subset,
                                write_mot_file)
from benchmark.timing import benchmark_pipeline

METRICS_VERSION = 2  # 2 = TrackEval HOTA/CLEAR + COCO mAP (v1 = home-grown)


def run_study(config_path, mode="full", force_cpu=False):
    cfg = yaml.safe_load(open(config_path))
    device = "cpu" if (mode == "smoke" or force_cpu) else cfg.get("device", "cpu")
    print(f"[study] {cfg['experiment']} mode={mode} device={device}", flush=True)

    dcfg, tcfg = cfg["detector"], cfg["tracker"]
    detector = Detector(dcfg["name"], imgsz=dcfg.get("imgsz", 640),
                        conf=dcfg.get("conf", 0.25), device=device,
                        classes=dcfg.get("classes"), iou=dcfg.get("iou"))

    data_cfg = cfg["data"]
    dataset = SportsMOT(split=data_cfg.get("split", "test"))
    seqs = dataset.sequences()
    max_clips = 1 if mode == "smoke" else data_cfg.get("max_clips") or len(seqs)
    seqs = seqs[:max_clips]

    work = Path("results/_work") / cfg["experiment"]
    shutil.rmtree(work, ignore_errors=True)
    gt_dir, trk_dir = work / "gt", work / "tracker"
    trk_dir.mkdir(parents=True)

    # ---- accuracy pass: every frame of every selected clip ----
    all_pred, all_gt = {}, {}
    for si, seq in enumerate(seqs):
        frames = dataset.frames(seq)
        if mode == "smoke":
            frames = frames[:30]
        tracker = Tracker(tcfg["name"])
        by_frame = {}
        for f in frames:
            fid = int(f.stem)
            img = cv2.imread(str(f))
            dets = detector.infer(img)
            by_frame[fid] = tracker.update(dets, img)
            all_pred[(si, fid)] = list(dets)
        last = int(frames[-1].stem)
        for fid, boxes in dataset.ground_truth(seq).items():
            if fid <= last:
                all_gt[(si, fid)] = boxes
        write_mot_file(trk_dir / f"{seq}.txt", by_frame)
        write_gt_subset(dataset.root / seq, gt_dir / seq, last)
        print(f"  seq {seq}: {len(frames)} frames", flush=True)

    trk_m = run_trackeval(gt_dir, trk_dir, seqs, work)
    det_m = coco_map(all_pred, all_gt)

    # ---- speed pass on the first clip (pre-decoded, warmup, repeats) ----
    first = dataset.frames(seqs[0])
    n_t, warm, reps = (30, 5, 2) if mode == "smoke" else (500, 50, 5)
    attempts = []
    for _ in range(3 if mode == "full" else 1):  # retry a transiently contended pass
        speed = benchmark_pipeline(detector, lambda: Tracker(tcfg["name"]), first,
                                   device, n_frames=n_t, warmup=warm, repeats=reps)
        attempts.append(speed)
        if not speed["contended"]:
            break
    speed = min(attempts, key=lambda s: (s["contended"], s["latency_repeat_cv_pct"]))
    speed["timing_attempts"] = len(attempts)

    result = {
        "experiment": cfg["experiment"], "config": cfg, "mode": mode,
        "device": device, "metrics_version": METRICS_VERSION,
        "n_clips": len(seqs), "n_frames_eval": len(all_pred),
        "n_gt": sum(len(v) for v in all_gt.values()),
        "n_pred": sum(len(v) for v in all_pred.values()),
        **speed, **trk_m,
        "map": det_m["map"], "map50": det_m["map50"], "map75": det_m["map75"],
        **gpu_stats(),
    }
    out = Path(cfg["benchmark"]["save_to"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    shutil.rmtree(work, ignore_errors=True)
    print(f"[study] wrote {out}")
    keys = ("fps", "fps_e2e_with_decode", "latency_p50_ms", "latency_p95_ms",
            "latency_repeat_cv_pct", "contended", "hota", "deta", "assa",
            "idf1", "mota", "map", "map50")
    print(json.dumps({k: result.get(k) for k in keys}, indent=1))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", default="full", choices=["full", "smoke"])
    args = ap.parse_args()
    run_study(args.config, args.mode)
