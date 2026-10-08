#!/usr/bin/env python3
"""Per-frame latency of the jersey stage (jersey env): legibility on every
player crop of the frame, PARSeq on the legible ones, as one batch each.

Usage (jersey env):
  python scripts/bench_jersey_stage.py --seq data/raw/soccernet/gsr/valid/SNGS-021 --frames 300
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_jersey_parseq import batch_tensor, crops_for_frame, load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--repo", default="/data/saurav/repos/jersey-number-pipeline")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--min-h", type=int, default=60)
    ap.add_argument("--leg-thr", type=float, default=0.3)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", default="results/bench_jersey_stage.json")
    args = ap.parse_args()
    leg, stm = load_models(args.repo, "cuda")
    if args.half:
        leg, stm = leg.half(), stm.half()
    hw = tuple(stm.hparams.img_size)
    d = json.loads((Path(args.seq) / "Labels-GameState.json").read_text())
    by_img = {}
    for a in d["annotations"]:
        if (a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper")
                and a["bbox_image"]["h"] >= args.min_h):
            by_img.setdefault(a["image_id"], []).append(a["bbox_image"])
    t_leg, t_str, n_crops, n_read = [], [], [], []
    for i, im in enumerate(d["images"][: args.frames + args.warmup]):
        frame = cv2.imread(str(Path(args.seq) / "img1" / im["file_name"]))
        crops = [c for c in crops_for_frame(frame, by_img.get(im["image_id"], [])) if c is not None]
        if not crops:
            continue
        dt = torch.float16 if args.half else torch.float32
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            p = leg(batch_tensor([c[0] for c in crops], (256, 256), "cuda", "imagenet").to(dt)).view(-1)
        torch.cuda.synchronize(); a = time.perf_counter() - t
        idx = [j for j, v in enumerate(p.float().cpu().numpy()) if v >= args.leg_thr]
        b = 0.0
        if idx:
            t = time.perf_counter()
            with torch.no_grad():
                stm(batch_tensor([crops[j][1] for j in idx], hw, "cuda", "half").to(dt))
            torch.cuda.synchronize(); b = time.perf_counter() - t
        if i >= args.warmup:
            t_leg.append(a * 1000); t_str.append(b * 1000); n_crops.append(len(crops)); n_read.append(len(idx))
    res = {"frames": len(t_leg), "half": args.half, "crops_per_frame_mean": round(float(np.mean(n_crops)), 1),
           "parseq_crops_per_frame_mean": round(float(np.mean(n_read)), 1),
           "legibility_ms_p50": round(float(np.median(t_leg)), 2),
           "parseq_ms_p50": round(float(np.median(t_str)), 2),
           "jersey_ms_per_frame_p50": round(float(np.median(np.add(t_leg, t_str))), 2),
           "jersey_ms_per_frame_p95": round(float(np.percentile(np.add(t_leg, t_str), 95)), 2)}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
