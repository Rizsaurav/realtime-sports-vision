#!/usr/bin/env python3
"""Run a detector once over a split and cache raw detections per sequence.

Detections are stored at a low confidence floor (default 0.05) so trackers
that use low-score boxes (ByteTrack's second association) can be swept
offline on CPU without re-running the GPU.

Usage:
  python scripts/dump_detections.py --config configs/baseline_yolo26s_ft.yaml \
      --conf 0.05 --out results/dets/yolo26s_ft
Writes <out>/<seq>.npz (fid int32[N], det float32[N,5] = x1,y1,x2,y2,score)
and <out>/meta.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data import SportsMOT
from models.detector import Detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--root", default="data/raw/sportsmot", help="MOT-layout dataset root")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d = cfg["detector"]
    det = Detector(d["name"], imgsz=d.get("imgsz", 640), conf=args.conf,
                   device=args.device, classes=d.get("classes"), iou=d.get("iou"),
                   weights=d.get("weights"))
    split = args.split or cfg["data"].get("split", "val")
    ds = SportsMOT(root=args.root, split=split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0, n = time.time(), 0
    seqs = ds.sequences()[: args.max_seqs or None]
    for seq in seqs:
        fids, rows = [], []
        for f in ds.frames(seq)[: args.max_frames or None]:
            fid = int(f.stem)
            for x1, y1, x2, y2, s, _ in det.infer(cv2.imread(str(f))):
                fids.append(fid)
                rows.append((x1, y1, x2, y2, s))
            n += 1
        np.savez_compressed(out / f"{seq}.npz", fid=np.asarray(fids, np.int32),
                            det=np.asarray(rows, np.float32).reshape(-1, 5))
        print(f"  {seq}: {len(rows)} dets", flush=True)
    (out / "meta.json").write_text(json.dumps({
        "config": args.config, "experiment": cfg["experiment"], "split": split, "root": args.root,
        "conf_floor": args.conf, "frames": n, "seconds": round(time.time() - t0, 1),
        "detector": d}, indent=1))
    print(f"[dump] {n} frames -> {out}")


if __name__ == "__main__":
    main()
