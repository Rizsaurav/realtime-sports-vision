#!/usr/bin/env python3
"""Torso crops + jersey labels from SoccerNet-GSR for a two-digit classifier.

Label per crop = the track's jersey number (GSR stores it per tracklet), split
into (tens, units): "7" -> (10, 7), "23" -> (2, 3); class 10 = no digit.
Tracks whose jersey is null (never readable) give (10, 10) = "no number".
These labels are noisy per frame (the number is not visible in every frame);
the online per-track vote at inference time is what absorbs that.

Usage:
  python scripts/prepare_jersey_crops.py --gt data/raw/soccernet/gsr \
      --splits train valid --stride 3 --min-h 40 --out data/processed/jersey
Writes <out>/<split>.npz: X uint8[N,64,64,3], y int64[N,2], seq, frame, track, ann_id
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

S = 64


def crop(frame, b):
    x, y, w, h = b["x"], b["y"], b["w"], b["h"]
    x1, x2 = int(max(0, x - 0.05 * w)), int(min(frame.shape[1], x + 1.05 * w))
    y1, y2 = int(max(0, y + 0.08 * h)), int(min(frame.shape[0], y + 0.62 * h))
    c = frame[y1:y2, x1:x2]
    return None if c.size == 0 else cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA)


def label(j):
    if j is None or not str(j).isdigit():
        return 10, 10
    n = int(j)
    return (10, n) if n < 10 else (n // 10, n % 10)


def one_seq(args):
    seq_dir, stride, min_h, keep_null = args
    seq_dir = Path(seq_dir)
    cv2.setNumThreads(1)
    d = json.loads((seq_dir / "Labels-GameState.json").read_text())
    by_img = {}
    for a in d["annotations"]:
        if (a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper")
                and a["bbox_image"]["h"] >= min_h):
            by_img.setdefault(a["image_id"], []).append(a)
    X, y, meta = [], [], []
    for fi, im in enumerate(d["images"]):
        if fi % stride or im["image_id"] not in by_img:
            continue
        frame = cv2.imread(str(seq_dir / "img1" / im["file_name"]))
        for a in by_img[im["image_id"]]:
            lab = label(a["attributes"]["jersey"])
            if lab == (10, 10) and not keep_null:
                continue
            c = crop(frame, a["bbox_image"])
            if c is not None:
                X.append(c)
                y.append(lab)
                meta.append((seq_dir.name, fi, a["track_id"], a["id"]))
    return X, y, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--min-h", type=int, default=40)
    ap.add_argument("--out", default="data/processed/jersey")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        seqs = sorted(p for p in (Path(args.gt) / split).iterdir()
                      if (p / "Labels-GameState.json").exists())
        stride = args.stride if split == "train" else 1
        with Pool(args.workers) as pool:
            parts = pool.map(one_seq, [(str(s), stride, args.min_h, True) for s in seqs])
        X = np.stack([c for p in parts for c in p[0]])
        y = np.asarray([l for p in parts for l in p[1]], np.int64)
        meta = [m for p in parts for m in p[2]]
        np.savez(out / f"{split}.npz", X=X, y=y,
                 seq=np.asarray([m[0] for m in meta]), frame=np.asarray([m[1] for m in meta]),
                 track=np.asarray([m[2] for m in meta]), ann_id=np.asarray([m[3] for m in meta]))
        has = (y != 10).any(1).mean()
        print(f"{split}: {len(X)} crops from {len(seqs)} clips, {100 * has:.1f}% with a number")


if __name__ == "__main__":
    main()
