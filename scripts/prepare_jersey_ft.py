#!/usr/bin/env python3
"""Weakly-labelled jersey crops from SoccerNet-GSR for fine-tuning the reader.

For every player/goalkeeper box whose track has a jersey number (GSR labels
jerseys per tracklet), cut the shoulder-to-hip torso (pose keypoints, box-band
fallback) and keep it only if the legibility model says the number is
readable. Label = the track's number. Runs in the jersey env.

Usage:
  python scripts/prepare_jersey_ft.py --split train --pose results/pose_gt_train.npz \
      --stride 2 --leg-thr 0.5 --out data/processed/jersey_ft
Writes <out>/images/*.jpg and <out>/labels.tsv (file, label, seq, legibility).
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_jersey_parseq import batch_tensor, crops_for_frame, load_models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="train")
    ap.add_argument("--repo", default="/data/saurav/repos/jersey-number-pipeline")
    ap.add_argument("--pose", required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-h", type=int, default=60)
    ap.add_argument("--leg-thr", type=float, default=0.5)
    ap.add_argument("--out", default="data/processed/jersey_ft")
    args = ap.parse_args()

    leg, _ = load_models(args.repo, "cuda")
    pz = np.load(args.pose)
    pose = dict(zip(pz["ann_id"].tolist(), pz["kps"]))
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rows, pool = [], ThreadPoolExecutor(16)
    for s in sorted(p for p in (Path(args.gt) / args.split).iterdir()
                    if (p / "Labels-GameState.json").exists()):
        d = json.loads((s / "Labels-GameState.json").read_text())
        by_img = {}
        for a in d["annotations"]:
            at = a.get("attributes", {})
            if (a.get("supercategory") == "object" and at.get("role") in ("player", "goalkeeper")
                    and at.get("jersey") is not None and a["bbox_image"]["h"] >= args.min_h):
                by_img.setdefault(a["image_id"], []).append(a)
        ims = [(fi, im) for fi, im in enumerate(d["images"])
               if fi % args.stride == 0 and im["image_id"] in by_img]
        frames = pool.map(lambda x: cv2.imread(str(s / "img1" / x[1]["file_name"])), ims)
        meta, whole, torso = [], [], []
        for (fi, im), fr in zip(ims, frames):
            anns = by_img[im["image_id"]]
            crops = crops_for_frame(fr, [a["bbox_image"] for a in anns],
                                    kps=[pose.get(a["id"]) for a in anns])
            for a, c in zip(anns, crops):
                if c is not None:
                    meta.append((fi, a["id"], str(a["attributes"]["jersey"])))
                    whole.append(c[0])
                    torso.append(c[1])
        legs = np.zeros(len(meta), np.float32)
        with torch.no_grad():
            for i in range(0, len(meta), 512):
                legs[i:i + 512] = leg(batch_tensor(whole[i:i + 512], (256, 256), "cuda",
                                                   "imagenet")).view(-1).cpu().numpy()
        kept = 0
        for (fi, aid, lab), l, t in zip(meta, legs, torso):
            if l < args.leg_thr:
                continue
            name = f"{s.name}_{fi:04d}_{aid}.jpg"
            cv2.imwrite(str(out / "images" / name), cv2.cvtColor(t, cv2.COLOR_RGB2BGR))
            rows.append(f"{name}\t{lab}\t{s.name}\t{l:.3f}")
            kept += 1
        print(f"  {s.name}: {len(meta)} numbered crops, {kept} legible", flush=True)
    (out / "labels.tsv").write_text("\n".join(rows) + "\n")
    print(f"[jersey-ft] {len(rows)} crops -> {out}")


if __name__ == "__main__":
    main()
