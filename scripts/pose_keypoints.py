#!/usr/bin/env python3
"""Body keypoints for SoccerNet-GSR player boxes (for pose-guided torso crops).

Runs an Ultralytics pose model on full frames, matches its people to the
ground-truth (or perception) player boxes by IoU, and stores the 17 COCO
keypoints per box.

Usage:
  python scripts/pose_keypoints.py --gt data/raw/soccernet/gsr --split valid \
      --seqs SNGS-021 ... --model yolo26s-pose.pt --out results/_work/pose_subset.npz
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def iou_matrix(A, B):
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); b = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (a[:, None] + b[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="data/raw/soccernet/gsr")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--seqs", nargs="+", default=None)
    ap.add_argument("--model", default="yolo26s-pose.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-h", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--perception", default="", help="use boxes from a gsr_perceive.py dir")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    seqs = sorted(p for p in (Path(args.gt) / args.split).iterdir()
                  if (p / "Labels-GameState.json").exists()
                  and (not args.seqs or p.name in args.seqs))
    si, sn = map(int, args.shard.split("/"))
    seqs = seqs[si::sn]
    ids, kps = [], []
    for s in seqs:
        d = json.loads((s / "Labels-GameState.json").read_text())
        by_img = {}
        if args.perception:
            pr = json.loads((Path(args.perception) / f"{s.name}.json").read_text())
            for i, r in enumerate(pr["rows"]):
                x, y, w, h = r["bbox"]
                if r["cls"] in ("player", "goalkeeper") and h >= args.min_h:
                    by_img.setdefault(r["image_id"], []).append(
                        {"id": f"{s.name}:{i}", "bbox_image": {"x": x, "y": y, "w": w, "h": h}})
        else:
            for a in d["annotations"]:
                if (a.get("supercategory") == "object" and a["attributes"]["role"] in ("player", "goalkeeper")
                        and a["bbox_image"]["h"] >= args.min_h):
                    by_img.setdefault(a["image_id"], []).append(a)
        n = 0
        for im in d["images"]:
            anns = by_img.get(im["image_id"])
            if not anns:
                continue
            r = model.predict(cv2.imread(str(s / "img1" / im["file_name"])), imgsz=args.imgsz,
                              conf=0.1, device=args.device, verbose=False)[0]
            if r.keypoints is None or len(r.boxes) == 0:
                continue
            P = r.boxes.xyxy.cpu().numpy()
            K = r.keypoints.data.cpu().numpy()                  # (n, 17, 3)
            G = np.array([[a["bbox_image"]["x"], a["bbox_image"]["y"],
                           a["bbox_image"]["x"] + a["bbox_image"]["w"],
                           a["bbox_image"]["y"] + a["bbox_image"]["h"]] for a in anns])
            iou = iou_matrix(G, P)
            for gi, a in enumerate(anns):
                pj = int(iou[gi].argmax())
                if iou[gi, pj] >= 0.5:
                    ids.append(a["id"])
                    kps.append(K[pj])
                    n += 1
        print(f"  {s.name}: keypoints for {n} boxes", flush=True)
    np.savez(args.out, ann_id=np.asarray(ids), kps=np.asarray(kps, np.float32))
    print(f"[pose] wrote {args.out}")


if __name__ == "__main__":
    main()
