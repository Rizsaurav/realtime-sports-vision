#!/usr/bin/env python3
"""Convert SportsMOT (MOT format) to an Ultralytics detection dataset.

One class ("player"). Frames are symlinked, not copied. Training splits are
subsampled with --stride because consecutive frames are near-duplicates.

Usage:
  python scripts/prepare_sportsmot_yolo.py --out data/processed/sportsmot_yolo \
      --train-stride 3 --val-stride 10
"""
import argparse
from pathlib import Path

import yaml

ROOT = Path("data/raw/sportsmot")


def img_size(seq_dir):
    info = {}
    for line in (seq_dir / "seqinfo.ini").read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return int(info["imWidth"]), int(info["imHeight"])


def convert_split(split, out, stride, min_px=4):
    n_img = n_box = 0
    img_out, lbl_out = out / "images" / split, out / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for seq_dir in sorted(p for p in (ROOT / split).iterdir() if p.is_dir()):
        W, H = img_size(seq_dir)
        by_frame = {}
        for line in (seq_dir / "gt" / "gt.txt").read_text().splitlines():
            p = line.split(",")
            if len(p) < 6:
                continue
            fid = int(float(p[0]))
            x1, y1 = max(0.0, float(p[2])), max(0.0, float(p[3]))
            x2, y2 = min(W, float(p[2]) + float(p[4])), min(H, float(p[3]) + float(p[5]))
            if x2 - x1 < min_px or y2 - y1 < min_px:
                continue
            by_frame.setdefault(fid, []).append(
                f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        for img in sorted((seq_dir / "img1").glob("*.jpg")):
            fid = int(img.stem)
            if (fid - 1) % stride or fid not in by_frame:
                continue
            name = f"{seq_dir.name}_{img.stem}"
            dst = img_out / f"{name}.jpg"
            if not dst.exists():
                dst.symlink_to(img.resolve())
            (lbl_out / f"{name}.txt").write_text("\n".join(by_frame[fid]) + "\n")
            n_img += 1
            n_box += len(by_frame[fid])
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/sportsmot_yolo")
    ap.add_argument("--train-stride", type=int, default=3)
    ap.add_argument("--val-stride", type=int, default=10)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    stats = {s: convert_split(s, out, st) for s, st in
             (("train", args.train_stride), ("val", args.val_stride))}
    (out / "data.yaml").write_text(yaml.safe_dump({
        "path": str(out), "train": "images/train", "val": "images/val",
        "names": {0: "player"}}))
    for s, (ni, nb) in stats.items():
        print(f"{s}: {ni} images, {nb} boxes")
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
